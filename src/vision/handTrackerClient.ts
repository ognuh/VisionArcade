import type { HandLandmarker } from '@mediapipe/tasks-vision';
import type { AiEngineMode, GpuInfo, Landmark3D } from '../types/vision';
import { createHandTracker } from './handTracker';

export interface TrackingResultPacket {
  landmarks: Landmark3D[][];
  aiFps: number;
  inferMs: number;
  delegate: 'GPU' | 'CPU';
}

const AI_W = 320;
const AI_H = 180;
const FPS_WINDOW_SIZE = 20;

/**
 * Direct Zero-Copy HandTracker Client (320x180 Hardware Resize + Latest-Frame 1-Slot Queue):
 * - Captures hardware-scaled 320x180 ImageBitmap on main thread and transfers zero-copy to Worker.
 * - Latest-Frame Slot prevents the 30 -> 15 FPS halving when Worker is briefly busy: keeps only
 *   the newest ImageBitmap (closing superseded ones immediately -> 0 leaks) and fires it the
 *   instant the Worker returns a result.
 * - Supports Auto-Benchmark Delegate (GPU -> CPU WASM SIMD fallback when GPU queue saturates at 120 FPS)
 *   plus manual AUTO / GPU / CPU override.
 */
export class HandTrackerClient {
  private worker: Worker | null = null;
  private fallbackLandmarker: HandLandmarker | null = null;
  private fallbackCanvas: HTMLCanvasElement | null = null;
  private fallbackCtx: CanvasRenderingContext2D | null = null;

  private isProcessing = false;
  private isCapturing = false;
  private disposed = false;

  // Latest-Frame Slot (depth 1): holds at most one newest waiting ImageBitmap
  private pendingBitmap: ImageBitmap | null = null;
  private pendingTs = 0;

  private engineMode: AiEngineMode = 'AUTO';
  private activeDelegate: 'GPU' | 'CPU' = 'GPU';
  private avgInferMs = 0;
  private lastDetectTs = 0;

  // Dedicated Worker completion timestamp ring buffer for accurate AI FPS readout
  private completionTimes = new Float64Array(FPS_WINDOW_SIZE);
  private completionCursor = 0;
  private completionCount = 0;

  public gpuInfo: GpuInfo = {
    delegate: 'GPU',
    renderer: 'Initializing...',
    tier: 'UNKNOWN',
  };

  private onResultCallback: ((packet: TrackingResultPacket) => void) | null = null;
  private onDelegateChangeCallback: ((delegate: 'GPU' | 'CPU') => void) | null = null;

  async init(
    onResult: (packet: TrackingResultPacket) => void,
    initialEngineMode: AiEngineMode = 'AUTO',
    onDelegateChange?: (delegate: 'GPU' | 'CPU') => void,
  ): Promise<GpuInfo> {
    this.onResultCallback = onResult;
    this.onDelegateChangeCallback = onDelegateChange ?? null;
    this.engineMode = initialEngineMode;

    const canUseWorker =
      typeof Worker !== 'undefined' &&
      typeof OffscreenCanvas !== 'undefined' &&
      typeof createImageBitmap === 'function';

    if (canUseWorker) {
      try {
        const gpuInfo = await this.initWorker(initialEngineMode);
        this.gpuInfo = gpuInfo;
        return gpuInfo;
      } catch (err) {
        console.warn('Web Worker HandLandmarker unavailable, falling back to main thread:', err);
        this.worker?.terminate();
        this.worker = null;
      }
    }

    // Automatic Main-Thread fallback
    const tracker = await createHandTracker(false);
    if (this.disposed) {
      tracker.landmarker.close();
      return tracker.gpuInfo;
    }
    this.fallbackLandmarker = tracker.landmarker;
    this.fallbackCanvas = document.createElement('canvas');
    this.fallbackCanvas.width = AI_W;
    this.fallbackCanvas.height = AI_H;
    this.fallbackCtx = this.fallbackCanvas.getContext('2d');
    this.gpuInfo = tracker.gpuInfo;
    this.activeDelegate = tracker.gpuInfo.delegate;
    this.onDelegateChangeCallback?.(this.activeDelegate);
    return tracker.gpuInfo;
  }

  public setEngineMode(mode: AiEngineMode): void {
    if (this.engineMode === mode) return;
    this.engineMode = mode;
    if (this.worker && !this.disposed) {
      this.worker.postMessage({ type: 'SET_ENGINE', engineMode: mode });
    }
  }

  public getTelemetry(): {
    isWorker: boolean;
    delegate: 'GPU' | 'CPU';
    avgInferMs: number;
    aiFps: number;
  } {
    return {
      isWorker: this.worker !== null,
      delegate: this.activeDelegate,
      avgInferMs: Math.round(this.avgInferMs * 10) / 10,
      aiFps: this.computeAiFps(),
    };
  }

  private recordCompletion(nowMs: number, inferMs: number): number {
    this.completionTimes[this.completionCursor] = nowMs;
    this.completionCursor = (this.completionCursor + 1) % FPS_WINDOW_SIZE;
    if (this.completionCount < FPS_WINDOW_SIZE) {
      this.completionCount += 1;
    }
    if (inferMs > 0) {
      this.avgInferMs = this.avgInferMs > 0 ? this.avgInferMs * 0.85 + inferMs * 0.15 : inferMs;
    }
    return this.computeAiFps();
  }

  private computeAiFps(): number {
    if (this.completionCount < 2) return 30;
    const newestIdx = (this.completionCursor - 1 + FPS_WINDOW_SIZE) % FPS_WINDOW_SIZE;
    const oldestIdx =
      this.completionCount < FPS_WINDOW_SIZE ? 0 : this.completionCursor;
    const spanMs = this.completionTimes[newestIdx] - this.completionTimes[oldestIdx];
    if (spanMs <= 1) return 30;
    return Math.min(60, Math.round(((this.completionCount - 1) * 1000) / spanMs));
  }

  private dispatchToWorker(bitmap: ImageBitmap, timestampMs: number): void {
    if (this.disposed || !this.worker) {
      try {
        bitmap.close();
      } catch {
        // ignore
      }
      this.isProcessing = false;
      return;
    }
    this.isProcessing = true;
    this.worker.postMessage({ type: 'DETECT', bitmap, timestampMs }, [bitmap]);
  }

  private flushPendingToWorker(): void {
    if (this.disposed || !this.worker) {
      this.clearPendingBitmap();
      this.isProcessing = false;
      return;
    }
    if (this.pendingBitmap) {
      const nextBmp = this.pendingBitmap;
      const nextTs = this.pendingTs;
      this.pendingBitmap = null;
      this.dispatchToWorker(nextBmp, nextTs);
    } else {
      this.isProcessing = false;
    }
  }

  private clearPendingBitmap(): void {
    if (this.pendingBitmap) {
      try {
        this.pendingBitmap.close();
      } catch {
        // ignore
      }
      this.pendingBitmap = null;
    }
  }

  private initWorker(initialMode: AiEngineMode): Promise<GpuInfo> {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL('../workers/handTracker.worker.ts', import.meta.url), {
        type: 'module',
      });
      this.worker = worker;

      const timeoutId = window.setTimeout(() => {
        reject(new Error('Worker MediaPipe init timed out'));
      }, 15000);

      worker.onerror = (ev) => {
        window.clearTimeout(timeoutId);
        reject(new Error(ev.message || 'Worker runtime error'));
      };

      worker.onmessage = (e: MessageEvent) => {
        const msg = e.data;
        if (msg.type === 'READY') {
          window.clearTimeout(timeoutId);
          this.activeDelegate = (msg.delegate ?? msg.gpuInfo?.delegate ?? 'GPU') as 'GPU' | 'CPU';
          this.onDelegateChangeCallback?.(this.activeDelegate);
          resolve(msg.gpuInfo as GpuInfo);
          return;
        }
        if (msg.type === 'DELEGATE_CHANGED') {
          this.activeDelegate = (msg.delegate ?? 'CPU') as 'GPU' | 'CPU';
          this.onDelegateChangeCallback?.(this.activeDelegate);
          return;
        }
        if (msg.type === 'ERROR') {
          window.clearTimeout(timeoutId);
          reject(new Error(String(msg.message || 'Worker init error')));
          return;
        }
        if (msg.type === 'RESULT') {
          if (this.disposed) {
            this.clearPendingBitmap();
            this.isProcessing = false;
            return;
          }

          const now = performance.now();
          const inferMs = Number(msg.inferMs ?? 0);
          const delegate = (msg.delegate ?? this.activeDelegate) as 'GPU' | 'CPU';
          this.activeDelegate = delegate;
          const aiFps = this.recordCompletion(now, inferMs);

          // Immediately fire the newest queued frame BEFORE invoking main-thread callbacks
          // so Worker stays saturated at full 30 FPS cadence without half-stepping!
          this.flushPendingToWorker();

          this.onResultCallback?.({
            landmarks: (msg.landmarks ?? []) as Landmark3D[][],
            aiFps,
            inferMs,
            delegate,
          });
        }
      };

      worker.postMessage({ type: 'INIT', engineMode: initialMode });
    });
  }

  /**
   * Extracts a hardware-resized 320x180 ImageBitmap and either dispatches immediately
   * or replaces the single Latest-Frame Slot (closing any older waiting ImageBitmap).
   */
  submitFrame(video: HTMLVideoElement, timestampMs: number): void {
    if (this.disposed || video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }

    if (this.worker) {
      if (this.isCapturing) return;
      this.isCapturing = true;

      createImageBitmap(video, {
        resizeWidth: AI_W,
        resizeHeight: AI_H,
        resizeQuality: 'low',
      })
        .then((bitmap) => {
          this.isCapturing = false;
          if (this.disposed || !this.worker) {
            try {
              bitmap.close();
            } catch {
              // ignore
            }
            return;
          }

          if (!this.isProcessing) {
            this.dispatchToWorker(bitmap, timestampMs);
          } else {
            // Latest-Frame Slot: close stale waiting bitmap, keep only the newest frame
            this.clearPendingBitmap();
            this.pendingBitmap = bitmap;
            this.pendingTs = timestampMs;
          }
        })
        .catch(() => {
          this.isCapturing = false;
        });
      return;
    }

    // Main-thread fallback
    if (this.fallbackLandmarker && this.fallbackCanvas && this.fallbackCtx && !this.isProcessing) {
      this.isProcessing = true;
      window.setTimeout(() => {
        if (this.disposed || !this.fallbackLandmarker || !this.fallbackCanvas || !this.fallbackCtx) {
          this.isProcessing = false;
          return;
        }
        try {
          this.fallbackCtx.drawImage(video, 0, 0, AI_W, AI_H);
          const now = performance.now();
          const ts = Math.max(this.lastDetectTs + 1, Math.round(now));
          this.lastDetectTs = ts;
          const t0 = performance.now();
          const result = this.fallbackLandmarker.detectForVideo(this.fallbackCanvas, ts);
          const inferMs = performance.now() - t0;
          const aiFps = this.recordCompletion(now, inferMs);

          this.onResultCallback?.({
            landmarks: (result.landmarks ?? []) as Landmark3D[][],
            aiFps,
            inferMs,
            delegate: this.activeDelegate,
          });
        } finally {
          this.isProcessing = false;
        }
      }, 0);
    }
  }

  dispose(): void {
    this.disposed = true;
    this.onResultCallback = null;
    this.onDelegateChangeCallback = null;
    this.clearPendingBitmap();
    if (this.worker) {
      try {
        this.worker.postMessage({ type: 'CLOSE' });
      } catch {
        // ignore
      }
      this.worker.terminate();
      this.worker = null;
    }
    if (this.fallbackLandmarker) {
      this.fallbackLandmarker.close();
      this.fallbackLandmarker = null;
    }
  }
}
