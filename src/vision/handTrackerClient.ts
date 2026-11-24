import type { HandLandmarker } from '@mediapipe/tasks-vision';
import type { AiEngineMode, GpuInfo, Landmark3D, TrackerTaskMode } from '../types/vision';
import { createHandTracker } from './handTracker';

export interface TrackingResultPacket {
  taskMode: TrackerTaskMode;
  landmarks: Landmark3D[][];
  poseLandmarks: Landmark3D[] | null;
  aiFps: number;
  inferMs: number;
  delegate: 'GPU' | 'CPU';
}

const AI_W = 320;
const AI_H = 180;
const FPS_WINDOW_SIZE = 20;

/**
 * Direct Zero-Copy Unified Hand / Pose Tracker Client (320x180 Hardware Resize + Latest-Frame 1-Slot Queue):
 * - Mutually exclusive taskMode: 'HAND' for VFX modes, 'POSE' (pose_landmarker_lite) for Runner mode.
 * - Captures hardware-scaled 320x180 ImageBitmap on main thread and transfers zero-copy to Worker.
 * - Latest-Frame Slot eliminates 30 -> 15 FPS halving and strictly closes superseded bitmaps (0 leaks).
 */
export class HandTrackerClient {
  private worker: Worker | null = null;
  private fallbackLandmarker: HandLandmarker | null = null;
  private fallbackCanvas: HTMLCanvasElement | null = null;
  private fallbackCtx: CanvasRenderingContext2D | null = null;

  private isProcessing = false;
  private isCapturing = false;
  private disposed = false;

  private pendingBitmap: ImageBitmap | null = null;
  private pendingTs = 0;

  private engineMode: AiEngineMode = 'GPU';
  private taskMode: TrackerTaskMode = 'HAND';
  private activeDelegate: 'GPU' | 'CPU' = 'GPU';
  private avgInferMs = 0;
  private lastDetectTs = 0;

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
    initialEngineMode: AiEngineMode = 'GPU',
    onDelegateChange?: (delegate: 'GPU' | 'CPU') => void,
    initialTaskMode: TrackerTaskMode = 'HAND',
  ): Promise<GpuInfo> {
    this.onResultCallback = onResult;
    this.onDelegateChangeCallback = onDelegateChange ?? null;
    this.engineMode = initialEngineMode;
    this.taskMode = initialTaskMode;

    const canUseWorker =
      typeof Worker !== 'undefined' &&
      typeof OffscreenCanvas !== 'undefined' &&
      typeof createImageBitmap === 'function';

    if (canUseWorker) {
      try {
        const gpuInfo = await this.initWorker(initialEngineMode, initialTaskMode);
        this.gpuInfo = gpuInfo;
        return gpuInfo;
      } catch (err) {
        console.warn('Web Worker HandLandmarker unavailable, falling back to main thread:', err);
        this.worker?.terminate();
        this.worker = null;
      }
    }

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

  public setTaskMode(mode: TrackerTaskMode): void {
    if (this.taskMode === mode) return;
    this.taskMode = mode;
    if (this.worker && !this.disposed) {
      this.worker.postMessage({ type: 'SET_TASK_MODE', taskMode: mode });
    }
  }

  public getTelemetry(): {
    isWorker: boolean;
    taskMode: TrackerTaskMode;
    delegate: 'GPU' | 'CPU';
    avgInferMs: number;
    aiFps: number;
  } {
    return {
      isWorker: this.worker !== null,
      taskMode: this.taskMode,
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
    const oldestIdx = this.completionCount < FPS_WINDOW_SIZE ? 0 : this.completionCursor;
    const spanMs = this.completionTimes[newestIdx] - this.completionTimes[oldestIdx];
    if (spanMs <= 1) return 30;
    return Math.min(120, Math.round(((this.completionCount - 1) * 1000) / spanMs));
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

  private initWorker(
    initialMode: AiEngineMode,
    initialTask: TrackerTaskMode,
  ): Promise<GpuInfo> {
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
          const pktTask = (msg.taskMode ?? this.taskMode) as TrackerTaskMode;
          this.activeDelegate = delegate;
          const aiFps = this.recordCompletion(now, inferMs);

          this.flushPendingToWorker();

          this.onResultCallback?.({
            taskMode: pktTask,
            landmarks: (msg.landmarks ?? []) as Landmark3D[][],
            poseLandmarks: (msg.poseLandmarks ?? null) as Landmark3D[] | null,
            aiFps,
            inferMs,
            delegate,
          });
        }
      };

      worker.postMessage({ type: 'INIT', engineMode: initialMode, taskMode: initialTask });
    });
  }

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
            taskMode: 'HAND',
            landmarks: (result.landmarks ?? []) as Landmark3D[][],
            poseLandmarks: null,
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
