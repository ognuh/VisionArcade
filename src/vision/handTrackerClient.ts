import type { HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import type {
  AiEngineMode,
  GpuInfo,
  Landmark3D,
  MaxHandsOption,
  TrackerTaskMode,
} from '../types/vision';
import { createMainThreadTracker } from './handTracker';

export interface TrackingResultPacket {
  taskMode: TrackerTaskMode;
  landmarks: Landmark3D[][];
  poseLandmarks: Landmark3D[] | null;
  aiFps: number;
  inferMs: number;
  avgInferMs: number;
  delegate: 'GPU' | 'CPU';
}

const AI_W = 256;
const AI_H = 144;
const FPS_WINDOW_SIZE = 20;

/**
 * Clean Unthrottled Zero-Copy GPU Tracker Client (256x144):
 * - Simple inFlight lock + 1-slot latestBitmap queue synced directly to requestVideoFrameCallback.
 * - Immediately closes every ImageBitmap in finally and whenever superseded (0 VRAM leaks).
 * - Locked GPU delegate by default; zero artificial idle throttling or backpressure frame-skipping.
 */
export class HandTrackerClient {
  private worker: Worker | null = null;
  private mainHandLandmarker: HandLandmarker | null = null;
  private mainPoseLandmarker: PoseLandmarker | null = null;
  private mainCanvas: HTMLCanvasElement | null = null;
  private mainCtx: CanvasRenderingContext2D | null = null;

  private isProcessing = false;
  private isCapturing = false;
  private disposed = false;

  private pendingBitmap: ImageBitmap | null = null;
  private pendingTs = 0;

  private engineMode: AiEngineMode = 'GPU';
  private taskMode: TrackerTaskMode = 'HAND';
  private maxHands: MaxHandsOption = 2;
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
    initialMaxHands: MaxHandsOption = 2,
  ): Promise<GpuInfo> {
    this.onResultCallback = onResult;
    this.onDelegateChangeCallback = onDelegateChange ?? null;
    this.engineMode = initialEngineMode;
    this.taskMode = initialTaskMode;
    this.maxHands = initialMaxHands;

    const canUseWorker =
      typeof Worker !== 'undefined' &&
      typeof OffscreenCanvas !== 'undefined' &&
      typeof createImageBitmap === 'function';

    if (canUseWorker) {
      try {
        const gpuInfo = await this.initWorker(initialEngineMode, initialTaskMode, initialMaxHands);
        this.gpuInfo = gpuInfo;
        return gpuInfo;
      } catch (err) {
        this.worker?.terminate();
        this.worker = null;
        if (initialEngineMode === 'CPU') {
          console.warn('Worker CPU tracker unavailable, using main thread:', err);
        }
      }
    }

    return this.initMainThreadFallback(initialEngineMode, initialTaskMode, initialMaxHands);
  }

  private async initMainThreadFallback(
    wantDelegate: AiEngineMode,
    task: TrackerTaskMode,
    numHands: MaxHandsOption,
  ): Promise<GpuInfo> {
    const tracker = await createMainThreadTracker(wantDelegate, task, numHands);
    if (this.disposed) {
      tracker.handLandmarker?.close();
      tracker.poseLandmarker?.close();
      return tracker.gpuInfo;
    }
    this.mainHandLandmarker?.close();
    this.mainPoseLandmarker?.close();
    this.mainHandLandmarker = tracker.handLandmarker;
    this.mainPoseLandmarker = tracker.poseLandmarker;
    if (!this.mainCanvas) {
      this.mainCanvas = document.createElement('canvas');
      this.mainCanvas.width = AI_W;
      this.mainCanvas.height = AI_H;
      this.mainCtx = this.mainCanvas.getContext('2d');
    }
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
    } else if (!this.disposed) {
      void this.initMainThreadFallback(mode, this.taskMode, this.maxHands);
    }
  }

  public setTaskMode(mode: TrackerTaskMode): void {
    if (this.taskMode === mode) return;
    this.taskMode = mode;
    if (this.worker && !this.disposed) {
      this.worker.postMessage({ type: 'SET_TASK_MODE', taskMode: mode });
    } else if (!this.disposed) {
      void this.initMainThreadFallback(this.engineMode, mode, this.maxHands);
    }
  }

  public setMaxHands(maxHands: MaxHandsOption): void {
    if (this.maxHands === maxHands) return;
    this.maxHands = maxHands;
    if (this.worker && !this.disposed) {
      this.worker.postMessage({ type: 'SET_MAX_HANDS', maxHands });
    } else if (!this.disposed && this.taskMode === 'HAND') {
      void this.initMainThreadFallback(this.engineMode, 'HAND', maxHands);
    }
  }

  public getTelemetry(): {
    isWorker: boolean;
    taskMode: TrackerTaskMode;
    maxHands: MaxHandsOption;
    delegate: 'GPU' | 'CPU';
    avgInferMs: number;
    aiFps: number;
  } {
    return {
      isWorker: this.worker !== null,
      taskMode: this.taskMode,
      maxHands: this.maxHands,
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
      this.avgInferMs = this.avgInferMs > 0 ? this.avgInferMs * 0.82 + inferMs * 0.18 : inferMs;
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
    initialMaxHands: MaxHandsOption,
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
        if (msg.type === 'GPU_UNAVAILABLE') {
          window.clearTimeout(timeoutId);
          reject(new Error(String(msg.message || 'Worker GPU delegate unavailable')));
          return;
        }
        if (msg.type === 'DELEGATE_CHANGED') {
          this.activeDelegate = (msg.delegate ?? 'GPU') as 'GPU' | 'CPU';
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
          const landmarks = (msg.landmarks ?? []) as Landmark3D[][];
          const poseLandmarks = (msg.poseLandmarks ?? null) as Landmark3D[] | null;

          this.activeDelegate = delegate;
          const aiFps = this.recordCompletion(now, inferMs);

          this.flushPendingToWorker();

          this.onResultCallback?.({
            taskMode: pktTask,
            landmarks,
            poseLandmarks,
            aiFps,
            inferMs,
            avgInferMs: Math.round(this.avgInferMs * 10) / 10,
            delegate,
          });
        }
      };

      worker.postMessage({
        type: 'INIT',
        engineMode: initialMode,
        taskMode: initialTask,
        maxHands: initialMaxHands,
      });
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

    const activeMainModel =
      this.taskMode === 'POSE' ? this.mainPoseLandmarker : this.mainHandLandmarker;
    if (activeMainModel && this.mainCanvas && this.mainCtx && !this.isProcessing) {
      this.isProcessing = true;
      window.setTimeout(() => {
        if (this.disposed || !this.mainCanvas || !this.mainCtx) {
          this.isProcessing = false;
          return;
        }
        try {
          this.mainCtx.drawImage(video, 0, 0, AI_W, AI_H);
          const doneNow = performance.now();
          const ts = Math.max(this.lastDetectTs + 1, Math.round(doneNow));
          this.lastDetectTs = ts;
          const t0 = performance.now();
          let landmarks: Landmark3D[][] = [];
          let poseLandmarks: Landmark3D[] | null = null;

          if (this.taskMode === 'POSE' && this.mainPoseLandmarker) {
            const res = this.mainPoseLandmarker.detectForVideo(this.mainCanvas, ts);
            poseLandmarks = (res.landmarks?.[0] ?? null) as Landmark3D[] | null;
          } else if (this.mainHandLandmarker) {
            const res = this.mainHandLandmarker.detectForVideo(this.mainCanvas, ts);
            landmarks = (res.landmarks ?? []) as Landmark3D[][];
          }

          const inferMs = performance.now() - t0;
          const aiFps = this.recordCompletion(doneNow, inferMs);

          this.onResultCallback?.({
            taskMode: this.taskMode,
            landmarks,
            poseLandmarks,
            aiFps,
            inferMs,
            avgInferMs: Math.round(this.avgInferMs * 10) / 10,
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
    if (this.mainHandLandmarker) {
      this.mainHandLandmarker.close();
      this.mainHandLandmarker = null;
    }
    if (this.mainPoseLandmarker) {
      this.mainPoseLandmarker.close();
      this.mainPoseLandmarker = null;
    }
  }
}
