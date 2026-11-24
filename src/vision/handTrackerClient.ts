import type { HandLandmarker } from '@mediapipe/tasks-vision';
import type { GpuInfo, Landmark3D } from '../types/vision';
import type { LightingStatus } from '../workers/handTracker.worker';
import { createHandTracker } from './handTracker';

export type { LightingStatus };

export interface TrackingResultPacket {
  landmarks: Landmark3D[][];
  lighting: LightingStatus;
  avgLuma: number;
  aiFps: number;
}

const AI_W = 480;
const AI_H = 270;
const LUMA_CHECK_MS = 500;

/**
 * Multi-Threaded HandTracker Client:
 * - Primary path: Runs MediaPipe HandLandmarker + Auto-Exposure preprocessor inside a dedicated
 *   Web Worker using transferable ImageBitmap (zero-copy, zero main-thread blocking, isProcessing backpressure).
 * - Automatic fallback: If Worker/OffscreenCanvas fails, seamlessly falls back to Main Thread
 *   with time-sliced 320x180 adaptive exposure preprocessing.
 */
export class HandTrackerClient {
  private worker: Worker | null = null;
  private fallbackLandmarker: HandLandmarker | null = null;
  private fallbackCanvas: HTMLCanvasElement | null = null;
  private fallbackCtx: CanvasRenderingContext2D | null = null;
  private lowLightLut = new Uint8ClampedArray(256);
  private highlightLut = new Uint8ClampedArray(256);

  private isProcessing = false;
  private disposed = false;
  private lastResultMs = 0;
  private emaAiDt = 1 / 30;
  private lastDetectTs = 0;
  private lastLumaMs = 0;
  private lighting: LightingStatus = 'NORMAL';
  private emaLuma = 128;
  private lumaInit = false;

  public gpuInfo: GpuInfo = {
    delegate: 'GPU',
    renderer: 'Initializing...',
    tier: 'UNKNOWN',
  };

  private onResultCallback: ((packet: TrackingResultPacket) => void) | null = null;

  constructor() {
    for (let i = 0; i < 256; i += 1) {
      const n = i / 255;
      const lowC = (Math.pow(n, 0.55) - 0.5) * 1.35 + 0.52;
      this.lowLightLut[i] = Math.max(0, Math.min(255, Math.round(lowC * 255)));
      const hiC = (Math.pow(n, 1.32) - 0.5) * 1.18 + 0.48;
      this.highlightLut[i] = Math.max(0, Math.min(255, Math.round(hiC * 255)));
    }
  }

  async init(onResult: (packet: TrackingResultPacket) => void): Promise<GpuInfo> {
    this.onResultCallback = onResult;

    const canUseWorker =
      typeof Worker !== 'undefined' &&
      typeof OffscreenCanvas !== 'undefined' &&
      typeof createImageBitmap === 'function';

    if (canUseWorker) {
      try {
        const gpuInfo = await this.initWorker();
        this.gpuInfo = gpuInfo;
        return gpuInfo;
      } catch (err) {
        console.warn('Web Worker HandLandmarker unavailable, falling back to main thread:', err);
        this.worker?.terminate();
        this.worker = null;
      }
    }

    // Automatic Main-Thread fallback with time-slicing & adaptive exposure preprocessing
    const tracker = await createHandTracker(false);
    if (this.disposed) {
      tracker.landmarker.close();
      return tracker.gpuInfo;
    }
    this.fallbackLandmarker = tracker.landmarker;
    this.fallbackCanvas = document.createElement('canvas');
    this.fallbackCanvas.width = AI_W;
    this.fallbackCanvas.height = AI_H;
    this.fallbackCtx = this.fallbackCanvas.getContext('2d', { willReadFrequently: true });
    this.gpuInfo = tracker.gpuInfo;
    return tracker.gpuInfo;
  }

  private initWorker(): Promise<GpuInfo> {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL('../workers/handTracker.worker.ts', import.meta.url), {
        type: 'module',
      });
      this.worker = worker;

      const timeoutId = window.setTimeout(() => {
        reject(new Error('Worker MediaPipe init timed out'));
      }, 12000);

      worker.onerror = (ev) => {
        window.clearTimeout(timeoutId);
        reject(new Error(ev.message || 'Worker runtime error'));
      };

      worker.onmessage = (e: MessageEvent) => {
        const msg = e.data;
        if (msg.type === 'READY') {
          window.clearTimeout(timeoutId);
          resolve(msg.gpuInfo as GpuInfo);
          return;
        }
        if (msg.type === 'ERROR') {
          window.clearTimeout(timeoutId);
          reject(new Error(String(msg.message || 'Worker init error')));
          return;
        }
        if (msg.type === 'RESULT') {
          this.isProcessing = false;
          if (this.disposed) return;

          const now = performance.now();
          if (this.lastResultMs > 0) {
            const dtSec = Math.max(0.005, Math.min(0.25, (now - this.lastResultMs) / 1000));
            this.emaAiDt = this.emaAiDt * 0.85 + dtSec * 0.15;
          }
          this.lastResultMs = now;

          this.lighting = msg.lighting as LightingStatus;
          this.emaLuma = Number(msg.avgLuma ?? 128);

          this.onResultCallback?.({
            landmarks: (msg.landmarks ?? []) as Landmark3D[][],
            lighting: this.lighting,
            avgLuma: Math.round(this.emaLuma),
            aiFps: Math.round(1 / Math.max(this.emaAiDt, 0.01)),
          });
        }
      };

      worker.postMessage({ type: 'INIT' });
    });
  }

  /** Submits a new webcam frame only if the tracker is idle (isProcessing === false). */
  submitFrame(video: HTMLVideoElement, timestampMs: number): void {
    if (this.disposed || this.isProcessing || video.readyState < 2) return;

    if (this.worker) {
      this.isProcessing = true;
      createImageBitmap(video)
        .then((bitmap) => {
          if (this.disposed || !this.worker) {
            bitmap.close();
            this.isProcessing = false;
            return;
          }
          this.worker.postMessage({ type: 'DETECT', bitmap, timestampMs }, [bitmap]);
        })
        .catch(() => {
          this.isProcessing = false;
        });
      return;
    }

    // Time-sliced main-thread fallback
    if (this.fallbackLandmarker && this.fallbackCanvas && this.fallbackCtx) {
      this.isProcessing = true;
      window.setTimeout(() => {
        if (this.disposed || !this.fallbackLandmarker || !this.fallbackCanvas || !this.fallbackCtx) {
          this.isProcessing = false;
          return;
        }
        try {
          this.fallbackCtx.drawImage(video, 0, 0, AI_W, AI_H);
          const now = performance.now();
          const shouldMeasure = now - this.lastLumaMs >= LUMA_CHECK_MS;
          if (shouldMeasure || this.lighting !== 'NORMAL') {
            const img = this.fallbackCtx.getImageData(0, 0, AI_W, AI_H);
            const data = img.data;
            if (shouldMeasure) {
              this.lastLumaMs = now;
              let sum = 0;
              let count = 0;
              for (let i = 0; i < data.length; i += 32) {
                sum += data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
                count += 1;
              }
              const frameLuma = count > 0 ? sum / count : 128;
              if (!this.lumaInit) {
                this.emaLuma = frameLuma;
                this.lumaInit = true;
              } else {
                this.emaLuma = this.emaLuma * 0.75 + frameLuma * 0.25;
              }

              // Schmitt-trigger hysteresis: ON < 75, OFF > 100
              if (this.lighting === 'LOW_LIGHT_BOOST') {
                if (this.emaLuma > 100) {
                  this.lighting = this.emaLuma > 188 ? 'HIGHLIGHT_TAME' : 'NORMAL';
                }
              } else if (this.lighting === 'HIGHLIGHT_TAME') {
                if (this.emaLuma < 165) {
                  this.lighting = this.emaLuma < 75 ? 'LOW_LIGHT_BOOST' : 'NORMAL';
                }
              } else if (this.emaLuma < 75) {
                this.lighting = 'LOW_LIGHT_BOOST';
              } else if (this.emaLuma > 188) {
                this.lighting = 'HIGHLIGHT_TAME';
              }
            }
            if (this.lighting !== 'NORMAL') {
              const lut = this.lighting === 'LOW_LIGHT_BOOST' ? this.lowLightLut : this.highlightLut;
              for (let i = 0; i < data.length; i += 4) {
                data[i] = lut[data[i]];
                data[i + 1] = lut[data[i + 1]];
                data[i + 2] = lut[data[i + 2]];
              }
              this.fallbackCtx.putImageData(img, 0, 0);
            }
          }

          const ts = Math.max(this.lastDetectTs + 1, Math.round(now));
          this.lastDetectTs = ts;
          const result = this.fallbackLandmarker.detectForVideo(this.fallbackCanvas, ts);

          if (this.lastResultMs > 0) {
            const dtSec = Math.max(0.005, Math.min(0.25, (now - this.lastResultMs) / 1000));
            this.emaAiDt = this.emaAiDt * 0.85 + dtSec * 0.15;
          }
          this.lastResultMs = now;

          this.onResultCallback?.({
            landmarks: (result.landmarks ?? []) as Landmark3D[][],
            lighting: this.lighting,
            avgLuma: Math.round(this.emaLuma),
            aiFps: Math.round(1 / Math.max(this.emaAiDt, 0.01)),
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
