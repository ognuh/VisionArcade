import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { GpuInfo, GpuTier, Landmark3D } from '../types/vision';

const MEDIAPIPE_WASM_ROOT =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';

const HAND_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const AI_W = 480;
const AI_H = 270;
const LUMA_SAMPLE_INTERVAL_MS = 500;

export type LightingStatus = 'NORMAL' | 'LOW_LIGHT_BOOST' | 'HIGHLIGHT_TAME';

interface InitRequest {
  type: 'INIT';
}

interface DetectRequest {
  type: 'DETECT';
  bitmap: ImageBitmap;
  timestampMs: number;
}

interface CloseRequest {
  type: 'CLOSE';
}

type WorkerRequest = InitRequest | DetectRequest | CloseRequest;

let landmarker: HandLandmarker | null = null;
let prepCanvas: OffscreenCanvas | null = null;
let prepCtx: OffscreenCanvasRenderingContext2D | null = null;
let lastLumaCheckMs = 0;
let lightingMode: LightingStatus = 'NORMAL';
let emaLumaValue = 128;
let lumaInitialized = false;

// Hysteresis thresholds to eliminate rapid toggling in borderline lighting
const LOW_LIGHT_ON_LUMA = 75;
const LOW_LIGHT_OFF_LUMA = 100;
const HIGHLIGHT_ON_LUMA = 188;
const HIGHLIGHT_OFF_LUMA = 165;
const LUMA_EMA_ALPHA = 0.25;

// Pre-computed 256-entry LUTs for Zero-GC real-time low-light gamma lift & backlight highlight taming
const lowLightLut = new Uint8ClampedArray(256);
const highlightLut = new Uint8ClampedArray(256);

for (let i = 0; i < 256; i += 1) {
  const n = i / 255;
  // Low-light: Gamma Lift (gamma ~0.55) + Contrast 1.35
  const gammaLifted = Math.pow(n, 0.55);
  const lowC = (gammaLifted - 0.5) * 1.35 + 0.52;
  lowLightLut[i] = Math.max(0, Math.min(255, Math.round(lowC * 255)));

  // Backlight / Overexposed: Lower highlights + boost midtone separation
  const tamed = Math.pow(n, 1.32);
  const hiC = (tamed - 0.5) * 1.18 + 0.48;
  highlightLut[i] = Math.max(0, Math.min(255, Math.round(hiC * 255)));
}

function classifyGpuTier(renderer: string): GpuTier {
  const lower = renderer.toLowerCase();
  if (
    lower.includes('swiftshader') ||
    lower.includes('llvmpipe') ||
    lower.includes('microsoft basic render')
  ) {
    return 'SOFTWARE';
  }
  if (
    lower.includes('nvidia') ||
    lower.includes('geforce') ||
    lower.includes('rtx') ||
    lower.includes('gtx') ||
    lower.includes('radeon rx')
  ) {
    return 'DISCRETE';
  }
  if (lower.includes('intel') || lower.includes('uhd') || lower.includes('iris') || lower.includes('apple m')) {
    return 'INTEGRATED';
  }
  return 'UNKNOWN';
}

function detectWorkerGpu(): { renderer: string; tier: GpuTier } {
  try {
    const c = new OffscreenCanvas(16, 16);
    const gl =
      c.getContext('webgl2', { powerPreference: 'high-performance' }) ||
      c.getContext('webgl', { powerPreference: 'high-performance' });
    if (!gl) return { renderer: 'Worker CPU', tier: 'SOFTWARE' };
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const raw = ext
      ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) ?? '')
      : String(gl.getParameter(gl.RENDERER) ?? '');
    const cleaned = raw
      .replace(/^ANGLE\s*\([^,]+,\s*/i, '')
      .replace(/\s*\(0x[0-9a-f]+\).*$/i, '')
      .replace(/\s*Direct3D11.*$/i, '')
      .replace(/\)\s*$/, '')
      .trim();
    return { renderer: cleaned || raw || 'WebGL GPU', tier: classifyGpuTier(raw) };
  } catch {
    return { renderer: 'Worker CPU', tier: 'SOFTWARE' };
  }
}

/** Adaptive Auto-Exposure & Contrast Boost on 480x270 OffscreenCanvas strictly for AI input. */
function preprocessForAi(bitmap: ImageBitmap, nowMs: number): OffscreenCanvas {
  if (!prepCanvas || !prepCtx) {
    prepCanvas = new OffscreenCanvas(AI_W, AI_H);
    prepCtx = prepCanvas.getContext('2d', { willReadFrequently: true });
  }
  const ctx = prepCtx!;
  ctx.drawImage(bitmap, 0, 0, AI_W, AI_H);

  const shouldMeasure = nowMs - lastLumaCheckMs >= LUMA_SAMPLE_INTERVAL_MS;
  const needsLut = lightingMode !== 'NORMAL';

  if (shouldMeasure || needsLut) {
    const img = ctx.getImageData(0, 0, AI_W, AI_H);
    const data = img.data;

    if (shouldMeasure) {
      lastLumaCheckMs = nowMs;
      let sum = 0;
      let count = 0;
      // Fast stride-8 pixel luminance scan (< 0.1ms)
      for (let i = 0; i < data.length; i += 32) {
        sum += data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
        count += 1;
      }
      const frameLuma = count > 0 ? sum / count : 128;
      if (!lumaInitialized) {
        emaLumaValue = frameLuma;
        lumaInitialized = true;
      } else {
        emaLumaValue = emaLumaValue * (1 - LUMA_EMA_ALPHA) + frameLuma * LUMA_EMA_ALPHA;
      }

      // Schmitt-trigger hysteresis: ON < 75, OFF > 100 (prevents flickering)
      if (lightingMode === 'LOW_LIGHT_BOOST') {
        if (emaLumaValue > LOW_LIGHT_OFF_LUMA) {
          lightingMode = emaLumaValue > HIGHLIGHT_ON_LUMA ? 'HIGHLIGHT_TAME' : 'NORMAL';
        }
      } else if (lightingMode === 'HIGHLIGHT_TAME') {
        if (emaLumaValue < HIGHLIGHT_OFF_LUMA) {
          lightingMode = emaLumaValue < LOW_LIGHT_ON_LUMA ? 'LOW_LIGHT_BOOST' : 'NORMAL';
        }
      } else {
        if (emaLumaValue < LOW_LIGHT_ON_LUMA) {
          lightingMode = 'LOW_LIGHT_BOOST';
        } else if (emaLumaValue > HIGHLIGHT_ON_LUMA) {
          lightingMode = 'HIGHLIGHT_TAME';
        }
      }
    }

    if (lightingMode !== 'NORMAL') {
      const lut = lightingMode === 'LOW_LIGHT_BOOST' ? lowLightLut : highlightLut;
      for (let i = 0; i < data.length; i += 4) {
        data[i] = lut[data[i]];
        data[i + 1] = lut[data[i + 1]];
        data[i + 2] = lut[data[i + 2]];
      }
      ctx.putImageData(img, 0, 0);
    }
  }

  return prepCanvas;
}

async function initTracker(): Promise<void> {
  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
    const gpu = detectWorkerGpu();

    try {
      const gpuCanvas = new OffscreenCanvas(AI_W, AI_H);
      landmarker = await HandLandmarker.createFromOptions(vision, {
        canvas: gpuCanvas as unknown as HTMLCanvasElement,
        baseOptions: {
          modelAssetPath: HAND_LANDMARKER_MODEL_URL,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence: 0.35,
        minTrackingConfidence: 0.4,
      });

      const gpuInfo: GpuInfo = {
        delegate: 'GPU',
        renderer: `${gpu.renderer} (Worker)`,
        tier: gpu.tier,
      };
      self.postMessage({ type: 'READY', gpuInfo });
      return;
    } catch {
      // Fallback to CPU WASM inside worker (still completely off main thread!)
      landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_LANDMARKER_MODEL_URL,
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence: 0.35,
        minTrackingConfidence: 0.4,
      });

      const gpuInfo: GpuInfo = {
        delegate: 'CPU',
        renderer: 'Worker WASM',
        tier: 'SOFTWARE',
      };
      self.postMessage({ type: 'READY', gpuInfo });
    }
  } catch (err) {
    self.postMessage({
      type: 'ERROR',
      message: err instanceof Error ? err.message : 'Worker HandLandmarker init failed',
    });
  }
}

let lastDetectTs = 0;

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (msg.type === 'INIT') {
    void initTracker();
    return;
  }

  if (msg.type === 'CLOSE') {
    landmarker?.close();
    landmarker = null;
    return;
  }

  if (msg.type === 'DETECT') {
    const bitmap = msg.bitmap;
    try {
      if (!landmarker) {
        bitmap.close();
        self.postMessage({
          type: 'RESULT',
          landmarks: [] as Landmark3D[][],
          lighting: lightingMode,
          avgLuma: Math.round(emaLumaValue),
          timestampMs: msg.timestampMs,
        });
        return;
      }

      const processed = preprocessForAi(bitmap, msg.timestampMs);
      bitmap.close();

      const ts = Math.max(lastDetectTs + 1, Math.round(msg.timestampMs));
      lastDetectTs = ts;

      const result = landmarker.detectForVideo(
        processed as unknown as HTMLCanvasElement,
        ts,
      );

      self.postMessage({
        type: 'RESULT',
        landmarks: (result.landmarks ?? []) as Landmark3D[][],
        lighting: lightingMode,
        avgLuma: Math.round(emaLumaValue),
        timestampMs: msg.timestampMs,
      });
    } catch {
      try {
        bitmap.close();
      } catch {
        // already closed
      }
      self.postMessage({
        type: 'RESULT',
        landmarks: [] as Landmark3D[][],
        lighting: lightingMode,
        avgLuma: Math.round(emaLumaValue),
        timestampMs: msg.timestampMs,
      });
    }
  }
};
