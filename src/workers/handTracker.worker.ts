import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import type { AiEngineMode, GpuInfo, GpuTier, Landmark3D, TrackerTaskMode } from '../types/vision';

const ALLOWED_WASM_LOADERS = new Set([
  '/mediapipe/vision_wasm_module_internal.js',
  '/mediapipe/vision_wasm_internal.js',
  '/mediapipe/vision_wasm_nosimd_internal.js',
]);

// Strictly allowlisted same-origin native ES import() hook for MediaPipe's official ES Module WASM loader.
// Zero eval() / new Function() -> compatible with strict CSP (script-src 'self' 'wasm-unsafe-eval').
(self as unknown as { import?: (url: string) => Promise<void> }).import = async (url: string) => {
  const parsed = new URL(url, self.location.origin);
  if (parsed.origin !== self.location.origin || !ALLOWED_WASM_LOADERS.has(parsed.pathname)) {
    throw new Error(`Blocked unauthorized MediaPipe loader URL: ${parsed.pathname}`);
  }
  const mod = (await import(/* @vite-ignore */ parsed.href)) as { default?: unknown };
  const g = self as unknown as { ModuleFactory?: unknown };
  if (!g.ModuleFactory && mod?.default) {
    g.ModuleFactory = mod.default;
  }
};

const MEDIAPIPE_WASM_ROOT = '/mediapipe';

const HAND_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const POSE_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const AI_W = 320;
const AI_H = 180;
const WARMUP_FRAMES = 3;
const BENCHMARK_FRAMES = 25;
const GPU_MAX_AVG_INFER_MS = 24;
const GPU_MIN_AI_FPS = 24;

interface InitRequest {
  type: 'INIT';
  engineMode?: AiEngineMode;
  taskMode?: TrackerTaskMode;
}

interface DetectRequest {
  type: 'DETECT';
  bitmap: ImageBitmap;
  timestampMs: number;
}

interface SetEngineRequest {
  type: 'SET_ENGINE';
  engineMode: AiEngineMode;
}

interface SetTaskModeRequest {
  type: 'SET_TASK_MODE';
  taskMode: TrackerTaskMode;
}

interface CloseRequest {
  type: 'CLOSE';
}

type WorkerRequest =
  | InitRequest
  | DetectRequest
  | SetEngineRequest
  | SetTaskModeRequest
  | CloseRequest;

type VisionFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

let visionFileset: VisionFileset | null = null;
let handLandmarker: HandLandmarker | null = null;
let handDelegate: 'GPU' | 'CPU' | null = null;
let poseLandmarker: PoseLandmarker | null = null;
let poseDelegate: 'GPU' | 'CPU' | null = null;

let engineMode: AiEngineMode = 'GPU';
let taskMode: TrackerTaskMode = 'HAND';
let activeDelegate: 'GPU' | 'CPU' = 'GPU';
let isSwitching = false;
let lastDetectTs = 0;

// Auto-Benchmark state (first 25 steady-state frames after 3 cold warmup frames on GPU)
let benchDone = false;
let warmupCount = 0;
let benchCount = 0;
let benchInferSumMs = 0;
let benchStartMs = 0;
let benchInputSpanSumMs = 0;
let benchPrevInputTs = 0;

function resetBenchmark(): void {
  benchDone = false;
  warmupCount = 0;
  benchCount = 0;
  benchInferSumMs = 0;
  benchStartMs = 0;
  benchInputSpanSumMs = 0;
  benchPrevInputTs = 0;
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
  if (
    lower.includes('intel') ||
    lower.includes('uhd') ||
    lower.includes('iris') ||
    lower.includes('apple m')
  ) {
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

async function createHandLandmarker(
  vision: VisionFileset,
  delegate: 'GPU' | 'CPU',
): Promise<HandLandmarker> {
  if (delegate === 'GPU') {
    const glHost = new OffscreenCanvas(AI_W, AI_H);
    return HandLandmarker.createFromOptions(vision, {
      canvas: glHost as unknown as HTMLCanvasElement,
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
  }

  return HandLandmarker.createFromOptions(vision, {
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
}

async function createPoseLandmarker(
  vision: VisionFileset,
  delegate: 'GPU' | 'CPU',
): Promise<PoseLandmarker> {
  if (delegate === 'GPU') {
    const glHost = new OffscreenCanvas(AI_W, AI_H);
    return PoseLandmarker.createFromOptions(vision, {
      canvas: glHost as unknown as HTMLCanvasElement,
      baseOptions: {
        modelAssetPath: POSE_LANDMARKER_MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.45,
      minPosePresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
      outputSegmentationMasks: false,
    });
  }

  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: POSE_LANDMARKER_MODEL_URL,
      delegate: 'CPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.45,
    minPosePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
    outputSegmentationMasks: false,
  });
}

/** Ensures ONLY the active landmarker (HAND or POSE) is ready with the requested delegate. */
async function ensureActiveModel(
  desiredTask: TrackerTaskMode,
  desiredDelegate: 'GPU' | 'CPU',
): Promise<'GPU' | 'CPU'> {
  if (!visionFileset) return desiredDelegate;

  if (desiredTask === 'HAND') {
    if (handLandmarker && handDelegate === desiredDelegate) return desiredDelegate;
    const old = handLandmarker;
    handLandmarker = null;
    handDelegate = null;
    old?.close();

    try {
      handLandmarker = await createHandLandmarker(visionFileset, desiredDelegate);
      handDelegate = desiredDelegate;
      return desiredDelegate;
    } catch {
      handLandmarker = await createHandLandmarker(visionFileset, 'CPU');
      handDelegate = 'CPU';
      return 'CPU';
    }
  } else {
    if (poseLandmarker && poseDelegate === desiredDelegate) return desiredDelegate;
    const old = poseLandmarker;
    poseLandmarker = null;
    poseDelegate = null;
    old?.close();

    try {
      poseLandmarker = await createPoseLandmarker(visionFileset, desiredDelegate);
      poseDelegate = desiredDelegate;
      return desiredDelegate;
    } catch {
      poseLandmarker = await createPoseLandmarker(visionFileset, 'CPU');
      poseDelegate = 'CPU';
      return 'CPU';
    }
  }
}

async function hotSwitch(
  targetTask: TrackerTaskMode,
  targetDelegate: 'GPU' | 'CPU',
  reason: string,
  avgInferMs = 0,
): Promise<void> {
  if (isSwitching || !visionFileset) return;
  isSwitching = true;
  try {
    taskMode = targetTask;
    const actual = await ensureActiveModel(targetTask, targetDelegate);
    const delegateChanged = actual !== activeDelegate;
    activeDelegate = actual;

    if (delegateChanged || reason === 'manual' || reason === 'auto-benchmark') {
      self.postMessage({
        type: 'DELEGATE_CHANGED',
        delegate: activeDelegate,
        taskMode,
        reason,
        avgInferMs: Math.round(avgInferMs * 10) / 10,
      });
    }
  } finally {
    isSwitching = false;
  }
}

async function initTracker(initialMode: AiEngineMode, initialTask: TrackerTaskMode): Promise<void> {
  engineMode = initialMode;
  taskMode = initialTask;
  resetBenchmark();

  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
    // Use MediaPipe's official ES Module WASM loader in ES Module Workers (sets globalThis.ModuleFactory natively, zero eval)
    if (vision.wasmLoaderPath.endsWith('/vision_wasm_internal.js')) {
      vision.wasmLoaderPath = `${MEDIAPIPE_WASM_ROOT}/vision_wasm_module_internal.js`;
      vision.wasmBinaryPath = `${MEDIAPIPE_WASM_ROOT}/vision_wasm_module_internal.wasm`;
    }
    visionFileset = vision;
    const gpu = detectWorkerGpu();

    const wantDelegate: 'GPU' | 'CPU' = initialMode === 'CPU' ? 'CPU' : 'GPU';
    const actual = await ensureActiveModel(initialTask, wantDelegate);
    activeDelegate = actual;
    if (actual === 'CPU') benchDone = true;

    const gpuInfo: GpuInfo =
      actual === 'GPU'
        ? { delegate: 'GPU', renderer: `${gpu.renderer} (Worker)`, tier: gpu.tier }
        : { delegate: 'CPU', renderer: 'Worker WASM SIMD', tier: 'SOFTWARE' };

    self.postMessage({ type: 'READY', gpuInfo, delegate: activeDelegate, taskMode });
  } catch (err) {
    self.postMessage({
      type: 'ERROR',
      message: err instanceof Error ? err.message : 'Worker MediaPipe init failed',
    });
  }
}

function sanitizeEngineMode(mode: unknown): AiEngineMode {
  return mode === 'GPU' || mode === 'CPU' || mode === 'AUTO' ? mode : 'GPU';
}

function sanitizeTaskMode(mode: unknown): TrackerTaskMode {
  return mode === 'POSE' ? 'POSE' : 'HAND';
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (!msg || typeof msg !== 'object') return;

  if (msg.type === 'INIT') {
    void initTracker(sanitizeEngineMode(msg.engineMode), sanitizeTaskMode(msg.taskMode));
    return;
  }

  if (msg.type === 'SET_ENGINE') {
    engineMode = sanitizeEngineMode(msg.engineMode);
    resetBenchmark();
    if (engineMode === 'CPU') {
      benchDone = true;
      void hotSwitch(taskMode, 'CPU', 'manual');
    } else if (engineMode === 'GPU') {
      benchDone = true;
      void hotSwitch(taskMode, 'GPU', 'manual');
    } else {
      void hotSwitch(taskMode, 'GPU', 'auto-rebenchmark');
    }
    return;
  }

  if (msg.type === 'SET_TASK_MODE') {
    const nextTask = sanitizeTaskMode(msg.taskMode);
    if (nextTask === taskMode) return;
    taskMode = nextTask;
    // Mutually exclusive: only run & ensure the requested model (HAND or POSE)
    void hotSwitch(nextTask, activeDelegate, 'task-switch');
    return;
  }

  if (msg.type === 'CLOSE') {
    handLandmarker?.close();
    handLandmarker = null;
    handDelegate = null;
    poseLandmarker?.close();
    poseLandmarker = null;
    poseDelegate = null;
    visionFileset = null;
    return;
  }

  if (msg.type === 'DETECT') {
    const bitmap = msg.bitmap;
    if (!(bitmap instanceof ImageBitmap)) return;

    try {
      const currentTask = taskMode;
      const activeModel = currentTask === 'POSE' ? poseLandmarker : handLandmarker;

      if (!activeModel || isSwitching) {
        self.postMessage({
          type: 'RESULT',
          taskMode: currentTask,
          landmarks: [] as Landmark3D[][],
          poseLandmarks: null as Landmark3D[] | null,
          inferMs: 0,
          delegate: activeDelegate,
          timestampMs: msg.timestampMs,
        });
        return;
      }

      const ts = Math.max(lastDetectTs + 1, Math.round(msg.timestampMs));
      lastDetectTs = ts;

      const t0 = performance.now();
      let handOut: Landmark3D[][] = [];
      let poseOut: Landmark3D[] | null = null;

      if (currentTask === 'POSE' && poseLandmarker) {
        const res = poseLandmarker.detectForVideo(bitmap as unknown as HTMLCanvasElement, ts);
        poseOut = (res.landmarks?.[0] ?? null) as Landmark3D[] | null;
      } else if (handLandmarker) {
        const res = handLandmarker.detectForVideo(bitmap as unknown as HTMLCanvasElement, ts);
        handOut = (res.landmarks ?? []) as Landmark3D[][];
      }

      const inferMs = performance.now() - t0;

      // Auto-Benchmark over 25 steady-state GPU frames (skipping first 3 cold warmup frames)
      if (engineMode === 'AUTO' && !benchDone && activeDelegate === 'GPU') {
        if (warmupCount < WARMUP_FRAMES) {
          warmupCount += 1;
        } else {
          const now = performance.now();
          if (benchCount === 0) benchStartMs = now;
          if (benchPrevInputTs > 0) {
            benchInputSpanSumMs += Math.max(0, msg.timestampMs - benchPrevInputTs);
          }
          benchPrevInputTs = msg.timestampMs;
          benchCount += 1;
          benchInferSumMs += inferMs;

          if (benchCount >= BENCHMARK_FRAMES) {
            benchDone = true;
            const avgInferMs = benchInferSumMs / benchCount;
            const elapsedSec = Math.max(0.05, (now - benchStartMs) / 1000);
            const actualAiFps = benchCount / elapsedSec;
            const avgInputCadenceMs =
              benchCount > 1 ? benchInputSpanSumMs / (benchCount - 1) : 33.3;
            const webcamFeeding30 = avgInputCadenceMs <= 45;

            if (avgInferMs > GPU_MAX_AVG_INFER_MS || (webcamFeeding30 && actualAiFps < GPU_MIN_AI_FPS)) {
              void hotSwitch(taskMode, 'CPU', 'auto-benchmark', avgInferMs);
            }
          }
        }
      }

      self.postMessage({
        type: 'RESULT',
        taskMode: currentTask,
        landmarks: handOut,
        poseLandmarks: poseOut,
        inferMs,
        delegate: activeDelegate,
        timestampMs: msg.timestampMs,
      });
    } catch {
      self.postMessage({
        type: 'RESULT',
        taskMode,
        landmarks: [] as Landmark3D[][],
        poseLandmarks: null as Landmark3D[] | null,
        inferMs: 0,
        delegate: activeDelegate,
        timestampMs: msg.timestampMs,
      });
    } finally {
      try {
        bitmap.close();
      } catch {
        // already closed
      }
    }
  }
};
