import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import type {
  AiEngineMode,
  GpuInfo,
  GpuTier,
  Landmark3D,
  MaxHandsOption,
  TrackerTaskMode,
} from '../types/vision';

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

// 256x144: 36% fewer pixels than 320x180 -> lower GPU memory pressure & zero readPixels stalls
const AI_W = 256;
const AI_H = 144;

interface InitRequest {
  type: 'INIT';
  engineMode?: AiEngineMode;
  taskMode?: TrackerTaskMode;
  maxHands?: MaxHandsOption;
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

interface SetMaxHandsRequest {
  type: 'SET_MAX_HANDS';
  maxHands: MaxHandsOption;
}

interface CloseRequest {
  type: 'CLOSE';
}

type WorkerRequest =
  | InitRequest
  | DetectRequest
  | SetEngineRequest
  | SetTaskModeRequest
  | SetMaxHandsRequest
  | CloseRequest;

type VisionFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

let visionFileset: VisionFileset | null = null;
let handLandmarker: HandLandmarker | null = null;
let handDelegate: 'GPU' | 'CPU' | null = null;
let handNumConfigured: MaxHandsOption = 2;
let poseLandmarker: PoseLandmarker | null = null;
let poseDelegate: 'GPU' | 'CPU' | null = null;

let engineMode: AiEngineMode = 'GPU';
let taskMode: TrackerTaskMode = 'HAND';
let maxHands: MaxHandsOption = 2;
let activeDelegate: 'GPU' | 'CPU' = 'GPU';
let isSwitching = false;
let lastDetectTs = 0;

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

function detectWorkerGpu(): { renderer: string; tier: GpuTier; hasWebGl2: boolean } {
  try {
    const c = new OffscreenCanvas(16, 16);
    const gl2 = c.getContext('webgl2', { powerPreference: 'high-performance' });
    const gl = gl2 || c.getContext('webgl', { powerPreference: 'high-performance' });
    if (!gl) return { renderer: 'No Worker WebGL', tier: 'SOFTWARE', hasWebGl2: false };
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
    return { renderer: cleaned || raw || 'WebGL GPU', tier: classifyGpuTier(raw), hasWebGl2: Boolean(gl2) };
  } catch {
    return { renderer: 'No Worker WebGL', tier: 'SOFTWARE', hasWebGl2: false };
  }
}

async function createHandLandmarker(
  vision: VisionFileset,
  delegate: 'GPU' | 'CPU',
  numHands: MaxHandsOption,
): Promise<HandLandmarker> {
  const opts = {
    baseOptions: {
      modelAssetPath: HAND_LANDMARKER_MODEL_URL,
      delegate,
    },
    runningMode: 'VIDEO' as const,
    numHands,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.50,
    minTrackingConfidence: 0.45,
  };

  if (delegate === 'GPU') {
    const glHost = new OffscreenCanvas(AI_W, AI_H);
    return HandLandmarker.createFromOptions(vision, {
      canvas: glHost as unknown as HTMLCanvasElement,
      ...opts,
    });
  }
  return HandLandmarker.createFromOptions(vision, opts);
}

async function createPoseLandmarker(
  vision: VisionFileset,
  delegate: 'GPU' | 'CPU',
): Promise<PoseLandmarker> {
  const opts = {
    baseOptions: {
      modelAssetPath: POSE_LANDMARKER_MODEL_URL,
      delegate,
    },
    runningMode: 'VIDEO' as const,
    numPoses: 1,
    minPoseDetectionConfidence: 0.50,
    minPosePresenceConfidence: 0.50,
    minTrackingConfidence: 0.45,
  };

  if (delegate === 'GPU') {
    const glHost = new OffscreenCanvas(AI_W, AI_H);
    return PoseLandmarker.createFromOptions(vision, {
      canvas: glHost as unknown as HTMLCanvasElement,
      ...opts,
    });
  }
  return PoseLandmarker.createFromOptions(vision, opts);
}

function closeHandModel(): void {
  if (handLandmarker) {
    try {
      handLandmarker.close();
    } catch {
      // ignore
    }
    handLandmarker = null;
    handDelegate = null;
  }
}

function closePoseModel(): void {
  if (poseLandmarker) {
    try {
      poseLandmarker.close();
    } catch {
      // ignore
    }
    poseLandmarker = null;
    poseDelegate = null;
  }
}

/**
 * Ensures ONLY the active model (HAND or POSE) is loaded with the exact requested delegate.
 * Strictly NEVER downgrades GPU -> CPU silently; throws if GPU fails so client can try Main-Thread GPU.
 */
async function ensureActiveModel(
  targetTask: TrackerTaskMode,
  targetDelegate: 'GPU' | 'CPU',
  targetMaxHands: MaxHandsOption,
): Promise<void> {
  if (!visionFileset) throw new Error('MediaPipe FilesetResolver not initialized');

  if (targetTask === 'POSE') {
    if (poseLandmarker && poseDelegate === targetDelegate) return;
    closePoseModel();
    poseLandmarker = await createPoseLandmarker(visionFileset, targetDelegate);
    poseDelegate = targetDelegate;
    return;
  }

  if (handLandmarker && handDelegate === targetDelegate && handNumConfigured === targetMaxHands) {
    return;
  }
  closeHandModel();
  handLandmarker = await createHandLandmarker(visionFileset, targetDelegate, targetMaxHands);
  handDelegate = targetDelegate;
  handNumConfigured = targetMaxHands;
}

async function reconfigure(
  nextTask: TrackerTaskMode,
  nextDelegate: 'GPU' | 'CPU',
  nextMaxHands: MaxHandsOption,
): Promise<void> {
  if (!visionFileset || isSwitching) return;
  isSwitching = true;
  try {
    await ensureActiveModel(nextTask, nextDelegate, nextMaxHands);
    taskMode = nextTask;
    activeDelegate = nextDelegate;
    maxHands = nextMaxHands;
    lastDetectTs = 0;
    self.postMessage({
      type: 'DELEGATE_CHANGED',
      delegate: activeDelegate,
      taskMode,
      maxHands,
    });
  } catch (err) {
    self.postMessage({
      type: 'GPU_UNAVAILABLE',
      message: err instanceof Error ? err.message : 'Worker GPU delegate failed',
    });
  } finally {
    isSwitching = false;
  }
}

async function initTracker(
  initialMode: AiEngineMode,
  initialTask: TrackerTaskMode,
  initialMaxHands: MaxHandsOption,
): Promise<void> {
  engineMode = initialMode;
  taskMode = initialTask;
  maxHands = initialMaxHands;

  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
    if (vision.wasmLoaderPath.endsWith('/vision_wasm_internal.js')) {
      vision.wasmLoaderPath = `${MEDIAPIPE_WASM_ROOT}/vision_wasm_module_internal.js`;
      vision.wasmBinaryPath = `${MEDIAPIPE_WASM_ROOT}/vision_wasm_module_internal.wasm`;
    }
    visionFileset = vision;
    const gpu = detectWorkerGpu();

    // If user requested GPU and Worker OffscreenCanvas lacks WebGL2, inform client to try Main-Thread GPU
    if (initialMode === 'GPU' && !gpu.hasWebGl2) {
      self.postMessage({
        type: 'GPU_UNAVAILABLE',
        message: 'Worker OffscreenCanvas WebGL2 unavailable',
      });
      return;
    }

    await ensureActiveModel(initialTask, initialMode, initialMaxHands);
    activeDelegate = initialMode;

    const gpuInfo: GpuInfo =
      activeDelegate === 'GPU'
        ? { delegate: 'GPU', renderer: `${gpu.renderer} (Worker)`, tier: gpu.tier }
        : { delegate: 'CPU', renderer: 'Worker WASM SIMD', tier: 'SOFTWARE' };

    self.postMessage({ type: 'READY', gpuInfo, delegate: activeDelegate, taskMode });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Worker MediaPipe init failed';
    self.postMessage({
      type: initialMode === 'GPU' ? 'GPU_UNAVAILABLE' : 'ERROR',
      message: msg,
    });
  }
}

function sanitizeEngineMode(mode: unknown): AiEngineMode {
  return mode === 'CPU' ? 'CPU' : 'GPU';
}

function sanitizeTaskMode(mode: unknown): TrackerTaskMode {
  return mode === 'POSE' ? 'POSE' : 'HAND';
}

function sanitizeMaxHands(val: unknown): MaxHandsOption {
  return val === 1 ? 1 : 2;
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (!msg || typeof msg !== 'object') return;

  if (msg.type === 'INIT') {
    void initTracker(
      sanitizeEngineMode(msg.engineMode),
      sanitizeTaskMode(msg.taskMode),
      sanitizeMaxHands(msg.maxHands),
    );
    return;
  }

  if (msg.type === 'SET_ENGINE') {
    const nextEngine = sanitizeEngineMode(msg.engineMode);
    if (nextEngine === engineMode && nextEngine === activeDelegate) return;
    engineMode = nextEngine;
    void reconfigure(taskMode, nextEngine, maxHands);
    return;
  }

  if (msg.type === 'SET_TASK_MODE') {
    const nextTask = sanitizeTaskMode(msg.taskMode);
    if (nextTask === taskMode) return;
    void reconfigure(nextTask, activeDelegate, maxHands);
    return;
  }

  if (msg.type === 'SET_MAX_HANDS') {
    const nextMax = sanitizeMaxHands(msg.maxHands);
    if (nextMax === maxHands) return;
    maxHands = nextMax;
    if (taskMode === 'HAND') {
      void reconfigure('HAND', activeDelegate, nextMax);
    }
    return;
  }

  if (msg.type === 'CLOSE') {
    closeHandModel();
    closePoseModel();
    visionFileset = null;
    return;
  }

  if (msg.type === 'DETECT') {
    const { bitmap } = msg;
    if (!(bitmap instanceof ImageBitmap)) return;

    try {
      const currentTask = taskMode;
      const readyModel = currentTask === 'POSE' ? poseLandmarker : handLandmarker;
      if (!readyModel || isSwitching) {
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
      // Strictly release GPU/CPU bitmap memory immediately after every single frame
      try {
        bitmap.close();
      } catch {
        // already closed
      }
    }
  }
};
