import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import type {
  AiEngineMode,
  GpuInfo,
  GpuTier,
  MaxHandsOption,
  TrackerTaskMode,
} from '../types/vision';

const MEDIAPIPE_WASM_ROOT = '/mediapipe';

const HAND_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const POSE_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export interface MainThreadTrackerInstance {
  handLandmarker: HandLandmarker | null;
  poseLandmarker: PoseLandmarker | null;
  gpuInfo: GpuInfo;
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

function detectWebGlRenderer(): { renderer: string; tier: GpuTier } {
  try {
    const canvas = document.createElement('canvas');
    const gl =
      canvas.getContext('webgl2', { powerPreference: 'high-performance' }) ||
      canvas.getContext('webgl', { powerPreference: 'high-performance' });
    if (!gl) return { renderer: 'WebGL Unavailable', tier: 'SOFTWARE' };
    const debugExt = gl.getExtension('WEBGL_debug_renderer_info');
    const raw = debugExt
      ? String(gl.getParameter(debugExt.UNMASKED_RENDERER_WEBGL) ?? '')
      : String(gl.getParameter(gl.RENDERER) ?? '');
    const cleaned = raw
      .replace(/^ANGLE\s*\([^,]+,\s*/i, '')
      .replace(/\s*\(0x[0-9a-f]+\).*$/i, '')
      .replace(/\s*Direct3D11.*$/i, '')
      .replace(/\)\s*$/, '')
      .trim();
    return { renderer: cleaned || raw || 'WebGL GPU', tier: classifyGpuTier(raw) };
  } catch {
    return { renderer: 'Unknown GPU', tier: 'UNKNOWN' };
  }
}

async function buildModel(
  vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>,
  delegate: 'GPU' | 'CPU',
  task: TrackerTaskMode,
  numHands: MaxHandsOption,
): Promise<{ handLandmarker: HandLandmarker | null; poseLandmarker: PoseLandmarker | null }> {
  const gpuCanvas =
    delegate === 'GPU'
      ? Object.assign(document.createElement('canvas'), { width: 256, height: 144 })
      : undefined;

  if (task === 'POSE') {
    const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      ...(gpuCanvas ? { canvas: gpuCanvas } : {}),
      baseOptions: { modelAssetPath: POSE_LANDMARKER_MODEL_URL, delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.45,
    });
    return { handLandmarker: null, poseLandmarker };
  }

  const handLandmarker = await HandLandmarker.createFromOptions(vision, {
    ...(gpuCanvas ? { canvas: gpuCanvas } : {}),
    baseOptions: { modelAssetPath: HAND_LANDMARKER_MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numHands,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.45,
  });
  return { handLandmarker, poseLandmarker: null };
}

/**
 * Main-Thread GPU/CPU tracker fallback (used ONLY if Worker OffscreenCanvas lacks WebGL2):
 * Keeps GPU delegate when requested; falls back to CPU only on genuine GPU hardware failure or CPU mode.
 */
export async function createMainThreadTracker(
  wantDelegate: AiEngineMode,
  task: TrackerTaskMode,
  numHands: MaxHandsOption,
): Promise<MainThreadTrackerInstance> {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
  const detected = detectWebGlRenderer();

  if (wantDelegate === 'CPU') {
    const models = await buildModel(vision, 'CPU', task, numHands);
    return {
      ...models,
      gpuInfo: { delegate: 'CPU', renderer: `CPU (WASM) | ${detected.renderer}`, tier: 'SOFTWARE' },
    };
  }

  try {
    const models = await buildModel(vision, 'GPU', task, numHands);
    return {
      ...models,
      gpuInfo: { delegate: 'GPU', renderer: detected.renderer, tier: detected.tier },
    };
  } catch (gpuError) {
    const reason = gpuError instanceof Error ? gpuError.message : String(gpuError);
    console.warn('Main-thread GPU delegate unavailable, falling back to CPU:', gpuError);
    const models = await buildModel(vision, 'CPU', task, numHands);
    return {
      ...models,
      gpuInfo: {
        delegate: 'CPU',
        renderer: `CPU (WASM) | ${detected.renderer}`,
        tier: 'SOFTWARE',
        fallbackReason: reason,
      },
    };
  }
}
