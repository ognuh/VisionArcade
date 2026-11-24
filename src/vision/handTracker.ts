import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { GpuInfo, GpuTier } from '../types/vision';

const MEDIAPIPE_WASM_ROOT = '/mediapipe';

const HAND_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export interface HandTrackerInstance {
  landmarker: HandLandmarker;
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
    lower.includes('quadro') ||
    lower.includes('radeon rx') ||
    lower.includes('arc a') ||
    lower.includes('arc b')
  ) {
    return 'DISCRETE';
  }
  if (
    lower.includes('intel') ||
    lower.includes('uhd') ||
    lower.includes('iris') ||
    lower.includes('radeon(tm) graphics') ||
    lower.includes('vega') ||
    lower.includes('apple m')
  ) {
    return 'INTEGRATED';
  }
  return 'UNKNOWN';
}

function detectWebGlRenderer(): { renderer: string; tier: GpuTier; hasWebGl2: boolean } {
  try {
    const canvas = document.createElement('canvas');
    const gl2 = canvas.getContext('webgl2', { powerPreference: 'high-performance' });
    const gl = gl2 || canvas.getContext('webgl', { powerPreference: 'high-performance' });
    if (!gl) {
      return { renderer: 'Trình duyệt không bật WebGL', tier: 'SOFTWARE', hasWebGl2: false };
    }
    const debugExt = gl.getExtension('WEBGL_debug_renderer_info');
    const rawRenderer = debugExt
      ? String(gl.getParameter(debugExt.UNMASKED_RENDERER_WEBGL) ?? '')
      : String(gl.getParameter(gl.RENDERER) ?? '');

    const cleaned = rawRenderer
      .replace(/^ANGLE\s*\([^,]+,\s*/i, '')
      .replace(/\s*\(0x[0-9a-f]+\).*$/i, '')
      .replace(/\s*Direct3D11.*$/i, '')
      .replace(/\)\s*$/, '')
      .trim();

    const renderer = cleaned || rawRenderer || 'WebGL GPU';
    return { renderer, tier: classifyGpuTier(rawRenderer), hasWebGl2: Boolean(gl2) };
  } catch {
    return { renderer: 'Không rõ GPU', tier: 'UNKNOWN', hasWebGl2: false };
  }
}

export async function createHandTracker(forceGpu = false): Promise<HandTrackerInstance> {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
  const detected = detectWebGlRenderer();

  try {
    // Dedicated untouched canvas required by MediaPipe VisionTaskOptions for GPU texture binding.
    const gpuCanvas = document.createElement('canvas');
    gpuCanvas.width = 480;
    gpuCanvas.height = 270;

    const landmarker = await HandLandmarker.createFromOptions(vision, {
      canvas: gpuCanvas,
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
    return {
      landmarker,
      gpuInfo: {
        delegate: 'GPU',
        renderer: detected.renderer,
        tier: detected.tier,
      },
    };
  } catch (gpuError) {
    const reason = gpuError instanceof Error ? gpuError.message : String(gpuError);
    if (forceGpu) {
      throw new Error(
        `Không thể khởi chạy WebGL GPU (${detected.renderer}). Lý do: ${reason}. Hãy bật Hardware Acceleration trong trình duyệt hoặc tắt ép GPU.`,
      );
    }
    console.warn('GPU delegate unavailable, falling back to CPU:', gpuError);
    const landmarker = await HandLandmarker.createFromOptions(vision, {
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
    return {
      landmarker,
      gpuInfo: {
        delegate: 'CPU',
        renderer: `CPU (WASM) | ${detected.renderer}`,
        tier: 'SOFTWARE',
        fallbackReason: reason,
      },
    };
  }
}
