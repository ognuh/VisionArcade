import type { Landmark3D } from '../types/vision';

export type MockPoseId =
  | 'NONE'
  | 'ONE_FINGER'
  | 'TWO_FINGERS'
  | 'OPEN_ONE'
  | 'FIST_ONE'
  | 'OPEN_TWO_HANDS'
  | 'QUAD_TWO_HANDS'
  | 'AIR_SWIPE_RIGHT'
  | 'AIR_SWIPE_LEFT'
  | 'AIR_SWIPE_RETURN'
  | 'DOCK_HOVER_CENTER'
  | 'RIPPLE_WAVE_BURST'
  | 'FLUID_INTERACT'
  | 'HAND_BLINK'
  | 'FIST_BOTH'
  | 'RUNNER_STAND'
  | 'RUNNER_HANDS_UP'
  | 'RUNNER_LEFT'
  | 'RUNNER_RIGHT'
  | 'RUNNER_JUMP'
  | 'RUNNER_CROUCH'
  | 'RUNNER_DEMO_SEQ';

interface FingerSpec {
  mcp: [number, number];
  dir: [number, number];
  extended: boolean;
  len: number;
}

/** Builds a anatomically proportioned 21-landmark MediaPipe hand around (cx, cy). */
function buildMockHand(
  cx: number,
  cy: number,
  scale: number,
  fingers: { thumb: boolean; index: boolean; middle: boolean; ring: boolean; pinky: boolean },
  flipX = 1,
): Landmark3D[] {
  const pts: Landmark3D[] = new Array(21);
  const s = scale;
  // 0: Wrist
  pts[0] = { x: cx, y: cy + s * 0.62, z: 0 };

  // Thumb (1, 2, 3, 4)
  pts[1] = { x: cx - flipX * s * 0.24, y: cy + s * 0.44, z: -0.01 };
  pts[2] = { x: cx - flipX * s * 0.36, y: cy + s * 0.22, z: -0.015 };
  if (fingers.thumb) {
    pts[3] = { x: cx - flipX * s * 0.56, y: cy + s * 0.04, z: -0.02 };
    pts[4] = { x: cx - flipX * s * 0.72, y: cy - s * 0.12, z: -0.025 };
  } else {
    pts[3] = { x: cx - flipX * s * 0.22, y: cy + s * 0.16, z: -0.02 };
    pts[4] = { x: cx - flipX * s * 0.10, y: cy + s * 0.18, z: -0.025 };
  }

  const specs: FingerSpec[] = [
    { mcp: [-flipX * 0.26, -0.10], dir: [-flipX * 0.12, -1.0], extended: fingers.index, len: 0.76 },
    { mcp: [-flipX * 0.06, -0.16], dir: [-flipX * 0.02, -1.0], extended: fingers.middle, len: 0.84 },
    { mcp: [flipX * 0.13, -0.12], dir: [flipX * 0.08, -1.0], extended: fingers.ring, len: 0.78 },
    { mcp: [flipX * 0.30, -0.02], dir: [flipX * 0.20, -0.96], extended: fingers.pinky, len: 0.66 },
  ];

  for (let f = 0; f < 4; f += 1) {
    const sp = specs[f];
    const base = 5 + f * 4;
    const mx = cx + sp.mcp[0] * s;
    const my = cy + sp.mcp[1] * s;
    pts[base] = { x: mx, y: my, z: 0 };

    if (sp.extended) {
      pts[base + 1] = { x: mx + sp.dir[0] * s * sp.len * 0.42, y: my + sp.dir[1] * s * sp.len * 0.42, z: -0.01 };
      pts[base + 2] = { x: mx + sp.dir[0] * s * sp.len * 0.74, y: my + sp.dir[1] * s * sp.len * 0.74, z: -0.015 };
      pts[base + 3] = { x: mx + sp.dir[0] * s * sp.len * 1.04, y: my + sp.dir[1] * s * sp.len * 1.04, z: -0.02 };
    } else {
      // Curled inward toward palm center (tip closer to wrist than PIP)
      pts[base + 1] = { x: mx + sp.dir[0] * s * 0.26, y: my - s * 0.20, z: -0.01 };
      pts[base + 2] = { x: mx + sp.dir[0] * s * 0.10, y: my + s * 0.05, z: -0.02 };
      pts[base + 3] = { x: mx, y: my + s * 0.24, z: -0.025 };
    }
  }

  return pts;
}

/** Generates deterministic synthetic 2-hand landmarks for ?test=1 automated testing. */
export function generateMockHands(pose: MockPoseId, elapsedSec: number): Landmark3D[][] {
  if (pose === 'NONE') return [];

  if (pose === 'ONE_FINGER') {
    const driftX = Math.sin(elapsedSec * 1.4) * 0.012;
    return [
      buildMockHand(
        0.5 + driftX,
        0.67,
        0.16,
        { thumb: false, index: true, middle: false, ring: false, pinky: false },
        1,
      ),
    ];
  }

  if (pose === 'TWO_FINGERS') {
    return [
      buildMockHand(
        0.5,
        0.67,
        0.16,
        { thumb: false, index: true, middle: true, ring: false, pinky: false },
        1,
      ),
    ];
  }

  if (pose === 'OPEN_ONE') {
    return [
      buildMockHand(
        0.5,
        0.54,
        0.16,
        { thumb: true, index: true, middle: true, ring: true, pinky: true },
        1,
      ),
    ];
  }

  if (pose === 'FIST_ONE') {
    return [
      buildMockHand(
        0.5,
        0.54,
        0.16,
        { thumb: false, index: false, middle: false, ring: false, pinky: false },
        1,
      ),
    ];
  }

  if (pose === 'OPEN_TWO_HANDS') {
    // Wide-open 2 hands (5 extended fingers each, norm dist ~0.48 >= 0.28, slight tilt)
    return [
      buildMockHand(
        0.26,
        0.54,
        0.15,
        { thumb: true, index: true, middle: true, ring: true, pinky: true },
        1,
      ),
      buildMockHand(
        0.74,
        0.48,
        0.15,
        { thumb: true, index: true, middle: true, ring: true, pinky: true },
        -1,
      ),
    ];
  }

  if (pose === 'QUAD_TWO_HANDS') {
    // 2 hands extending Index (8) + Pinky (20) to verify full 4-point quad masks
    return [
      buildMockHand(
        0.30,
        0.58,
        0.16,
        { thumb: false, index: true, middle: false, ring: false, pinky: true },
        1,
      ),
      buildMockHand(
        0.70,
        0.58,
        0.16,
        { thumb: false, index: true, middle: false, ring: false, pinky: true },
        -1,
      ),
    ];
  }

  if (pose === 'AIR_SWIPE_RIGHT' || pose === 'AIR_SWIPE_LEFT') {
    // Smooth 240ms horizontal swipe inside mid-screen Y=0.45 (mirrored coords: rawX decreasing = screen right!)
    const t = Math.min(1, elapsedSec / 0.24);
    const rawX = pose === 'AIR_SWIPE_RIGHT' ? 0.74 - t * 0.48 : 0.26 + t * 0.48;
    return [
      buildMockHand(
        rawX,
        0.45,
        0.15,
        { thumb: true, index: true, middle: true, ring: true, pinky: true },
        1,
      ),
    ];
  }

  if (pose === 'AIR_SWIPE_RETURN') {
    // Swipes RIGHT (0..0.24s) then immediately pulls hand back LEFT (0.28..0.58s) to prove Return-Stroke Lock!
    let rawX = 0.26;
    if (elapsedSec <= 0.24) {
      rawX = 0.74 - (elapsedSec / 0.24) * 0.48;
    } else if (elapsedSec <= 0.58) {
      const k = Math.min(1, (elapsedSec - 0.24) / 0.34);
      rawX = 0.26 + k * 0.48;
    } else {
      rawX = 0.74;
    }
    return [
      buildMockHand(
        rawX,
        0.45,
        0.15,
        { thumb: true, index: true, middle: true, ring: true, pinky: true },
        1,
      ),
    ];
  }

  if (pose === 'DOCK_HOVER_CENTER') {
    // Index finger tip aimed directly at bottom-center Floating Dock (y ~ 0.945 in 720p)
    return [
      buildMockHand(
        0.52,
        1.07,
        0.15,
        { thumb: false, index: true, middle: false, ring: false, pinky: false },
        1,
      ),
    ];
  }

  if (pose === 'RIPPLE_WAVE_BURST') {
    // 0..0.7s smooth hand sweep creating liquid wake -> 0.7..0.95s tight Fist -> 0.95s+ Open Shockwave Burst!
    if (elapsedSec < 0.7) {
      const t = elapsedSec / 0.7;
      const cx = 0.25 + t * 0.50;
      const cy = 0.48 + Math.sin(t * Math.PI * 2) * 0.14;
      return [
        buildMockHand(
          cx,
          cy,
          0.15,
          { thumb: true, index: true, middle: false, ring: false, pinky: false },
          1,
        ),
      ];
    }
    if (elapsedSec < 0.95) {
      return [
        buildMockHand(
          0.5,
          0.5,
          0.15,
          { thumb: false, index: false, middle: false, ring: false, pinky: false },
          1,
        ),
      ];
    }
    return [
      buildMockHand(
        0.5,
        0.5,
        0.16,
        { thumb: true, index: true, middle: true, ring: true, pinky: true },
        1,
      ),
    ];
  }

  if (pose === 'FLUID_INTERACT') {
    // 0..0.55s gentle 2-hand sweep -> 0.55..1.35s Pinch Vortex (thumb 4 & index 8 touching) -> 1.35s+ Open Supernova Burst!
    if (elapsedSec < 0.55) {
      const t = elapsedSec / 0.55;
      return [
        buildMockHand(
          0.25 + t * 0.22,
          0.52 + Math.sin(t * Math.PI) * 0.1,
          0.15,
          { thumb: false, index: true, middle: true, ring: false, pinky: false },
          1,
        ),
        buildMockHand(
          0.75 - t * 0.22,
          0.48 - Math.sin(t * Math.PI) * 0.1,
          0.15,
          { thumb: false, index: true, middle: true, ring: false, pinky: false },
          -1,
        ),
      ];
    }
    if (elapsedSec < 1.35) {
      const hand = buildMockHand(
        0.5,
        0.5,
        0.15,
        { thumb: false, index: false, middle: false, ring: false, pinky: false },
        1,
      );
      // Force thumb tip (4) & index tip (8) to pinch tightly together (< 0.015 dist)
      hand[4] = { x: 0.495, y: 0.46, z: -0.02 };
      hand[8] = { x: 0.505, y: 0.46, z: -0.02 };
      return [hand];
    }
    return [
      buildMockHand(
        0.5,
        0.5,
        0.16,
        { thumb: true, index: true, middle: true, ring: true, pinky: true },
        1,
      ),
    ];
  }

  if (pose === 'HAND_BLINK') {
    const open = Math.floor(elapsedSec * 5.5) % 2 === 0;
    return [
      buildMockHand(
        0.5,
        0.58,
        0.15,
        { thumb: open, index: open, middle: open, ring: open, pinky: open },
        1,
      ),
    ];
  }

  // FIST_BOTH (and Runner full-body poses return no hands)
  if (pose.startsWith('RUNNER_')) return [];

  return [
    buildMockHand(
      0.32,
      0.56,
      0.15,
      { thumb: false, index: false, middle: false, ring: false, pinky: false },
      1,
    ),
    buildMockHand(
      0.68,
      0.56,
      0.15,
      { thumb: false, index: false, middle: false, ring: false, pinky: false },
      -1,
    ),
  ];
}

function buildMockBodyPose(
  rawMidX: number,
  midY: number,
  shoulderW: number,
  handsUp: boolean,
): Landmark3D[] {
  const pts: Landmark3D[] = Array.from({ length: 33 }, () => ({
    x: rawMidX,
    y: midY + 0.25,
    z: 0,
    visibility: 0.95,
  }));
  const halfW = shoulderW * 0.5;
  pts[0] = { x: rawMidX, y: midY - shoulderW * 0.72, z: -0.05, visibility: 0.98 };
  pts[11] = { x: rawMidX + halfW, y: midY, z: 0, visibility: 0.99 };
  pts[12] = { x: rawMidX - halfW, y: midY, z: 0, visibility: 0.99 };
  if (handsUp) {
    pts[13] = { x: rawMidX + halfW * 1.35, y: midY - shoulderW * 0.45, z: -0.02, visibility: 0.97 };
    pts[14] = { x: rawMidX - halfW * 1.35, y: midY - shoulderW * 0.45, z: -0.02, visibility: 0.97 };
    pts[15] = { x: rawMidX + halfW * 1.15, y: midY - shoulderW * 1.28, z: -0.04, visibility: 0.97 };
    pts[16] = { x: rawMidX - halfW * 1.15, y: midY - shoulderW * 1.28, z: -0.04, visibility: 0.97 };
  } else {
    pts[13] = { x: rawMidX + halfW * 1.28, y: midY + shoulderW * 0.62, z: -0.02, visibility: 0.97 };
    pts[14] = { x: rawMidX - halfW * 1.28, y: midY + shoulderW * 0.62, z: -0.02, visibility: 0.97 };
    pts[15] = { x: rawMidX + halfW * 1.12, y: midY + shoulderW * 1.18, z: -0.04, visibility: 0.97 };
    pts[16] = { x: rawMidX - halfW * 1.12, y: midY + shoulderW * 1.18, z: -0.04, visibility: 0.97 };
  }
  pts[23] = { x: rawMidX + halfW * 0.72, y: midY + shoulderW * 1.38, z: 0, visibility: 0.92 };
  pts[24] = { x: rawMidX - halfW * 0.72, y: midY + shoulderW * 1.38, z: 0, visibility: 0.92 };
  return pts;
}

/** Generates deterministic 33-landmark MediaPipe Pose frames for ?test=1 Runner verification. */
export function generateMockPose(pose: MockPoseId, elapsedSec: number): Landmark3D[] | null {
  const sw = 0.22;
  const baseY = 0.52;
  if (pose === 'RUNNER_STAND') return buildMockBodyPose(0.5, baseY, sw, false);
  if (pose === 'RUNNER_HANDS_UP') return buildMockBodyPose(0.5, baseY, sw, true);
  if (pose === 'RUNNER_LEFT') return buildMockBodyPose(0.65, baseY, sw, false);
  if (pose === 'RUNNER_RIGHT') return buildMockBodyPose(0.35, baseY, sw, false);
  if (pose === 'RUNNER_JUMP') return buildMockBodyPose(0.5, baseY - 0.1, sw, false);
  if (pose === 'RUNNER_CROUCH') return buildMockBodyPose(0.5, baseY + 0.1, sw, false);
  if (pose === 'RUNNER_DEMO_SEQ') {
    const phase = elapsedSec % 3.0;
    if (phase < 0.45) return buildMockBodyPose(0.5, baseY, sw, false);
    if (phase < 1.05) return buildMockBodyPose(0.65, baseY, sw, false);
    if (phase < 1.65) return buildMockBodyPose(0.35, baseY, sw, false);
    if (phase < 2.25) return buildMockBodyPose(0.5, baseY - 0.1, sw, false);
    return buildMockBodyPose(0.5, baseY + 0.1, sw, false);
  }
  return buildMockBodyPose(0.5, baseY, sw, false);
}

/** Creates an animated studio-lit synthetic video MediaStream for headless ?test=1 verification. */
export function createMockCameraStream(): { stream: MediaStream; stop: () => void } {
  const canvas = document.createElement('canvas');
  canvas.width = 1280;
  canvas.height = 720;
  const ctx = canvas.getContext('2d')!;
  let running = true;
  let timerId = 0;
  let tick = 0;

  const drawFrame = () => {
    if (!running) return;
    tick += 0.035;

    // Moody studio backdrop (deep slate-indigo gradient + warm key light silhouette)
    const bg = ctx.createLinearGradient(0, 0, 1280, 720);
    bg.addColorStop(0, '#0f172a');
    bg.addColorStop(0.5, '#1e293b');
    bg.addColorStop(1, '#090d16');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 1280, 720);

    // Subtle geometric studio light bars so Glass/Thermal/Prism optical distortion is clearly visible
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.22)';
    ctx.lineWidth = 2;
    for (let x = 80; x < 1280; x += 96) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 720);
      ctx.stroke();
    }
    for (let y = 72; y < 720; y += 96) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(1280, y);
      ctx.stroke();
    }

    // Warm studio portrait silhouette in upper-center so Face Blur / Pixel / Cyber Mask are clearly verifiable
    const faceX = 640 + Math.sin(tick * 0.6) * 14;
    const faceY = 238;
    ctx.fillStyle = '#d6a582';
    ctx.beginPath();
    ctx.ellipse(faceX, faceY, 82, 102, 0, 0, Math.PI * 2);
    ctx.fill();
    // Stylized eyes & features so Blur/Pixel effect is clearly visible against grid
    ctx.fillStyle = '#1e293b';
    ctx.beginPath();
    ctx.arc(faceX - 28, faceY - 14, 9, 0, Math.PI * 2);
    ctx.arc(faceX + 28, faceY - 14, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#7c2d12';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(faceX, faceY + 26, 22, 0.15, Math.PI - 0.15);
    ctx.stroke();

    // Warm studio portrait glow in center-background
    const spotX = 640 + Math.sin(tick) * 28;
    const spotY = 390 + Math.cos(tick * 0.7) * 16;
    const spot = ctx.createRadialGradient(spotX, spotY, 30, spotX, spotY, 340);
    spot.addColorStop(0, 'rgba(251, 191, 136, 0.55)');
    spot.addColorStop(0.45, 'rgba(99, 102, 241, 0.26)');
    spot.addColorStop(1, 'rgba(15, 23, 42, 0)');
    ctx.fillStyle = spot;
    ctx.fillRect(0, 0, 1280, 720);

    timerId = window.setTimeout(drawFrame, 16);
  };

  drawFrame();
  const stream = canvas.captureStream(60);
  return {
    stream,
    stop: () => {
      running = false;
      window.clearTimeout(timerId);
      stream.getTracks().forEach((t) => t.stop());
    },
  };
}
