import type { PrismModeId } from '../types/vision';
import type { HandPose2D } from '../fx/types';

const HAND_BONES: ReadonlyArray<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const FINGERTIPS = [4, 8, 12, 16, 20];
const INTERMEDIATE_JOINTS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19];

function modeNeonColor(mode: PrismModeId): string {
  switch (mode) {
    case 'THERMAL':
      return '#fb923c';
    case 'PRISM':
      return '#f472b6';
    case 'COSMOS':
      return '#a855f7';
    case 'LIGHTNING':
      return '#38bdf8';
    case 'MYSTIC':
      return '#f59e0b';
    case 'RIPPLE':
      return '#67e8f9';
    case 'FLUID':
      return '#818cf8';
    default:
      return '#22d3ee';
  }
}

/**
 * Cybernetic HUD Hand Skeleton:
 * - 2-layer bones: 4px neon aura synced to active mode + 1.2px crisp white core
 * - Rotating targeting reticles on 5 fingertips (4, 8, 12, 16, 20)
 * - Hollow micro-hexagons on intermediate joints
 * - Palm energy reticle with degree ticks + 10px monospace telemetry beside wrist
 */
export function drawCyberneticHandHud(
  ctx: CanvasRenderingContext2D,
  hands: HandPose2D[],
  activeMode: PrismModeId,
  elapsedSec: number,
): void {
  if (hands.length === 0) return;

  const neon = modeNeonColor(activeMode);
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.font = '500 10px "IBM Plex Mono", monospace';
  ctx.textBaseline = 'middle';

  for (let hIdx = 0; hIdx < hands.length; hIdx += 1) {
    const h = hands[hIdx];
    const pts = h.pts;

    // 1. Two-layer bones: 4px neon glow under 1.2px pure white core
    ctx.globalCompositeOperation = 'lighter';
    ctx.beginPath();
    for (let b = 0; b < HAND_BONES.length; b += 1) {
      const [a, c] = HAND_BONES[b];
      ctx.moveTo(pts[a * 2], pts[a * 2 + 1]);
      ctx.lineTo(pts[c * 2], pts[c * 2 + 1]);
    }
    ctx.strokeStyle = neon;
    ctx.lineWidth = 4;
    ctx.globalAlpha = 0.42;
    ctx.stroke();

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = 0.92;
    ctx.stroke();

    // 2. Hollow technical micro-hexagons on intermediate joints
    ctx.strokeStyle = neon;
    ctx.lineWidth = 1.1;
    ctx.globalAlpha = 0.85;
    for (let j = 0; j < INTERMEDIATE_JOINTS.length; j += 1) {
      const idx = INTERMEDIATE_JOINTS[j];
      const jx = pts[idx * 2];
      const jy = pts[idx * 2 + 1];
      ctx.beginPath();
      for (let s = 0; s < 6; s += 1) {
        const a = (s * Math.PI) / 3;
        const hx = jx + Math.cos(a) * 3.6;
        const hy = jy + Math.sin(a) * 3.6;
        if (s === 0) ctx.moveTo(hx, hy);
        else ctx.lineTo(hx, hy);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // 3. Rotating precision reticles on 5 fingertips
    const spin = elapsedSec * 2.2;
    for (let t = 0; t < FINGERTIPS.length; t += 1) {
      const idx = FINGERTIPS[t];
      const tx = pts[idx * 2];
      const ty = pts[idx * 2 + 1];

      ctx.strokeStyle = neon;
      ctx.lineWidth = 1.3;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.arc(tx, ty, 7.5, spin, spin + Math.PI * 0.72);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(tx, ty, 7.5, spin + Math.PI, spin + Math.PI * 1.72);
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.globalAlpha = 0.95;
      ctx.beginPath();
      ctx.arc(tx, ty, 2.0, 0, Math.PI * 2);
      ctx.fill();
    }

    // 4. Palm center segmented degree-tick ring
    const pr = Math.max(14, Math.min(26, h.palmScale * 0.22));
    ctx.strokeStyle = neon;
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    ctx.arc(h.palmX, h.palmY, pr, -spin * 0.8, -spin * 0.8 + Math.PI * 1.55);
    ctx.stroke();

    for (let k = 0; k < 8; k += 1) {
      const a = (k * Math.PI) / 4 + spin * 0.4;
      ctx.beginPath();
      ctx.moveTo(h.palmX + Math.cos(a) * (pr - 3), h.palmY + Math.sin(a) * (pr - 3));
      ctx.lineTo(h.palmX + Math.cos(a) * (pr + 3), h.palmY + Math.sin(a) * (pr + 3));
      ctx.stroke();
    }

    // 5. Micro monospace telemetry beside wrist (#0)
    const wx = pts[0];
    const wy = pts[1];
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 0.88;
    ctx.fillStyle = '#ffffff';
    const label = `VEL:${h.palmSpeedMps.toFixed(1)}m/s | ${h.gesture} | ${activeMode}`;
    ctx.fillText(label, wx + 14, wy + 4);
  }

  ctx.restore();
}
