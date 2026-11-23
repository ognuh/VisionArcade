import type { CycleModeId } from '../types/vision';
import type { HandPose2D, Point2D, RenderPolygon } from './types';

function ptOf(h: HandPose2D, idx: number): Point2D {
  return { x: h.pts[idx * 2], y: h.pts[idx * 2 + 1] };
}

export function polygonArea(pts: Point2D[]): number {
  if (pts.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < pts.length; i += 1) {
    const a = pts[i];
    const b = pts[(i + 1) % pts.length];
    sum += a.x * b.y - b.x * a.y;
  }
  return Math.abs(sum) * 0.5;
}

/** Builds vivid Glass, Thermal, and Chromatic Prism polygons from tracked hands. */
export function buildOpticalPolygons(hands: HandPose2D[], activeMode: CycleModeId): RenderPolygon[] {
  if (hands.length === 0) return [];
  if (activeMode !== 'GLASS' && activeMode !== 'THERMAL' && activeMode !== 'PRISM') {
    return [];
  }

  if (activeMode === 'GLASS') {
    const h = hands[0];
    const tri: Point2D[] = [ptOf(h, 4), ptOf(h, 8), ptOf(h, 12)];
    if (polygonArea(tri) > 320) {
      return [
        {
          pts: tri,
          shaderType: 0,
          tint: [1, 1, 1, 1],
          additive: false,
          glow: '#e0f2fe',
        },
      ];
    }
    return [];
  }

  if (activeMode === 'THERMAL') {
    let tri: Point2D[];
    if (hands.length >= 2) {
      const L = hands[0];
      const R = hands[1];
      const candA = [ptOf(L, 8), ptOf(R, 4), ptOf(R, 8)];
      const candB = [ptOf(L, 8), ptOf(R, 4), ptOf(R, 20)];
      const candC = [ptOf(R, 8), ptOf(L, 4), ptOf(L, 20)];
      const areaA = polygonArea(candA);
      const areaB = polygonArea(candB);
      const areaC = polygonArea(candC);
      tri = areaC > areaB && areaC > areaA ? candC : areaB >= areaA ? candB : candA;
    } else {
      const h = hands[0];
      tri = [ptOf(h, 4), ptOf(h, 8), ptOf(h, 20)];
    }
    if (polygonArea(tri) > 360) {
      return [
        {
          pts: tri,
          shaderType: 1,
          tint: [1, 1, 1, 1],
          additive: false,
          glow: '#fde047',
        },
      ];
    }
    return [];
  }

  // PRISM: 3 vivid overlapping chromatic ribbons between Left & Right fingertips
  if (hands.length >= 2) {
    const L = hands[0];
    const R = hands[1];
    const candidates: RenderPolygon[] = [
      {
        pts: [ptOf(L, 4), ptOf(L, 8), ptOf(R, 8), ptOf(R, 4)],
        shaderType: 2,
        tint: [0.05, 0.72, 1.0, 0.94], // Vivid Electric Cyan-Blue
        additive: true,
        glow: '#22d3ee',
      },
      {
        pts: [ptOf(L, 8), ptOf(L, 12), ptOf(R, 12), ptOf(R, 8)],
        shaderType: 2,
        tint: [0.46, 1.0, 0.14, 0.94], // Electric Lime-Yellow
        additive: true,
        glow: '#a3e635',
      },
      {
        pts: [ptOf(L, 12), ptOf(L, 16), ptOf(R, 16), ptOf(R, 12)],
        shaderType: 2,
        tint: [1.0, 0.14, 0.96, 0.94], // Hot Neon Magenta
        additive: true,
        glow: '#f43f97',
      },
    ];
    return candidates.filter((c) => polygonArea(c.pts) > 260);
  }

  const h = hands[0];
  const tri = [ptOf(h, 4), ptOf(h, 8), ptOf(h, 12)];
  if (polygonArea(tri) > 320) {
    return [
      {
        pts: tri,
        shaderType: 2,
        tint: [0.08, 0.76, 1.0, 0.92],
        additive: true,
        glow: '#38bdf8',
      },
    ];
  }
  return [];
}

function tracePolyOffset(ctx: CanvasRenderingContext2D, pts: Point2D[], ox: number, oy: number): void {
  ctx.beginPath();
  ctx.moveTo(pts[0].x + ox, pts[0].y + oy);
  for (let i = 1; i < pts.length; i += 1) {
    ctx.lineTo(pts[i].x + ox, pts[i].y + oy);
  }
  ctx.closePath();
}

/**
 * Multi-Pass Chromatic Aberration Edge Rendering (0 shadowBlur):
 * Offsets razor-sharp Cyan (-2,-1) & Magenta (+2,+1) aberration rims underneath
 * a wide soft neon aura and pure white specular core so polygons pop off dark scenes.
 */
export function renderPrismChromaticEdges(
  ctx: CanvasRenderingContext2D,
  polygons: RenderPolygon[],
  fadeAlpha = 1,
): void {
  if (polygons.length === 0 || fadeAlpha <= 0.01) return;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineJoin = 'round';

  for (let i = 0; i < polygons.length; i += 1) {
    const poly = polygons[i];
    const pts = poly.pts;
    if (pts.length < 3) continue;

    // 1. Wide soft neon outer aura
    tracePolyOffset(ctx, pts, 0, 0);
    ctx.strokeStyle = poly.glow;
    ctx.lineWidth = 9;
    ctx.globalAlpha = 0.2 * fadeAlpha;
    ctx.stroke();

    // 2. Chromatic aberration split edges (Cyan left-up, Magenta right-down)
    tracePolyOffset(ctx, pts, -2.2, -1.2);
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 1.6;
    ctx.globalAlpha = 0.78 * fadeAlpha;
    ctx.stroke();

    tracePolyOffset(ctx, pts, 2.2, 1.2);
    ctx.strokeStyle = '#f43f97';
    ctx.lineWidth = 1.6;
    ctx.globalAlpha = 0.78 * fadeAlpha;
    ctx.stroke();

    // 3. Crisp primary color rim + pure white specular core
    tracePolyOffset(ctx, pts, 0, 0);
    ctx.strokeStyle = poly.glow;
    ctx.lineWidth = 2.4;
    ctx.globalAlpha = 0.85 * fadeAlpha;
    ctx.stroke();

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = 0.94 * fadeAlpha;
    ctx.stroke();
  }

  ctx.restore();
}

/** Zero-lag Multi-Pass outline for Motion Trail history frames. */
export function renderTrailOutline(
  ctx: CanvasRenderingContext2D,
  pts: Point2D[],
  color: string,
  alpha: number,
): void {
  if (pts.length < 3) return;
  tracePolyOffset(ctx, pts, 0, 0);
  ctx.strokeStyle = color;
  ctx.lineWidth = 6.5;
  ctx.globalAlpha = alpha * 0.24;
  ctx.stroke();

  ctx.lineWidth = 1.5;
  ctx.globalAlpha = alpha;
  ctx.stroke();
}
