import type { CycleModeId } from '../types/vision';
import type { HandPose2D, Point2D, RenderPolygon } from './types';

function ptOf(h: HandPose2D, idx: number): Point2D {
  return { x: h.pts[idx * 2], y: h.pts[idx * 2 + 1] };
}

function distPx(h: HandPose2D, a: number, b: number): number {
  return Math.hypot(h.pts[a * 2] - h.pts[b * 2], h.pts[a * 2 + 1] - h.pts[b * 2 + 1]);
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

/**
 * Sorts polygon vertices clockwise around their centroid (polar angle sort)
 * so 4-point quads NEVER twist into a bowtie when rotating, flipping, or crossing hands.
 */
export function sortVerticesClockwise(pts: Point2D[]): Point2D[] {
  if (pts.length <= 2) return pts;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < pts.length; i += 1) {
    cx += pts[i].x;
    cy += pts[i].y;
  }
  cx /= pts.length;
  cy /= pts.length;
  return [...pts].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx),
  );
}

/** Picks a hand's 2 primary anchor vertices (Index 8 + Pinky 20, or Index 8 + Thumb 4) unless only 1 finger is extended. */
function extractHandAnchors(h: HandPose2D): Point2D[] {
  const indexExt = distPx(h, 8, 0) > distPx(h, 6, 0) * 1.05;
  const pinkyExt = distPx(h, 20, 0) > distPx(h, 18, 0) * 1.04;
  const thumbExt = distPx(h, 4, 17) > distPx(h, 3, 17) * 1.02;

  // If strictly pointing 1 finger or pinched -> single vertex anchor (allows intentional 3-point triangle with other hand)
  if (h.gesture === 'POINT' && !pinkyExt) {
    return [ptOf(h, 8)];
  }
  if (h.gesture === 'PINCH' && !pinkyExt) {
    const t = ptOf(h, 4);
    const idx = ptOf(h, 8);
    return [{ x: (t.x + idx.x) * 0.5, y: (t.y + idx.y) * 0.5 }];
  }

  // Prefer Index (8) + Pinky (20) when pinky is raised; otherwise Index (8) + Thumb (4)
  if (indexExt && pinkyExt) {
    return [ptOf(h, 8), ptOf(h, 20)];
  }
  if (indexExt && thumbExt) {
    return [ptOf(h, 8), ptOf(h, 4)];
  }

  // Fallback: pick whichever pair (8-20 vs 8-4) spans wider
  return distPx(h, 8, 20) >= distPx(h, 8, 4)
    ? [ptOf(h, 8), ptOf(h, 20)]
    : [ptOf(h, 8), ptOf(h, 4)];
}

/**
 * Builds Glass, Thermal, and Chromatic Prism masks:
 * - 2 hands with Index+Pinky (8, 20) or Index+Thumb (8, 4) -> full 4-vertex clockwise-sorted Quad!
 * - 1 hand (or 2 hands where 1 hand points a single finger) -> crisp 3-vertex Triangle!
 */
export function buildOpticalPolygons(hands: HandPose2D[], activeMode: CycleModeId): RenderPolygon[] {
  if (hands.length === 0) return [];
  if (activeMode !== 'GLASS' && activeMode !== 'THERMAL' && activeMode !== 'PRISM') {
    return [];
  }

  if (activeMode === 'GLASS') {
    if (hands.length >= 2) {
      const aL = extractHandAnchors(hands[0]);
      const aR = extractHandAnchors(hands[1]);
      const quad = sortVerticesClockwise([...aL, ...aR]);
      if (quad.length >= 3 && polygonArea(quad) > 360) {
        return [
          {
            pts: quad,
            shaderType: 0,
            tint: [1, 1, 1, 1],
            additive: false,
            glow: '#e0f2fe',
          },
        ];
      }
      return [];
    }

    const h = hands[0];
    const tri = sortVerticesClockwise([ptOf(h, 4), ptOf(h, 8), ptOf(h, 12)]);
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
    if (hands.length >= 2) {
      const aL = extractHandAnchors(hands[0]);
      const aR = extractHandAnchors(hands[1]);
      const poly = sortVerticesClockwise([...aL, ...aR]);
      if (poly.length >= 3 && polygonArea(poly) > 360) {
        return [
          {
            pts: poly,
            shaderType: 1,
            tint: [1, 1, 1, 1],
            additive: false,
            glow: '#fde047',
          },
        ];
      }
      return [];
    }

    const h = hands[0];
    const tri = sortVerticesClockwise([ptOf(h, 4), ptOf(h, 8), ptOf(h, 20)]);
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

  // PRISM: 2-hand clockwise-sorted 4-point quads (never bowtie!)
  if (hands.length >= 2) {
    const L = hands[0];
    const R = hands[1];

    // Check if both hands are specifically holding Index(8)+Pinky(20) frame pose (middle/ring folded)
    const lMidFolded = distPx(L, 12, 0) < distPx(L, 10, 0) * 1.05;
    const rMidFolded = distPx(R, 12, 0) < distPx(R, 10, 0) * 1.05;
    if (lMidFolded && rMidFolded) {
      const quad = sortVerticesClockwise([ptOf(L, 8), ptOf(R, 8), ptOf(R, 20), ptOf(L, 20)]);
      if (polygonArea(quad) > 320) {
        return [
          {
            pts: quad,
            shaderType: 2,
            tint: [0.08, 0.78, 1.0, 0.94],
            additive: true,
            glow: '#22d3ee',
          },
        ];
      }
      return [];
    }

    const candidates: RenderPolygon[] = [
      {
        pts: sortVerticesClockwise([ptOf(L, 4), ptOf(L, 8), ptOf(R, 8), ptOf(R, 4)]),
        shaderType: 2,
        tint: [0.05, 0.72, 1.0, 0.94],
        additive: true,
        glow: '#22d3ee',
      },
      {
        pts: sortVerticesClockwise([ptOf(L, 8), ptOf(L, 12), ptOf(R, 12), ptOf(R, 8)]),
        shaderType: 2,
        tint: [0.46, 1.0, 0.14, 0.94],
        additive: true,
        glow: '#a3e635',
      },
      {
        pts: sortVerticesClockwise([ptOf(L, 12), ptOf(L, 20), ptOf(R, 20), ptOf(R, 12)]),
        shaderType: 2,
        tint: [1.0, 0.14, 0.96, 0.94],
        additive: true,
        glow: '#f43f97',
      },
    ];
    return candidates.filter((c) => polygonArea(c.pts) > 260);
  }

  const h = hands[0];
  const tri = sortVerticesClockwise([ptOf(h, 4), ptOf(h, 8), ptOf(h, 12)]);
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

    tracePolyOffset(ctx, pts, 0, 0);
    ctx.strokeStyle = poly.glow;
    ctx.lineWidth = 9;
    ctx.globalAlpha = 0.2 * fadeAlpha;
    ctx.stroke();

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
