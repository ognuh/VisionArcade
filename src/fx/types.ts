import type { GestureType } from '../types/vision';

export interface Point2D {
  x: number;
  y: number;
}

export interface HandPose2D {
  /** All 21 smoothed screen-space landmarks (x, y) in flat Float32Array[42]. */
  pts: Float32Array;
  palmX: number;
  palmY: number;
  palmScale: number;
  palmSpeedMps: number;
  gesture: GestureType;
}

export interface RenderPolygon {
  pts: Point2D[];
  shaderType: number; // 0 = GLASS, 1 = THERMAL, 2 = PRISM_BAND
  tint: [number, number, number, number];
  additive: boolean;
  glow: string;
}

/** Pre-renders a 32x32 soft radial glow sprite once at initialization (Zero-GC at 144Hz). */
export function createGlowSprite32(coreHex: string, midHex: string): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = 32;
  c.height = 32;
  const g = c.getContext('2d');
  if (g) {
    const grad = g.createRadialGradient(16, 16, 0, 16, 16, 16);
    grad.addColorStop(0, coreHex);
    grad.addColorStop(0.28, midHex);
    grad.addColorStop(0.66, `${midHex}44`);
    grad.addColorStop(1, `${midHex}00`);
    g.fillStyle = grad;
    g.fillRect(0, 0, 32, 32);
  }
  return c;
}
