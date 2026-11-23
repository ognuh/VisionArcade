import type { Landmark3D } from '../types/vision';

/**
 * One-Euro Filter per scalar axis: eliminates micro-jitter when holding still
 * while staying responsive with near-zero lag during fast hand motion.
 */
class OneEuroAxis {
  private xPrev: number | null = null;
  private dxPrev = 0;

  private alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  public filter(x: number, dt: number, minCutoff: number, beta: number, dCutoff = 1.0): number {
    if (this.xPrev === null) {
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }

    const dx = (x - this.xPrev) / dt;
    const aD = this.alpha(dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;

    const cutoff = minCutoff + beta * Math.abs(dxHat);
    const aX = this.alpha(cutoff, dt);
    const xHat = aX * x + (1 - aX) * this.xPrev;

    this.xPrev = xHat;
    this.dxPrev = dxHat;
    return xHat;
  }

  public reset(): void {
    this.xPrev = null;
    this.dxPrev = 0;
  }
}

interface LandmarkFilter3D {
  x: OneEuroAxis;
  y: OneEuroAxis;
  z: OneEuroAxis;
}

export class HandLandmarkSmoother {
  private hands: LandmarkFilter3D[][] = [];
  private readonly minCutoff: number;
  private readonly beta: number;

  constructor(minCutoff = 1.35, beta = 18.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
  }

  private createHandFilters(count: number): LandmarkFilter3D[] {
    const arr: LandmarkFilter3D[] = new Array(count);
    for (let i = 0; i < count; i++) {
      arr[i] = { x: new OneEuroAxis(), y: new OneEuroAxis(), z: new OneEuroAxis() };
    }
    return arr;
  }

  public update(rawHands: Landmark3D[][], dtSeconds: number): Landmark3D[][] {
    const dt = Math.min(Math.max(dtSeconds, 0.004), 0.08);

    if (rawHands.length === 0) {
      this.hands = [];
      return [];
    }

    const result: Landmark3D[][] = new Array(rawHands.length);

    for (let h = 0; h < rawHands.length; h++) {
      const raw = rawHands[h];
      if (!this.hands[h] || this.hands[h].length !== raw.length) {
        this.hands[h] = this.createHandFilters(raw.length);
      }

      const filters = this.hands[h];
      const out: Landmark3D[] = new Array(raw.length);

      for (let i = 0; i < raw.length; i++) {
        // Slightly steadier cutoff on thumb/index fingertips (4, 8) for precision line drawing.
        const isDrawTip = i === 4 || i === 8;
        const mc = isDrawTip ? this.minCutoff * 0.85 : this.minCutoff;
        const bt = isDrawTip ? this.beta * 1.1 : this.beta;

        out[i] = {
          x: filters[i].x.filter(raw[i].x, dt, mc, bt),
          y: filters[i].y.filter(raw[i].y, dt, mc, bt),
          z: filters[i].z.filter(raw[i].z, dt, mc, bt * 0.6),
        };
      }
      result[h] = out;
    }

    this.hands.length = rawHands.length;
    return result;
  }

  public reset(): void {
    this.hands = [];
  }
}
