/**
 * One-Euro Filter per scalar axis (minCutoff: 1.2, beta: 0.04):
 * Eliminates micro-jitter when holding still while staying responsive with near-zero lag during fast hand motion.
 */
export class OneEuroAxis {
  private xPrev: number | null = null;
  private dxPrev = 0;

  private alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  public filter(x: number, dt: number, minCutoff = 1.2, beta = 0.04, dCutoff = 1.0): number {
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
