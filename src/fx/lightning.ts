import { createGlowSprite32, type HandPose2D } from './types';

const MAIN_SEGS = 24;
const MAIN_PTS = MAIN_SEGS + 1;
const BRANCH_SEGS = 8;
const BRANCH_PTS = BRANCH_SEGS + 1;
const SUB_BRANCH_SEGS = 5;
const SUB_BRANCH_PTS = SUB_BRANCH_SEGS + 1;
const SPARK_POOL_SIZE = 56;
const FINGERTIP_IDS = [4, 8, 12, 16, 20];

/** Smooth continuous 2D trigonometric turbulence (Zero-GC, silky temporal writhing at 144Hz). */
function smoothPlasmaNoise(u: number, t: number, seed: number): number {
  const s1 = Math.sin(u * 13.0 + t * 14.5 + seed * 3.7);
  const s2 = Math.sin(u * 29.0 - t * 22.0 + seed * 7.1) * 0.52;
  const s3 = Math.cos(u * 53.0 + t * 31.0 - seed * 11.3) * 0.28;
  return (s1 + s2 + s3) / 1.8;
}

/**
 * Cinema-Grade Plasma Lightning FX (Zero-GC, 144Hz):
 * - Smooth temporal 2D noise (writhing high-voltage plasma, zero ugly random flicker)
 * - 2 Hands: 3–4 main plasma arcs + 10–14 recursive fractal branches + 5 fingertip-to-fingertip arcs when close
 * - 1 Hand (Tesla Coil): high-voltage arcs leaping palm -> 5 fingertips -> adjacent fingertips + corona discharges
 * - 3-Layer Additive Strokes (#6d28d9 wide aura -> #38bdf8 electric cyan -> #ffffff 1.5px core)
 * - Rotating palm plasma orbs + 56 gravity sparks via offscreen glow sprites
 */
export class LightningFx {
  private mainX = new Float32Array(MAIN_PTS);
  private mainY = new Float32Array(MAIN_PTS);
  private brX = new Float32Array(BRANCH_PTS);
  private brY = new Float32Array(BRANCH_PTS);
  private subX = new Float32Array(SUB_BRANCH_PTS);
  private subY = new Float32Array(SUB_BRANCH_PTS);

  // 56 pre-allocated plasma sparks with gravity
  private sx = new Float32Array(SPARK_POOL_SIZE);
  private sy = new Float32Array(SPARK_POOL_SIZE);
  private svx = new Float32Array(SPARK_POOL_SIZE);
  private svy = new Float32Array(SPARK_POOL_SIZE);
  private sLife = new Float32Array(SPARK_POOL_SIZE);
  private sMax = new Float32Array(SPARK_POOL_SIZE);
  private sType = new Uint8Array(SPARK_POOL_SIZE);
  private sparkCursor = 0;
  private elapsed = 0;

  private cyanSprite: HTMLCanvasElement;
  private violetSprite: HTMLCanvasElement;
  private whiteSprite: HTMLCanvasElement;

  constructor() {
    this.sLife.fill(99);
    this.sMax.fill(1);
    this.cyanSprite = createGlowSprite32('#ffffff', '#38bdf8');
    this.violetSprite = createGlowSprite32('#f5d0fe', '#7c3aed');
    this.whiteSprite = createGlowSprite32('#ffffff', '#bae6fd');
  }

  private spawnSpark(x: number, y: number, intensity: number): void {
    const i = this.sparkCursor;
    this.sparkCursor = (this.sparkCursor + 1) % SPARK_POOL_SIZE;
    const a = Math.random() * Math.PI * 2;
    const spd = (120 + Math.random() * 340) * Math.min(1.4, intensity);
    this.sx[i] = x + (Math.random() - 0.5) * 10;
    this.sy[i] = y + (Math.random() - 0.5) * 10;
    this.svx[i] = Math.cos(a) * spd;
    this.svy[i] = Math.sin(a) * spd - 45;
    this.sLife[i] = 0;
    this.sMax[i] = 0.18 + Math.random() * 0.26;
    this.sType[i] = i % 3;
  }

  private buildSmoothArc(
    xs: Float32Array,
    ys: Float32Array,
    pts: number,
    ax: number,
    ay: number,
    bx: number,
    by: number,
    amp: number,
    seed: number,
    taperEnd: boolean,
  ): void {
    const dx = bx - ax;
    const dy = by - ay;
    const dist = Math.hypot(dx, dy) || 1;
    const nx = -dy / dist;
    const ny = dx / dist;
    const segs = pts - 1;

    for (let i = 0; i < pts; i += 1) {
      const u = i / segs;
      const env = taperEnd ? Math.sin(u * Math.PI * 0.85) : Math.sin(u * Math.PI);
      const disp = smoothPlasmaNoise(u, this.elapsed, seed) * amp * env;
      xs[i] = ax + dx * u + nx * disp;
      ys[i] = ay + dy * u + ny * disp;
    }
  }

  /** 3-layer additive stroke: purple aura (#6d28d9, 0.18) -> electric cyan (#38bdf8, 0.65) -> pure white core (1.5px). */
  private strokePlasma3Layer(
    ctx: CanvasRenderingContext2D,
    xs: Float32Array,
    ys: Float32Array,
    pts: number,
    coreW: number,
    alphaScale: number,
  ): void {
    ctx.beginPath();
    ctx.moveTo(xs[0], ys[0]);
    for (let i = 1; i < pts; i += 1) {
      ctx.lineTo(xs[i], ys[i]);
    }

    // Layer 1: Wide electric purple outer aura
    ctx.strokeStyle = '#6d28d9';
    ctx.lineWidth = Math.max(4.5, coreW * 6.0);
    ctx.globalAlpha = 0.18 * alphaScale;
    ctx.stroke();

    // Layer 2: Vivid electric cyan plasma body
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = Math.max(2.0, coreW * 2.4);
    ctx.globalAlpha = 0.65 * alphaScale;
    ctx.stroke();

    // Layer 3: Pure white incandescent filament core
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = coreW;
    ctx.globalAlpha = 0.98 * alphaScale;
    ctx.stroke();
  }

  private drawMainArcWithFractalBranches(
    ctx: CanvasRenderingContext2D,
    ax: number,
    ay: number,
    bx: number,
    by: number,
    amp: number,
    coreW: number,
    seed: number,
    branchCount: number,
    fadeAlpha: number,
  ): void {
    this.buildSmoothArc(this.mainX, this.mainY, MAIN_PTS, ax, ay, bx, by, amp, seed, false);
    this.strokePlasma3Layer(ctx, this.mainX, this.mainY, MAIN_PTS, coreW, fadeAlpha);

    const baseAngle = Math.atan2(by - ay, bx - ax);
    const span = Math.hypot(bx - ax, by - ay);

    for (let b = 0; b < branchCount; b += 1) {
      // Smoothly drifting attachment index along the main bolt
      const frac = 0.16 + ((b + 0.5) / branchCount) * 0.68;
      const attachIdx = Math.max(2, Math.min(MAIN_PTS - 3, Math.round(frac * MAIN_SEGS)));
      const sx = this.mainX[attachIdx];
      const sy = this.mainY[attachIdx];

      const flicker = 0.55 + 0.45 * Math.sin(this.elapsed * 18.0 + seed * 4.1 + b * 2.7);
      if (flicker < 0.25) continue;

      const side = (b + Math.floor(seed)) % 2 === 0 ? 1 : -1;
      const angleOffset = side * (0.42 + 0.42 * Math.sin(this.elapsed * 6.0 + b * 3.1 + seed));
      const bAngle = baseAngle + angleOffset;
      const bLen = Math.max(30, span * (0.16 + 0.14 * flicker));
      const ex = sx + Math.cos(bAngle) * bLen;
      const ey = sy + Math.sin(bAngle) * bLen;

      this.buildSmoothArc(
        this.brX,
        this.brY,
        BRANCH_PTS,
        sx,
        sy,
        ex,
        ey,
        amp * 0.34,
        seed + b * 1.9,
        true,
      );
      this.strokePlasma3Layer(
        ctx,
        this.brX,
        this.brY,
        BRANCH_PTS,
        Math.max(0.75, coreW * 0.52),
        fadeAlpha * flicker * 0.85,
      );

      // Recursive secondary sub-fork off mid-branch
      if (b % 2 === 0) {
        const subStart = BRANCH_PTS >> 1;
        const ssx = this.brX[subStart];
        const ssy = this.brY[subStart];
        const subAngle = bAngle + side * 0.55;
        const subLen = bLen * 0.48;
        this.buildSmoothArc(
          this.subX,
          this.subY,
          SUB_BRANCH_PTS,
          ssx,
          ssy,
          ssx + Math.cos(subAngle) * subLen,
          ssy + Math.sin(subAngle) * subLen,
          amp * 0.18,
          seed + b * 5.3,
          true,
        );
        this.strokePlasma3Layer(
          ctx,
          this.subX,
          this.subY,
          SUB_BRANCH_PTS,
          0.65,
          fadeAlpha * flicker * 0.62,
        );
      }
    }
  }

  /** Draws a rotating electric plasma sphere inside the palm center. */
  private drawRotatingPalmOrb(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    fadeAlpha: number,
  ): void {
    const outerDia = radius * 3.6;
    ctx.globalAlpha = 0.78 * fadeAlpha;
    ctx.drawImage(this.violetSprite, cx - outerDia * 0.5, cy - outerDia * 0.5, outerDia, outerDia);

    const coreDia = radius * 1.9;
    ctx.globalAlpha = 0.96 * fadeAlpha;
    ctx.drawImage(this.whiteSprite, cx - coreDia * 0.5, cy - coreDia * 0.5, coreDia, coreDia);

    // Two fast counter-rotating plasma containment rings hugging the orb
    const spinA = this.elapsed * 5.4;
    const spinB = -this.elapsed * 4.2;

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.0;
    ctx.globalAlpha = 0.85 * fadeAlpha;
    ctx.beginPath();
    ctx.ellipse(cx, cy, radius * 1.15, radius * 0.62, spinA, 0, Math.PI * 1.45);
    ctx.stroke();

    ctx.strokeStyle = '#c084fc';
    ctx.lineWidth = 1.6;
    ctx.globalAlpha = 0.8 * fadeAlpha;
    ctx.beginPath();
    ctx.ellipse(cx, cy, radius * 0.95, radius * 0.55, spinB, Math.PI * 0.4, Math.PI * 1.9);
    ctx.stroke();
  }

  updateAndRender(
    ctx: CanvasRenderingContext2D,
    hands: HandPose2D[],
    dt: number,
    fadeAlpha = 1,
  ): void {
    this.elapsed += dt;
    if (fadeAlpha <= 0.01) return;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (hands.length >= 2) {
      const L = hands[0];
      const R = hands[1];
      const palmDist = Math.hypot(R.palmX - L.palmX, R.palmY - L.palmY);
      const proximity = Math.max(0.55, Math.min(1.65, 430 / Math.max(palmDist, 85)));
      const amp = Math.min(72, 24 + palmDist * 0.11);

      this.drawRotatingPalmOrb(ctx, L.palmX, L.palmY, 22 * proximity, fadeAlpha);
      this.drawRotatingPalmOrb(ctx, R.palmX, R.palmY, 22 * proximity, fadeAlpha);

      // Erupt contact sparks at both palms
      this.spawnSpark(L.palmX, L.palmY, proximity);
      this.spawnSpark(R.palmX, R.palmY, proximity);
      if (proximity > 0.95) {
        this.spawnSpark(L.palmX, L.palmY, proximity);
        this.spawnSpark(R.palmX, R.palmY, proximity);
      }

      // 3–4 main heavy plasma arcs + 10–12 fractal branches
      const mainCount = proximity > 0.9 ? 4 : 3;
      for (let m = 0; m < mainCount; m += 1) {
        const w = (m === 0 ? 1.85 : 1.35) * Math.min(1.35, proximity);
        this.drawMainArcWithFractalBranches(
          ctx,
          L.palmX,
          L.palmY,
          R.palmX,
          R.palmY,
          amp * (0.75 + m * 0.24),
          w,
          m * 3.17 + 0.5,
          m < 2 ? 4 : 2,
          fadeAlpha,
        );
      }

      // When hands come close (< 420px), ignite 5 symmetric fingertip-to-fingertip plasma arcs!
      if (palmDist < 420) {
        const fingerBlend = Math.min(1, (420 - palmDist) / 140) * fadeAlpha;
        for (let f = 0; f < FINGERTIP_IDS.length; f += 1) {
          const idx = FINGERTIP_IDS[f];
          this.buildSmoothArc(
            this.mainX,
            this.mainY,
            MAIN_PTS,
            L.pts[idx * 2],
            L.pts[idx * 2 + 1],
            R.pts[idx * 2],
            R.pts[idx * 2 + 1],
            amp * 0.45,
            12.3 + f * 2.1,
            false,
          );
          this.strokePlasma3Layer(ctx, this.mainX, this.mainY, MAIN_PTS, 1.1, fingerBlend * 0.85);
        }
      }
    } else if (hands.length === 1) {
      // Tesla Coil Hand Mode: palm plasma orb -> 5 fingertips -> fingertip chain -> open-air corona arcs
      const h = hands[0];
      this.drawRotatingPalmOrb(ctx, h.palmX, h.palmY, 20, fadeAlpha);
      this.spawnSpark(h.palmX, h.palmY, 0.9);

      for (let t = 0; t < FINGERTIP_IDS.length; t += 1) {
        const idx = FINGERTIP_IDS[t];
        const tx = h.pts[idx * 2];
        const ty = h.pts[idx * 2 + 1];

        // Palm -> Fingertip feed arc
        this.buildSmoothArc(
          this.mainX,
          this.mainY,
          MAIN_PTS,
          h.palmX,
          h.palmY,
          tx,
          ty,
          16,
          t * 2.4 + 1.1,
          false,
        );
        this.strokePlasma3Layer(ctx, this.mainX, this.mainY, MAIN_PTS, 1.2, fadeAlpha * 0.9);

        // Inter-fingertip jumping Tesla arcs (4->8->12->16->20)
        if (t < FINGERTIP_IDS.length - 1) {
          const nextIdx = FINGERTIP_IDS[t + 1];
          const nx = h.pts[nextIdx * 2];
          const ny = h.pts[nextIdx * 2 + 1];
          this.buildSmoothArc(
            this.brX,
            this.brY,
            BRANCH_PTS,
            tx,
            ty,
            nx,
            ny,
            14,
            t * 4.3 + 5.0,
            false,
          );
          this.strokePlasma3Layer(ctx, this.brX, this.brY, BRANCH_PTS, 1.05, fadeAlpha * 0.88);
        }

        // Open-air high-voltage corona discharge shooting outward from fingertip
        const dx = tx - h.palmX;
        const dy = ty - h.palmY;
        const len = Math.hypot(dx, dy) || 1;
        const reach = 48 + 38 * (0.5 + 0.5 * Math.sin(this.elapsed * 11 + t * 2.2));
        const ex = tx + (dx / len) * reach;
        const ey = ty + (dy / len) * reach;
        this.drawMainArcWithFractalBranches(
          ctx,
          tx,
          ty,
          ex,
          ey,
          20,
          1.15,
          t * 3.7 + 9.2,
          2,
          fadeAlpha * 0.85,
        );
      }
    }

    // Update & blit 56 gravity sparks
    for (let i = 0; i < SPARK_POOL_SIZE; i += 1) {
      if (this.sLife[i] >= this.sMax[i]) continue;
      this.sLife[i] += dt;
      if (this.sLife[i] >= this.sMax[i]) continue;
      this.svy[i] += 380 * dt;
      this.sx[i] += this.svx[i] * dt;
      this.sy[i] += this.svy[i] * dt;

      const alpha = (1 - this.sLife[i] / this.sMax[i]) * fadeAlpha;
      const sz = 6 + alpha * 10;
      const sprite =
        this.sType[i] === 0
          ? this.whiteSprite
          : this.sType[i] === 1
            ? this.cyanSprite
            : this.violetSprite;
      ctx.globalAlpha = alpha;
      ctx.drawImage(sprite, this.sx[i] - sz * 0.5, this.sy[i] - sz * 0.5, sz, sz);
    }

    ctx.restore();
  }
}
