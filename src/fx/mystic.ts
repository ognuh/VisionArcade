import { createGlowSprite32, type HandPose2D } from './types';

const EMBER_POOL_SIZE = 220;

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Doctor Strange Eldritch Mandala Shield (60 FPS Cinema Quality, Zero-GC, 0 shadowBlur):
 * - Staggered sequential draw-in animation (0 -> 0.72s): Core Seal -> Rune Ring -> Octagram Squares -> 72-tick Radar
 * - Pseudo-3D palm-normal projection matrix following hand tilt & roll
 * - 120 swirling depth-scaled fiery orange-gold embers (#FF8C00 -> #FFF3B0)
 */
export class MysticFx {
  private sx = new Float32Array(EMBER_POOL_SIZE);
  private sy = new Float32Array(EMBER_POOL_SIZE);
  private sz = new Float32Array(EMBER_POOL_SIZE); // pseudo-3D depth for size & glow scaling
  private svx = new Float32Array(EMBER_POOL_SIZE);
  private svy = new Float32Array(EMBER_POOL_SIZE);
  private sLife = new Float32Array(EMBER_POOL_SIZE);
  private sMax = new Float32Array(EMBER_POOL_SIZE);
  private sSize = new Float32Array(EMBER_POOL_SIZE);
  private cursor = 0;
  private elapsed = 0;
  private drawInTime = 0;
  private hadHands = false;
  // Ethereal unwind dissolve state when hands retract (550ms reverse spin + 180 inertial ember shower)
  private unwindTime = 1;
  private lastShields: {
    cx: number;
    cy: number;
    vx: number;
    vy: number;
    baseR: number;
    roll: number;
    scaleX: number;
    scaleY: number;
  }[] = [];
  private prevPalmPos: { x: number; y: number }[] = [];

  private goldSprite: HTMLCanvasElement;
  private orangeSprite: HTMLCanvasElement;

  constructor() {
    this.sLife.fill(99);
    this.sMax.fill(1);
    this.goldSprite = createGlowSprite32('#fff3b0', '#ff8c00');
    this.orangeSprite = createGlowSprite32('#fde047', '#ea580c');
  }

  reset(): void {
    this.drawInTime = 0;
    this.hadHands = false;
    this.unwindTime = 1;
    this.lastShields.length = 0;
  }

  private spawnEmber(x: number, y: number, vx: number, vy: number): void {
    const i = this.cursor;
    this.cursor = (this.cursor + 1) % EMBER_POOL_SIZE;
    this.sx[i] = x;
    this.sy[i] = y;
    this.sz[i] = 0.65 + Math.random() * 0.85;
    this.svx[i] = vx;
    this.svy[i] = vy;
    this.sLife[i] = 0;
    this.sMax[i] = 0.3 + Math.random() * 0.45;
    this.sSize[i] = 5 + Math.random() * 8.5;
  }

  private drawMultiPassArc(
    ctx: CanvasRenderingContext2D,
    r: number,
    sweepRad: number,
    outerW: number,
    coreW: number,
    outerCol: string,
    coreCol: string,
    alpha: number,
  ): void {
    if (sweepRad <= 0.02 || alpha <= 0.01) return;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, sweepRad);
    ctx.strokeStyle = outerCol;
    ctx.lineWidth = outerW;
    ctx.globalAlpha = 0.25 * alpha;
    ctx.stroke();

    ctx.strokeStyle = coreCol;
    ctx.lineWidth = coreW;
    ctx.globalAlpha = 0.94 * alpha;
    ctx.stroke();
  }

  /** Draws a square incrementally along its perimeter (progress 0..1) for staggered draw-in. */
  private drawProgressiveSquare(
    ctx: CanvasRenderingContext2D,
    half: number,
    progress: number,
    coreCol: string,
    alpha: number,
  ): void {
    if (progress <= 0.01) return;
    const corners: [number, number][] = [
      [-half, -half],
      [half, -half],
      [half, half],
      [-half, half],
      [-half, -half],
    ];
    const totalSegs = 4 * clamp01(progress);
    const fullSegs = Math.floor(totalSegs);
    const rem = totalSegs - fullSegs;

    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let s = 0; s < fullSegs; s += 1) {
      ctx.lineTo(corners[s + 1][0], corners[s + 1][1]);
    }
    if (fullSegs < 4 && rem > 0) {
      const a = corners[fullSegs];
      const b = corners[fullSegs + 1];
      ctx.lineTo(a[0] + (b[0] - a[0]) * rem, a[1] + (b[1] - a[1]) * rem);
    }

    ctx.strokeStyle = '#ff8c00';
    ctx.lineWidth = 6.5;
    ctx.globalAlpha = 0.24 * alpha;
    ctx.stroke();

    ctx.strokeStyle = coreCol;
    ctx.lineWidth = 1.75;
    ctx.globalAlpha = 0.94 * alpha;
    ctx.stroke();
  }

  updateAndRender(
    ctx: CanvasRenderingContext2D,
    hands: HandPose2D[],
    dt: number,
    fadeProgress = 1,
  ): void {
    if (hands.length === 0) {
      if (this.hadHands && this.lastShields.length > 0) {
        // Trigger 550ms Mystical Ember Unwind with ~180 luminous sparks inheriting hand velocity!
        this.unwindTime = 0;
        const perShield = Math.floor(180 / Math.max(this.lastShields.length, 1));
        for (let sIdx = 0; sIdx < this.lastShields.length; sIdx += 1) {
          const sh = this.lastShields[sIdx];
          for (let e = 0; e < perShield; e += 1) {
            const a = (e * Math.PI * 2) / perShield + Math.random() * 0.25;
            const r = sh.baseR * (0.28 + Math.random() * 0.78);
            const spd = 65 + Math.random() * 195;
            this.spawnEmber(
              sh.cx + Math.cos(a) * r * sh.scaleX,
              sh.cy + Math.sin(a) * r * sh.scaleY,
              Math.cos(a) * spd - Math.sin(a) * 55 + sh.vx * 0.35,
              Math.sin(a) * spd - 95 + sh.vy * 0.35,
            );
          }
        }
      }
      this.hadHands = false;
      this.drawInTime = 0;
      this.prevPalmPos.length = 0;
      if (this.unwindTime < 1) {
        this.unwindTime = Math.min(1, this.unwindTime + dt / 0.55);
      }
    } else {
      if (!this.hadHands) {
        this.hadHands = true;
        this.drawInTime = 0;
      }
      this.unwindTime = 1;
      this.drawInTime = Math.min(1.0, this.drawInTime + dt / 0.72);
    }

    const ease = 1 - Math.pow(1 - Math.min(1, fadeProgress), 3);
    const isUnwinding = hands.length === 0 && this.unwindTime < 1 && this.lastShields.length > 0;
    this.elapsed += dt * (isUnwinding ? 2.6 : 0.4 + 0.6 * clamp01(this.drawInTime * 1.4));
    if (ease <= 0.01) return;

    // Cache live shield transforms + hand velocity for inertial 550ms dissolve
    if (hands.length > 0) {
      this.lastShields.length = 0;
      for (let hIdx = 0; hIdx < hands.length; hIdx += 1) {
        const h = hands[hIdx];
        const prev = this.prevPalmPos[hIdx];
        const hvx = prev ? (h.palmX - prev.x) / Math.max(dt, 0.008) : 0;
        const hvy = prev ? (h.palmY - prev.y) / Math.max(dt, 0.008) : 0;
        this.prevPalmPos[hIdx] = { x: h.palmX, y: h.palmY };
        const ux = h.pts[17 * 2] - h.pts[5 * 2];
        const uy = h.pts[17 * 2 + 1] - h.pts[5 * 2 + 1];
        const vx = h.pts[9 * 2] - h.pts[0];
        const vy = h.pts[9 * 2 + 1] - h.pts[1];
        const uLen = Math.hypot(ux, uy) || 1;
        const vLen = Math.hypot(vx, vy) || 1;
        const crossMag = Math.abs(ux * vy - uy * vx) / (uLen * vLen);
        this.lastShields.push({
          cx: h.palmX,
          cy: h.palmY,
          vx: hvx,
          vy: hvy,
          baseR: Math.max(66, Math.min(205, h.palmScale * 1.55)),
          roll: Math.atan2(uy, ux),
          scaleX: Math.max(0.52, Math.min(1.0, uLen / Math.max(vLen * 0.72, 1))),
          scaleY: Math.max(0.55, Math.min(1.0, 0.44 + crossMag * 0.56)),
        });
      }
      this.prevPalmPos.length = hands.length;
    }

    const activeUnwindRem = isUnwinding ? 1 - this.unwindTime : 1;
    const renderAlpha = ease * (isUnwinding ? Math.pow(activeUnwindRem, 1.5) : 1);
    const expandMul = isUnwinding ? 1 + this.unwindTime * 0.2 : 1;
    const effTime = isUnwinding ? activeUnwindRem : this.drawInTime;

    // Staggered sequential layer draw-in (or reverse unwind when retracting hands!)
    const pCore = clamp01(effTime / 0.28);
    const pRune = clamp01((effTime - 0.14) / 0.34);
    const pSq1 = clamp01((effTime - 0.28) / 0.34);
    const pSq2 = clamp01((effTime - 0.38) / 0.34);
    const pOuter = clamp01((effTime - 0.45) / 0.55);

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const shieldsToDraw = hands.length > 0 || isUnwinding ? this.lastShields : [];

    for (let hIdx = 0; hIdx < shieldsToDraw.length; hIdx += 1) {
      const sh = shieldsToDraw[hIdx];
      const cx = sh.cx;
      const cy = sh.cy - (isUnwinding ? this.unwindTime * 22 : 0);
      const baseR = sh.baseR * expandMul;
      const roll = sh.roll;
      const scaleX = sh.scaleX;
      const scaleY = sh.scaleY;

      // Warm Eldritch ambient core glow
      const glow = ctx.createRadialGradient(cx, cy, baseR * 0.08, cx, cy, baseR * 1.35);
      glow.addColorStop(0, `rgba(255, 243, 176, ${0.44 * pCore * renderAlpha})`);
      glow.addColorStop(0.45, `rgba(255, 140, 0, ${0.26 * pRune * renderAlpha})`);
      glow.addColorStop(1, 'rgba(234, 88, 12, 0)');
      ctx.fillStyle = glow;
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, baseR * 1.35, 0, Math.PI * 2);
      ctx.fill();

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(roll);
      ctx.scale(scaleX, scaleY);

      // Step 1: Inner Core Seal draws first
      if (pCore > 0.01) {
        ctx.save();
        ctx.rotate(-this.elapsed * 2.9);
        const rCore = baseR * 0.22;
        this.drawMultiPassArc(ctx, rCore, Math.PI * 2 * pCore, 5.5, 1.6, '#ff8c00', '#ffffff', renderAlpha);
        if (pCore > 0.45) {
          ctx.strokeStyle = '#fff3b0';
          ctx.lineWidth = 1.25;
          ctx.globalAlpha = 0.92 * renderAlpha * clamp01((pCore - 0.45) / 0.55);
          ctx.beginPath();
          for (let k = 0; k < 3; k += 1) {
            const a = (k * Math.PI * 2) / 3;
            const px = Math.cos(a) * rCore;
            const py = Math.sin(a) * rCore;
            if (k === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
          ctx.closePath();
          ctx.stroke();
        }
        ctx.restore();
      }

      // Step 2: Geometric Rune Ring with 12 minimalist glyphs traces around
      if (pRune > 0.01) {
        ctx.save();
        ctx.rotate(this.elapsed * 1.8);
        const rRune = baseR * 0.48;
        this.drawMultiPassArc(ctx, rRune, Math.PI * 2 * pRune, 5.5, 1.35, '#ff8c00', '#fff3b0', renderAlpha);

        const visibleGlyphs = Math.floor(pRune * 12);
        if (visibleGlyphs > 0) {
          ctx.strokeStyle = '#fff3b0';
          ctx.lineWidth = 1.25;
          ctx.globalAlpha = 0.9 * renderAlpha;
          ctx.beginPath();
          for (let g = 0; g < visibleGlyphs; g += 1) {
            const a = (g * Math.PI * 2) / 12;
            const gx = Math.cos(a) * rRune;
            const gy = Math.sin(a) * rRune;
            const nx = -Math.sin(a);
            const ny = Math.cos(a);
            const rx = Math.cos(a);
            const ry = Math.sin(a);
            ctx.moveTo(gx - nx * 6, gy - ny * 6);
            ctx.lineTo(gx + rx * 6, gy + ry * 6);
            ctx.lineTo(gx + nx * 6, gy + ny * 6);
            if (g % 2 === 0) {
              ctx.moveTo(gx - rx * 4, gy - ry * 4);
              ctx.lineTo(gx + rx * 5, gy + ry * 5);
            }
          }
          ctx.stroke();
        }
        ctx.restore();
      }

      // Step 3: Two counter-rotating nested octagram squares draw progressively
      if (pSq1 > 0.01) {
        ctx.save();
        ctx.rotate(-this.elapsed * 1.25);
        this.drawProgressiveSquare(ctx, baseR * 0.63, pSq1, '#fff3b0', renderAlpha);
        ctx.restore();
      }
      if (pSq2 > 0.01) {
        ctx.save();
        ctx.rotate(-this.elapsed * 1.25 + Math.PI / 4);
        this.drawProgressiveSquare(ctx, baseR * 0.63, pSq2, '#ffb703', renderAlpha);
        ctx.restore();
      }

      // Step 4: Dense 72-tick outer radar degree ring completes the shield
      if (pOuter > 0.01) {
        ctx.save();
        ctx.rotate(this.elapsed * 0.85);
        this.drawMultiPassArc(ctx, baseR, Math.PI * 2 * pOuter, 7.5, 2.1, '#ff8c00', '#fff3b0', renderAlpha);
        this.drawMultiPassArc(
          ctx,
          baseR * 0.9,
          Math.PI * 2 * pOuter,
          4.2,
          1.1,
          '#ff8c00',
          '#fde047',
          renderAlpha,
        );

        const tickCount = Math.floor(pOuter * 72);
        if (tickCount > 0) {
          ctx.strokeStyle = '#fff3b0';
          ctx.lineWidth = 1.0;
          ctx.globalAlpha = 0.82 * renderAlpha;
          ctx.beginPath();
          for (let t = 0; t < tickCount; t += 1) {
            const a = (t * Math.PI * 2) / 72;
            const rIn = t % 6 === 0 ? baseR * 0.84 : t % 2 === 0 ? baseR * 0.9 : baseR * 0.94;
            ctx.moveTo(Math.cos(a) * rIn, Math.sin(a) * rIn);
            ctx.lineTo(Math.cos(a) * baseR, Math.sin(a) * baseR);
          }
          ctx.stroke();
        }
        ctx.restore();
      }

      ctx.restore();

      // Spawn swirling tangential embers while live
      if (!isUnwinding && pOuter > 0.2) {
        const cosR = Math.cos(roll);
        const sinR = Math.sin(roll);
        for (let s = 0; s < 3; s += 1) {
          const a = Math.random() * Math.PI * 2 * pOuter;
          const lx = Math.cos(a) * baseR * scaleX;
          const ly = Math.sin(a) * baseR * scaleY;
          const wx = cx + lx * cosR - ly * sinR;
          const wy = cy + lx * sinR + ly * cosR;

          const tvx = -Math.sin(a) * scaleX;
          const tvy = Math.cos(a) * scaleY;
          const spd = 125 + Math.random() * 235;
          this.spawnEmber(wx, wy, (tvx * cosR - tvy * sinR) * spd, (tvx * sinR + tvy * cosR) * spd);
        }
      }
    }

    // Update & blit 120 depth-scaled embers
    for (let i = 0; i < EMBER_POOL_SIZE; i += 1) {
      if (this.sLife[i] >= this.sMax[i]) continue;
      this.sLife[i] += dt;
      if (this.sLife[i] >= this.sMax[i]) continue;
      this.svy[i] += 220 * dt;
      this.sx[i] += this.svx[i] * this.sz[i] * dt;
      this.sy[i] += this.svy[i] * this.sz[i] * dt;

      const alpha = (1 - this.sLife[i] / this.sMax[i]) * ease;
      const sz = this.sSize[i] * this.sz[i] * (0.45 + alpha * 0.75);
      ctx.globalAlpha = alpha;
      ctx.drawImage(
        i % 2 === 0 ? this.goldSprite : this.orangeSprite,
        this.sx[i] - sz * 0.5,
        this.sy[i] - sz * 0.5,
        sz,
        sz,
      );
    }

    ctx.restore();
  }
}
