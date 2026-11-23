import { createGlowSprite32, type HandPose2D } from './types';

const EMBER_POOL_SIZE = 110;

/**
 * Doctor Strange Eldritch Mandala Shield (Zero-GC, 144Hz):
 * - Pseudo-3D palm-normal projection matrix derived from wrist (#0) & knuckles (#5, #9, #17)
 * - Counter-rotating multi-layered geometry: dense 72-tick degree ring, 2 nested octagram
 *   squares, 12 minimalist geometric rune glyphs, and blazing inner core seal
 * - Fiery orange-gold palette (#FF8C00 -> #FFF3B0) + 110 tangential ember sprites
 */
export class MysticFx {
  private sx = new Float32Array(EMBER_POOL_SIZE);
  private sy = new Float32Array(EMBER_POOL_SIZE);
  private svx = new Float32Array(EMBER_POOL_SIZE);
  private svy = new Float32Array(EMBER_POOL_SIZE);
  private sLife = new Float32Array(EMBER_POOL_SIZE);
  private sMax = new Float32Array(EMBER_POOL_SIZE);
  private sSize = new Float32Array(EMBER_POOL_SIZE);
  private cursor = 0;
  private elapsed = 0;

  private goldSprite: HTMLCanvasElement;
  private orangeSprite: HTMLCanvasElement;

  constructor() {
    this.sLife.fill(99);
    this.sMax.fill(1);
    this.goldSprite = createGlowSprite32('#fff3b0', '#ff8c00');
    this.orangeSprite = createGlowSprite32('#fde047', '#ea580c');
  }

  private spawnEmber(x: number, y: number, vx: number, vy: number): void {
    const i = this.cursor;
    this.cursor = (this.cursor + 1) % EMBER_POOL_SIZE;
    this.sx[i] = x;
    this.sy[i] = y;
    this.svx[i] = vx;
    this.svy[i] = vy;
    this.sLife[i] = 0;
    this.sMax[i] = 0.28 + Math.random() * 0.42;
    this.sSize[i] = 5 + Math.random() * 8;
  }

  private drawMultiPassRing(
    ctx: CanvasRenderingContext2D,
    r: number,
    outerW: number,
    coreW: number,
    outerCol: string,
    coreCol: string,
  ): void {
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.strokeStyle = outerCol;
    ctx.lineWidth = outerW;
    ctx.globalAlpha = 0.25;
    ctx.stroke();

    ctx.strokeStyle = coreCol;
    ctx.lineWidth = coreW;
    ctx.globalAlpha = 0.94;
    ctx.stroke();
  }

  updateAndRender(
    ctx: CanvasRenderingContext2D,
    hands: HandPose2D[],
    dt: number,
    fadeProgress = 1,
  ): void {
    const ease = 1 - Math.pow(1 - Math.min(1, fadeProgress), 3);
    // Smooth rotational spin-up when opening Mystic Shield
    this.elapsed += dt * (0.35 + 0.65 * ease);
    if (ease <= 0.01) return;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (let hIdx = 0; hIdx < hands.length; hIdx += 1) {
      const h = hands[hIdx];
      const cx = h.palmX;
      const cy = h.palmY;
      const baseR = Math.max(66, Math.min(205, h.palmScale * 1.55)) * (0.35 + 0.65 * ease);

      // Compute pseudo-3D palm orientation vectors from wrist(#0)->middle(#9) and index(#5)->pinky(#17)
      const ux = h.pts[17 * 2] - h.pts[5 * 2];
      const uy = h.pts[17 * 2 + 1] - h.pts[5 * 2 + 1];
      const vx = h.pts[9 * 2] - h.pts[0];
      const vy = h.pts[9 * 2 + 1] - h.pts[1];

      const uLen = Math.hypot(ux, uy) || 1;
      const vLen = Math.hypot(vx, vy) || 1;
      const roll = Math.atan2(uy, ux);
      // Perspective foreshortening along palm normal
      const crossMag = Math.abs(ux * vy - uy * vx) / (uLen * vLen);
      const scaleX = Math.max(0.55, Math.min(1.0, uLen / Math.max(vLen * 0.72, 1)));
      const scaleY = Math.max(0.58, Math.min(1.0, 0.45 + crossMag * 0.55));

      // Warm solar Eldritch ambient core glow
      const glow = ctx.createRadialGradient(cx, cy, baseR * 0.08, cx, cy, baseR * 1.3);
      glow.addColorStop(0, 'rgba(255, 243, 176, 0.42)');
      glow.addColorStop(0.45, 'rgba(255, 140, 0, 0.24)');
      glow.addColorStop(1, 'rgba(234, 88, 12, 0)');
      ctx.fillStyle = glow;
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, baseR * 1.3, 0, Math.PI * 2);
      ctx.fill();

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(roll);
      ctx.scale(scaleX, scaleY);

      // Layer 1: Dense 72-tick outer radar degree ring (clockwise)
      ctx.save();
      ctx.rotate(this.elapsed * 0.85);
      this.drawMultiPassRing(ctx, baseR, 7, 2.0, '#ff8c00', '#fff3b0');
      this.drawMultiPassRing(ctx, baseR * 0.9, 4, 1.1, '#ff8c00', '#fde047');

      ctx.strokeStyle = '#fff3b0';
      ctx.lineWidth = 1.0;
      ctx.globalAlpha = 0.82;
      ctx.beginPath();
      for (let t = 0; t < 72; t += 1) {
        const a = (t * Math.PI * 2) / 72;
        const rIn = t % 6 === 0 ? baseR * 0.84 : t % 2 === 0 ? baseR * 0.9 : baseR * 0.94;
        ctx.moveTo(Math.cos(a) * rIn, Math.sin(a) * rIn);
        ctx.lineTo(Math.cos(a) * baseR, Math.sin(a) * baseR);
      }
      ctx.stroke();
      ctx.restore();

      // Layer 2: Two counter-rotating nested squares forming an 8-point octagram star
      for (let sq = 0; sq < 2; sq += 1) {
        ctx.save();
        ctx.rotate(-this.elapsed * 1.25 + (sq * Math.PI) / 4);
        const half = baseR * 0.63;

        ctx.strokeStyle = '#ff8c00';
        ctx.lineWidth = 6;
        ctx.globalAlpha = 0.24;
        ctx.strokeRect(-half, -half, half * 2, half * 2);

        ctx.strokeStyle = sq === 0 ? '#fff3b0' : '#ffb703';
        ctx.lineWidth = 1.7;
        ctx.globalAlpha = 0.94;
        ctx.strokeRect(-half, -half, half * 2, half * 2);
        ctx.restore();
      }

      // Layer 3: Geometric rune ring with 12 minimalist arcane glyphs (clockwise)
      ctx.save();
      ctx.rotate(this.elapsed * 1.8);
      const rRune = baseR * 0.48;
      this.drawMultiPassRing(ctx, rRune, 5, 1.3, '#ff8c00', '#fff3b0');

      ctx.strokeStyle = '#fff3b0';
      ctx.lineWidth = 1.25;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      for (let g = 0; g < 12; g += 1) {
        const a = (g * Math.PI * 2) / 12;
        const gx = Math.cos(a) * rRune;
        const gy = Math.sin(a) * rRune;
        const nx = -Math.sin(a);
        const ny = Math.cos(a);
        const rx = Math.cos(a);
        const ry = Math.sin(a);
        // Minimalist chevron / cross glyph oriented tangentially
        ctx.moveTo(gx - nx * 6, gy - ny * 6);
        ctx.lineTo(gx + rx * 6, gy + ry * 6);
        ctx.lineTo(gx + nx * 6, gy + ny * 6);
        if (g % 2 === 0) {
          ctx.moveTo(gx - rx * 4, gy - ry * 4);
          ctx.lineTo(gx + rx * 5, gy + ry * 5);
        }
      }
      ctx.stroke();
      ctx.restore();

      // Layer 4: Fast counter-rotating inner core seal
      ctx.save();
      ctx.rotate(-this.elapsed * 2.9);
      const rCore = baseR * 0.22;
      this.drawMultiPassRing(ctx, rCore, 5, 1.5, '#ff8c00', '#ffffff');
      ctx.strokeStyle = '#fff3b0';
      ctx.lineWidth = 1.2;
      ctx.globalAlpha = 0.92;
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
      ctx.restore();

      ctx.restore();

      // Spawn 3 tangential sparks per frame off the spinning outer rim (projected with tilt)
      const cosR = Math.cos(roll);
      const sinR = Math.sin(roll);
      for (let s = 0; s < 3; s += 1) {
        const a = Math.random() * Math.PI * 2;
        const lx = Math.cos(a) * baseR * scaleX;
        const ly = Math.sin(a) * baseR * scaleY;
        const wx = cx + lx * cosR - ly * sinR;
        const wy = cy + lx * sinR + ly * cosR;

        const tvx = -Math.sin(a) * scaleX;
        const tvy = Math.cos(a) * scaleY;
        const spd = 120 + Math.random() * 220;
        const vxWorld = (tvx * cosR - tvy * sinR) * spd;
        const vyWorld = (tvx * sinR + tvy * cosR) * spd;
        this.spawnEmber(wx, wy, vxWorld, vyWorld);
      }
    }

    // Update & blit 110 glowing ember sprites
    for (let i = 0; i < EMBER_POOL_SIZE; i += 1) {
      if (this.sLife[i] >= this.sMax[i]) continue;
      this.sLife[i] += dt;
      if (this.sLife[i] >= this.sMax[i]) continue;
      this.svy[i] += 230 * dt;
      this.sx[i] += this.svx[i] * dt;
      this.sy[i] += this.svy[i] * dt;

      const alpha = 1 - this.sLife[i] / this.sMax[i];
      const sz = this.sSize[i] * (0.45 + alpha * 0.75);
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
