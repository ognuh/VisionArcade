import { createGlowSprite32, type HandPose2D } from './types';

const GALAXY_COUNT = 1500;

// Radial color tiers: 0=Core White-Gold, 1=Gold, 2=Cyan Nebula, 3=Violet Nebula, 4=Outer Magenta, 5=Twinkle Star
const SPRITE_SPECS: ReadonlyArray<[string, string]> = [
  ['#ffffff', '#fef08a'], // 0: White-Gold Core
  ['#fffbeb', '#fbbf24'], // 1: Warm Accretion Gold
  ['#ffffff', '#22d3ee'], // 2: Electric Cyan Mid-Disk
  ['#f5d0fe', '#a855f7'], // 3: Cosmic Violet Nebula
  ['#fdf2f8', '#ec4899'], // 4: Outer Magenta Rim
  ['#ffffff', '#e0f2fe'], // 5: Bright Twinkling Field Star
];

/**
 * 1,500-Particle 3D-Tilted Logarithmic Spiral Galaxy (Zero-GC, 144Hz):
 * - 3 logarithmic spiral arms projected through 3D perspective tilt
 * - Radial color stratification: blazing white-gold core + fast accretion disk ->
 *   cyan/violet nebula mid-disk -> magenta outer rim + twinkling stars
 * - Fist compression -> Supernova burst -> 2-hand rotating galactic bridge
 */
export class GalaxyFx {
  private px = new Float32Array(GALAXY_COUNT);
  private py = new Float32Array(GALAXY_COUNT);
  private vx = new Float32Array(GALAXY_COUNT);
  private vy = new Float32Array(GALAXY_COUNT);
  private radiusNorm = new Float32Array(GALAXY_COUNT); // 0..1 radial tier
  private orbitR = new Float32Array(GALAXY_COUNT);
  private orbitA = new Float32Array(GALAXY_COUNT);
  private orbitSpd = new Float32Array(GALAXY_COUNT);
  private heightZ = new Float32Array(GALAXY_COUNT);
  private bridgeT = new Float32Array(GALAXY_COUNT);
  private baseSize = new Float32Array(GALAXY_COUNT);
  private twinklePhase = new Float32Array(GALAXY_COUNT);
  private twinkleFreq = new Float32Array(GALAXY_COUNT);
  private spriteIdx = new Uint8Array(GALAXY_COUNT);

  private sprites: HTMLCanvasElement[];
  private initialized = false;
  private prevWasFist = false;
  private elapsed = 0;

  constructor() {
    this.sprites = SPRITE_SPECS.map(([core, mid]) => createGlowSprite32(core, mid));

    for (let i = 0; i < GALAXY_COUNT; i += 1) {
      const arm = i % 3;
      const u = Math.pow(Math.random(), 0.78); // denser near core
      const r = 14 + u * 215;

      // True logarithmic spiral arm angle + organic dispersion
      const logAngle = 2.65 * Math.log(1 + u * 5.2);
      const scatter = (Math.random() - 0.5) * (0.28 + u * 0.65);
      const angle = (arm * Math.PI * 2) / 3 + logAngle + scatter;

      this.radiusNorm[i] = u;
      this.orbitR[i] = r;
      this.orbitA[i] = angle;
      // Keplerian-feel inner core spins much faster than outer rim
      this.orbitSpd[i] = 0.7 + Math.pow(1 - u, 1.6) * 3.6;
      this.heightZ[i] = (Math.random() - 0.5) * (1 - u * 0.55) * 26;
      this.bridgeT[i] = Math.random();
      this.twinklePhase[i] = Math.random() * Math.PI * 2;
      this.twinkleFreq[i] = 2.5 + Math.random() * 5.5;

      // Radial color stratification
      let sIdx = 2;
      if (u < 0.18) {
        sIdx = Math.random() < 0.65 ? 0 : 1; // White-gold core
      } else if (u < 0.58) {
        sIdx = Math.random() < 0.52 ? 2 : 3; // Cyan / Violet nebula
      } else {
        sIdx = Math.random() < 0.72 ? 4 : 5; // Magenta outer rim + bright twinkling stars
      }
      this.spriteIdx[i] = sIdx;
      this.baseSize[i] =
        sIdx === 5
          ? 8.5 + Math.random() * 8.5
          : 4.5 + Math.pow(Math.random(), 1.9) * (u < 0.25 ? 11.5 : 8.5);
    }
  }

  reset(): void {
    this.initialized = false;
    this.prevWasFist = false;
  }

  updateAndRender(
    ctx: CanvasRenderingContext2D,
    hands: HandPose2D[],
    dt: number,
    fadeProgress = 1,
  ): void {
    if (hands.length === 0 || fadeProgress <= 0.01) {
      this.initialized = false;
      this.prevWasFist = false;
      return;
    }

    this.elapsed += dt;
    const ease = 1 - Math.pow(1 - Math.min(1, fadeProgress), 3);
    const h0 = hands[0];
    const h1 = hands.length >= 2 ? hands[1] : null;
    const isFist = h0.gesture === 'FIST' && (!h1 || h1.gesture === 'FIST');

    // Subtle 3D galaxy plane tilt (inclination ~55 deg + dynamic roll from hand orientation)
    const tiltCos = 0.54;
    const tiltSin = 0.84;
    const rollAngle =
      Math.atan2(h0.pts[17 * 2 + 1] - h0.pts[5 * 2 + 1], h0.pts[17 * 2] - h0.pts[5 * 2]) * 0.35;
    const cosRoll = Math.cos(rollAngle);
    const sinRoll = Math.sin(rollAngle);

    if (!this.initialized) {
      for (let i = 0; i < GALAXY_COUNT; i += 1) {
        const a = this.orbitA[i];
        const r = this.orbitR[i] * 0.25;
        this.px[i] = h0.palmX + Math.cos(a) * r;
        this.py[i] = h0.palmY + Math.sin(a) * r * tiltCos;
        this.vx[i] = 0;
        this.vy[i] = 0;
      }
      this.initialized = true;
    }

    // Supernova burst when snapping from FIST -> OPEN
    if (this.prevWasFist && !isFist && h0.gesture === 'OPEN') {
      for (let i = 0; i < GALAXY_COUNT; i += 1) {
        const dx = this.px[i] - h0.palmX;
        const dy = this.py[i] - h0.palmY;
        const len = Math.hypot(dx, dy) || 1;
        const burst = 560 + (i % 9) * 140;
        this.vx[i] += (dx / len) * burst;
        this.vy[i] += (dy / len) * burst;
      }
    }
    this.prevWasFist = isFist;

    const scale = Math.max(0.68, Math.min(1.65, h0.palmScale / 92)) * (0.35 + 0.65 * ease);
    const damp = Math.pow(isFist ? 0.84 : 0.9, dt * 60);
    const spring = isFist ? 30.0 : 13.0;

    // Precompute bridge vectors if 2 hands
    let bDx = 0;
    let bDy = 0;
    let bNx = 0;
    let bNy = 1;
    let bUx = 1;
    let bUy = 0;
    if (h1 && !isFist) {
      bDx = h1.palmX - h0.palmX;
      bDy = h1.palmY - h0.palmY;
      const dist = Math.hypot(bDx, bDy) || 1;
      bUx = bDx / dist;
      bUy = bDy / dist;
      bNx = -bUy;
      bNy = bUx;
    }

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = ease;

    // Draw luminous galactic nebula halo & fast-spinning accretion disk around black hole
    const drawGalacticCore = (cx: number, cy: number, cs: number) => {
      const nebR = 140 * cs;
      const neb = ctx.createRadialGradient(cx, cy, 2, cx, cy, nebR);
      neb.addColorStop(0, isFist ? 'rgba(255,255,255,0.98)' : 'rgba(254,240,138,0.62)');
      neb.addColorStop(0.2, 'rgba(34,211,238,0.34)');
      neb.addColorStop(0.5, 'rgba(168,85,247,0.20)');
      neb.addColorStop(0.8, 'rgba(236,72,153,0.09)');
      neb.addColorStop(1, 'rgba(15,23,42,0)');
      ctx.fillStyle = neb;
      ctx.beginPath();
      ctx.arc(cx, cy, nebR, 0, Math.PI * 2);
      ctx.fill();

      if (!isFist) {
        // Fast-spinning relativistic accretion disk rings
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(rollAngle);

        ctx.strokeStyle = 'rgba(254, 249, 195, 0.92)';
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.ellipse(0, 0, 22 * cs, 11 * cs, 0, this.elapsed * 4.5, this.elapsed * 4.5 + Math.PI * 1.65);
        ctx.stroke();

        ctx.strokeStyle = 'rgba(34, 211, 238, 0.72)';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.ellipse(0, 0, 31 * cs, 15 * cs, 0, -this.elapsed * 3.1, -this.elapsed * 3.1 + Math.PI * 1.45);
        ctx.stroke();

        // Dark event horizon center
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = 'rgba(3, 4, 10, 0.95)';
        ctx.beginPath();
        ctx.arc(0, 0, 9 * cs, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        ctx.globalCompositeOperation = 'lighter';
      }
    };

    drawGalacticCore(h0.palmX, h0.palmY, scale);
    if (h1 && !isFist) {
      drawGalacticCore(h1.palmX, h1.palmY, Math.max(0.65, Math.min(1.5, h1.palmScale / 92)));
    }

    // Update & blit 1,500 sprites with 3D perspective depth scaling & star twinkle
    const persp = 560;
    for (let i = 0; i < GALAXY_COUNT; i += 1) {
      this.orbitA[i] += this.orbitSpd[i] * dt;
      const a = this.orbitA[i];
      let tx: number;
      let ty: number;
      let depthScale = 1.0;

      if (h1 && !isFist) {
        const u = this.bridgeT[i];
        const bx = h0.palmX + bDx * u;
        const by = h0.palmY + bDy * u;
        const arch = Math.sin(u * Math.PI);
        const swirlR = (14 + this.orbitR[i] * 0.34) * (0.35 + arch * 0.9);
        const c = Math.cos(a * 1.4 + u * 6.5);
        const s = Math.sin(a * 1.4 + u * 6.5);
        tx = bx + bNx * (c * swirlR) + bUx * (s * swirlR * 0.32);
        ty = by + bNy * (c * swirlR) + bUy * (s * swirlR * 0.32);
        depthScale = 0.78 + (s + 1) * 0.24;
      } else {
        const r = isFist ? this.orbitR[i] * 0.075 : this.orbitR[i] * scale;
        const lx = Math.cos(a) * r;
        const lyFlat = Math.sin(a) * r;
        // 3D inclined disk projection
        const ly3d = lyFlat * tiltCos + this.heightZ[i] * tiltSin * (isFist ? 0.1 : 1.0);
        const lz3d = lyFlat * tiltSin - this.heightZ[i] * tiltCos;
        depthScale = persp / Math.max(persp - lz3d, 220);

        const rx = lx * cosRoll - ly3d * sinRoll;
        const ry = lx * sinRoll + ly3d * cosRoll;
        tx = h0.palmX + rx * depthScale;
        ty = h0.palmY + ry * depthScale;
      }

      this.vx[i] = (this.vx[i] + (tx - this.px[i]) * spring * dt) * damp;
      this.vy[i] = (this.vy[i] + (ty - this.py[i]) * spring * dt) * damp;
      this.px[i] += this.vx[i] * dt;
      this.py[i] += this.vy[i] * dt;

      // Subtle twinkle on outer stars and bright field stars
      const sIdx = this.spriteIdx[i];
      let twinkle = 1.0;
      if (sIdx === 5 || this.radiusNorm[i] > 0.65) {
        twinkle = 0.62 + 0.52 * Math.sin(this.elapsed * this.twinkleFreq[i] + this.twinklePhase[i]);
      }

      const drawSize = this.baseSize[i] * depthScale * twinkle;
      const half = drawSize * 0.5;
      ctx.drawImage(this.sprites[sIdx], this.px[i] - half, this.py[i] - half, drawSize, drawSize);
    }

    ctx.restore();
  }
}
