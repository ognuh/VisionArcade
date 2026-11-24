import * as THREE from 'three';
import type { GraphicsQuality } from '../types/vision';
import type { HandPose2D } from './types';

const ULTRA_PARTICLES = 15000;
const PRO_PARTICLES = 9000;
const PINCH_NORM_THRESHOLD = 0.056;
const CHARGE_HOLD_MIN_SEC = 0.10;
const SUPERNOVA_COOLDOWN_SEC = 0.58;
const MAX_PARTICLE_SPEED = 13.5;

const FLUID_VERT = `
attribute float aSpeed;
attribute float aSize;
attribute float aSeed;

uniform float uOpacity;
uniform vec2 uHalfBounds;
uniform float uPixelRatio;

varying float vSpeed;
varying float vSeed;
varying float vOpacity;
varying float vEdgeFade;

void main() {
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPos;
  vSpeed = aSpeed;
  vSeed = aSeed;
  vOpacity = uOpacity;

  // Smooth margin attenuation so borders (especially bottom edge) never accumulate glare
  vec2 normPos = abs(position.xy) / max(uHalfBounds, vec2(0.001));
  float fx = 1.0 - smoothstep(0.84, 1.02, normPos.x);
  float fy = 1.0 - smoothstep(0.78, 1.00, normPos.y);
  vEdgeFade = fx * fy;

  // Pure streak-free circular droplets: size varies subtly by velocity in 2.5px - 6.5px logical CSS px
  float s = clamp(aSpeed, 0.0, 1.0);
  float logicalPx = clamp(mix(2.5, 4.6, aSize) + smoothstep(0.10, 1.0, s) * 1.9, 2.5, 6.5);
  gl_PointSize = logicalPx * clamp(uPixelRatio, 1.0, 1.5);
}
`;

const FLUID_FRAG = `
varying float vSpeed;
varying float vSeed;
varying float vOpacity;
varying float vEdgeFade;

void main() {
  float effOpacity = vOpacity * vEdgeFade;
  if (effOpacity <= 0.005) discard;

  // Soft radial glow using gl_PointCoord (zero velocity stretch / zero diagonal lines)
  vec2 p = gl_PointCoord - 0.5;
  float distSq = dot(p, p);
  if (distSq > 0.25) discard;

  float halo = exp(-distSq * 11.5);
  float core = exp(-distSq * 46.0);
  float s = clamp(vSpeed, 0.0, 1.0);

  // Luxury HDR velocity gradient:
  // Deep Sapphire -> Electric Cyan -> Neon Violet -> Subtle Warm White-Gold ONLY at fastest vortex/burst core
  vec3 cSapphire  = vec3(0.04, 0.20, 0.78);
  vec3 cCobalt    = vec3(0.16, 0.14, 0.84);
  vec3 cCyan      = vec3(0.05, 0.86, 1.04);
  vec3 cNeonViol  = vec3(0.68, 0.20, 1.06);
  vec3 cWhiteGold = vec3(1.16, 1.05, 0.80);

  vec3 slowCol = mix(cSapphire, cCobalt, smoothstep(0.15, 0.85, vSeed));
  vec3 midCol  = mix(cCyan, cNeonViol, smoothstep(0.32, 0.88, fract(vSeed * 3.17)));
  vec3 col = mix(slowCol, cCyan, smoothstep(0.06, 0.40, s));
  col = mix(col, midCol, smoothstep(0.40, 0.78, s));

  // Subtle warm white-gold highlight strictly on peak-velocity inner core
  float peakGold = smoothstep(0.76, 1.0, s) * core;
  col = mix(col, cWhiteGold, peakGold * 0.68);
  col += vec3(0.10, 0.32, 0.44) * core * (0.24 + s * 0.32);

  // Strict per-particle alpha cap (0.28 - 0.45) + soft HDR ceiling keeps dense silk translucent, never flat white
  float shape = halo * 0.54 + core * 0.46;
  float maxAlpha = mix(0.28, 0.45, s);
  float alpha = clamp(shape * maxAlpha * effOpacity, 0.0, 0.45);
  vec3 outRgb = col * (0.64 + s * 0.34);
  outRgb = min(outRgb, vec3(1.22));
  gl_FragColor = vec4(outRgb, alpha);
}
`;

interface ScreenToWorldFn {
  (px: number, py: number): { x: number; y: number; pxToWorld: number };
}

interface FluidHandState {
  hasPrev: boolean;
  prevWx: number;
  prevWy: number;
  smoothVx: number;
  smoothVy: number;
  chargeHoldSec: number; // accumulates during Pinch OR Fist for Supernova Burst on open
  burstCooldownSec: number;
}

interface SupernovaWave {
  active: boolean;
  x: number;
  y: number;
  radius: number;
  age: number;
  duration: number;
}

function countExtendedFingers2D(h: HandPose2D): number {
  const wx = h.pts[0];
  const wy = h.pts[1];
  let count = 0;
  const pairs: [number, number][] = [
    [8, 6],
    [12, 10],
    [16, 14],
    [20, 18],
  ];
  for (let i = 0; i < pairs.length; i += 1) {
    const [tip, pip] = pairs[i];
    const dTip = Math.hypot(h.pts[tip * 2] - wx, h.pts[tip * 2 + 1] - wy);
    const dPip = Math.hypot(h.pts[pip * 2] - wx, h.pts[pip * 2 + 1] - wy);
    if (dTip > dPip * 1.12) count += 1;
  }
  const dThumb = Math.hypot(h.pts[8] - h.pts[34], h.pts[9] - h.pts[35]);
  const dBase = Math.hypot(h.pts[4] - h.pts[34], h.pts[5] - h.pts[35]);
  if (dThumb > dBase * 1.12) count += 1;
  return count;
}

/**
 * Cinema-Grade 15,000-Particle Zero-GC HDR Energy Silk Sandbox (Three.js Points + Custom GLSL):
 * - Zero Velocity Motion-Stretch: pure soft radial glow orbs (2.5px - 6.5px) eliminate all diagonal streaks
 * - Strictly divergence-free 2D stream-function Curl Noise + soft boundary repulsion prevents bottom-edge piling
 * - Controlled per-particle alpha (0.28 - 0.45) + HDR ceiling keeps dense ribbons deep Sapphire/Cyan/Violet
 * - Wave silk-ribbon wake, Pinch glowing galaxy vortex, and Fist/Pinch -> Open circular Supernova Burst ring
 */
export class FluidParticleSandbox {
  public readonly root = new THREE.Group();

  private activeCount = ULTRA_PARTICLES;
  private px = new Float32Array(ULTRA_PARTICLES);
  private py = new Float32Array(ULTRA_PARTICLES);
  private vx = new Float32Array(ULTRA_PARTICLES);
  private vy = new Float32Array(ULTRA_PARTICLES);
  private seeds = new Float32Array(ULTRA_PARTICLES);

  private posBuf = new Float32Array(ULTRA_PARTICLES * 3);
  private speedBuf = new Float32Array(ULTRA_PARTICLES);
  private sizeBuf = new Float32Array(ULTRA_PARTICLES);

  private posAttr: THREE.BufferAttribute;
  private speedAttr: THREE.BufferAttribute;
  private geo: THREE.BufferGeometry;
  private mat: THREE.ShaderMaterial;
  private points: THREE.Points;

  private elapsed = 0;
  private initializedBounds = false;
  private respawnSeed = 918273;

  private handStates: [FluidHandState, FluidHandState] = [
    { hasPrev: false, prevWx: 0, prevWy: 0, smoothVx: 0, smoothVy: 0, chargeHoldSec: 0, burstCooldownSec: 0 },
    { hasPrev: false, prevWx: 0, prevWy: 0, smoothVx: 0, smoothVy: 0, chargeHoldSec: 0, burstCooldownSec: 0 },
  ];

  private waves: [SupernovaWave, SupernovaWave] = [
    { active: false, x: 0, y: 0, radius: 0, age: 0, duration: 0.58 },
    { active: false, x: 0, y: 0, radius: 0, age: 0, duration: 0.58 },
  ];

  constructor() {
    this.root.visible = false;

    this.geo = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(this.posBuf, 3);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.speedAttr = new THREE.BufferAttribute(this.speedBuf, 1);
    this.speedAttr.setUsage(THREE.DynamicDrawUsage);

    this.seedParticles(6.4, 3.6);

    this.geo.setAttribute('position', this.posAttr);
    this.geo.setAttribute('aSpeed', this.speedAttr);
    this.geo.setAttribute('aSize', new THREE.BufferAttribute(this.sizeBuf, 1));
    this.geo.setAttribute('aSeed', new THREE.BufferAttribute(this.seeds, 1));
    this.geo.setDrawRange(0, this.activeCount);

    this.mat = new THREE.ShaderMaterial({
      uniforms: {
        uOpacity: { value: 0 },
        uHalfBounds: { value: new THREE.Vector2(6.4, 3.6) },
        uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 1.5) },
      },
      vertexShader: FLUID_VERT,
      fragmentShader: FLUID_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(this.geo, this.mat);
    this.points.frustumCulled = false;
    this.root.add(this.points);
  }

  public setQuality(quality: GraphicsQuality): void {
    const next = quality === 'ULTRA' ? ULTRA_PARTICLES : PRO_PARTICLES;
    if (next === this.activeCount) return;
    this.activeCount = next;
    this.geo.setDrawRange(0, this.activeCount);
  }

  public getParticleCount(): number {
    return this.activeCount;
  }

  private nextRand(): number {
    this.respawnSeed = (this.respawnSeed * 1664525 + 1013904223) >>> 0;
    return this.respawnSeed / 4294967296;
  }

  private seedParticles(hw: number, hh: number): void {
    let s = 133791;
    const rand = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };

    for (let i = 0; i < ULTRA_PARTICLES; i += 1) {
      // Distribute organically across gentle flowing silk strata inside safe 88% interior
      const x = (rand() * 2 - 1) * hw * 0.88;
      const stratum = (rand() * 2 - 1) * 0.78;
      const waveY = Math.sin(x * 0.62 + rand() * 0.85) * 0.16;
      const y = Math.max(-hh * 0.84, Math.min(hh * 0.84, (stratum + waveY) * hh));
      const vx0 = (rand() - 0.5) * 0.25;
      const vy0 = (rand() - 0.5) * 0.25;

      this.px[i] = x;
      this.py[i] = y;
      this.vx[i] = vx0;
      this.vy[i] = vy0;
      this.seeds[i] = rand();
      this.sizeBuf[i] = rand(); // 0..1 normalized, mapped to 2.5px..6.5px in vertex shader
      this.speedBuf[i] = 0.08;

      const i3 = i * 3;
      this.posBuf[i3] = x;
      this.posBuf[i3 + 1] = y;
      this.posBuf[i3 + 2] = (rand() - 0.5) * 0.3;
    }
    this.posAttr.needsUpdate = true;
    this.speedAttr.needsUpdate = true;
  }

  public reset(): void {
    for (let slot = 0; slot < 2; slot += 1) {
      this.handStates[slot].hasPrev = false;
      this.handStates[slot].smoothVx = 0;
      this.handStates[slot].smoothVy = 0;
      this.handStates[slot].chargeHoldSec = 0;
      this.waves[slot].active = false;
    }
  }

  public clear(): void {
    this.root.visible = false;
    this.mat.uniforms.uOpacity.value = 0;
  }

  public update(
    hands: HandPose2D[],
    dt: number,
    fadeAlpha: number,
    viewW: number,
    viewH: number,
    screenToWorld: ScreenToWorldFn,
  ): boolean {
    if (fadeAlpha <= 0.008) {
      this.clear();
      return false;
    }

    const safeDt = Math.min(Math.max(dt, 0.003), 0.04);
    this.elapsed += safeDt;

    const corner = screenToWorld(viewW, 0);
    const hw = Math.max(2.5, Math.abs(corner.x));
    const hh = Math.max(1.8, Math.abs(corner.y));
    if (!this.initializedBounds) {
      this.initializedBounds = true;
      this.seedParticles(hw, hh);
    }
    this.mat.uniforms.uHalfBounds.value.set(hw, hh);
    this.mat.uniforms.uPixelRatio.value = Math.min(window.devicePixelRatio || 1, 1.5);

    // 0 = none, 1 = Wave / Cohesive Silk Flow, 2 = Pinch Galaxy Vortex, 3 = Fist Energy Charge, 4 = Supernova Burst
    const hMode = [0, 0];
    const hX = [0, 0];
    const hY = [0, 0];
    const hVx = [0, 0];
    const hVy = [0, 0];
    const hSpd = [0, 0];
    const seen = [false, false];

    const invW = 1 / Math.max(viewW, 1);
    const invH = 1 / Math.max(viewH, 1);
    const velAlpha = 1 - Math.exp(-15 * safeDt);

    for (let i = 0; i < hands.length && i < 2; i += 1) {
      const h = hands[i];
      const slot = h.id === 1 ? 1 : 0;
      seen[slot] = true;
      const st = this.handStates[slot];

      if (st.burstCooldownSec > 0) {
        st.burstCooldownSec = Math.max(0, st.burstCooldownSec - safeDt);
      }

      const pinchNorm = Math.hypot(
        (h.pts[8] - h.pts[16]) * invW,
        (h.pts[9] - h.pts[17]) * invH,
      );
      const isPinch = pinchNorm < PINCH_NORM_THRESHOLD || h.gesture === 'PINCH';
      const extended = countExtendedFingers2D(h);
      const isFist = !isPinch && (extended <= 1 || h.gesture === 'FIST');
      const isOpenPalm = !isPinch && !isFist && extended >= 4;

      const targetPxX = isPinch ? (h.pts[8] + h.pts[16]) * 0.5 : h.palmX;
      const targetPxY = isPinch ? (h.pts[9] + h.pts[17]) * 0.5 : h.palmY;
      const wPos = screenToWorld(targetPxX, targetPxY);

      const rawVx = st.hasPrev ? (wPos.x - st.prevWx) / safeDt : 0;
      const rawVy = st.hasPrev ? (wPos.y - st.prevWy) / safeDt : 0;
      const clampedVx = Math.max(-12, Math.min(12, rawVx));
      const clampedVy = Math.max(-12, Math.min(12, rawVy));
      st.smoothVx += (clampedVx - st.smoothVx) * velAlpha;
      st.smoothVy += (clampedVy - st.smoothVy) * velAlpha;

      hX[slot] = wPos.x;
      hY[slot] = wPos.y;
      hVx[slot] = st.smoothVx;
      hVy[slot] = st.smoothVy;
      hSpd[slot] = Math.hypot(st.smoothVx, st.smoothVy);

      if (isPinch) {
        st.chargeHoldSec += safeDt;
        hMode[slot] = 2; // Pinch -> Glowing Galaxy Vortex
      } else if (isFist) {
        st.chargeHoldSec += safeDt;
        hMode[slot] = 3; // Fist -> Compact Energy Core Charge
      } else if (isOpenPalm) {
        if (st.chargeHoldSec >= CHARGE_HOLD_MIN_SEC && st.burstCooldownSec <= 0) {
          hMode[slot] = 4; // Supernova Burst Ring!
          st.burstCooldownSec = SUPERNOVA_COOLDOWN_SEC;
          const wave = this.waves[slot];
          wave.active = true;
          wave.x = wPos.x;
          wave.y = wPos.y;
          wave.radius = 0.25;
          wave.age = 0;
          wave.duration = 0.58;
        } else {
          hMode[slot] = 1; // Cohesive Silk-Ribbon Flow
        }
        st.chargeHoldSec = 0;
      } else {
        st.chargeHoldSec = Math.max(0, st.chargeHoldSec - safeDt * 2.5);
        hMode[slot] = 1; // Always hug & weave silk ribbons around hand trajectory
      }

      st.hasPrev = true;
      st.prevWx = wPos.x;
      st.prevWy = wPos.y;
    }

    for (let slot = 0; slot < 2; slot += 1) {
      if (!seen[slot]) {
        this.handStates[slot].hasPrev = false;
        this.handStates[slot].smoothVx = 0;
        this.handStates[slot].smoothVy = 0;
        this.handStates[slot].chargeHoldSec = 0;
        if (this.handStates[slot].burstCooldownSec > 0) {
          this.handStates[slot].burstCooldownSec = Math.max(
            0,
            this.handStates[slot].burstCooldownSec - safeDt,
          );
        }
      }
      const w = this.waves[slot];
      if (w.active) {
        w.age += safeDt;
        w.radius = 0.25 + (w.age / w.duration) * 5.6;
        if (w.age >= w.duration) w.active = false;
      }
    }

    const count = this.activeCount;
    const px = this.px;
    const py = this.py;
    const vx = this.vx;
    const vy = this.vy;
    const posBuf = this.posBuf;
    const speedBuf = this.speedBuf;
    const t = this.elapsed;

    const anyHand = hMode[0] !== 0 || hMode[1] !== 0;
    const ambientGain = anyHand ? 0.44 : 1.05; // Lower scatter when hands weave silk ribbons
    const drag = Math.exp((anyHand ? -2.25 : -1.85) * safeDt);
    const softX = hw * 0.84;
    const softY = hh * 0.80;
    const hardX = hw * 1.02;
    const hardY = hh * 1.02;
    const maxSpdSq = MAX_PARTICLE_SPEED * MAX_PARTICLE_SPEED;

    const m0 = hMode[0];
    const m1 = hMode[1];
    const x0 = hX[0];
    const y0 = hY[0];
    const vx0 = hVx[0];
    const vy0 = hVy[0];
    const sp0 = hSpd[0];
    const x1 = hX[1];
    const y1 = hY[1];
    const vx1 = hVx[1];
    const vy1 = hVy[1];
    const sp1 = hSpd[1];

    const w0Active = this.waves[0].active;
    const w0x = this.waves[0].x;
    const w0y = this.waves[0].y;
    const w0r = this.waves[0].radius;
    const w0Power = w0Active ? (1 - this.waves[0].age / this.waves[0].duration) * 24.0 : 0;

    const w1Active = this.waves[1].active;
    const w1x = this.waves[1].x;
    const w1y = this.waves[1].y;
    const w1r = this.waves[1].radius;
    const w1Power = w1Active ? (1 - this.waves[1].age / this.waves[1].duration) * 24.0 : 0;

    // Strictly divergence-free 2D stream-function curl phases (zero wall/bottom compression)
    const wt1 = t * 0.38;
    const wt2 = t * 0.31;
    const wt3 = t * 0.52;
    const wt4 = t * 0.45;

    for (let i = 0; i < count; i += 1) {
      let x = px[i];
      let y = py[i];
      let pvx = vx[i];
      let pvy = vy[i];

      const kx1 = x * 0.54 + wt1;
      const ky1 = y * 0.54 - wt2;
      const s1x = Math.sin(kx1);
      const c1x = Math.cos(kx1);
      const s1y = Math.sin(ky1);
      const c1y = Math.cos(ky1);

      const p2 = (x - y) * 0.42 + wt3;
      const p3 = (x + y) * 0.34 - wt4;
      const s2 = Math.sin(p2) * 0.56;
      const c3 = Math.cos(p3) * 0.42;

      pvx += (-s1x * s1y * 1.08 + s2 + c3) * ambientGain * safeDt;
      pvy += (-c1x * c1y * 1.08 + s2 - c3) * ambientGain * safeDt;

      // Hand 0 interaction
      if (m0 !== 0) {
        const dx = x - x0;
        const dy = y - y0;
        const d2 = dx * dx + dy * dy;
        if (m0 === 2 || m0 === 3) {
          // Pinch Galaxy Vortex (m=2) or Fist Compact Charge (m=3)
          if (d2 < 26.0) {
            const dist = Math.sqrt(d2) + 0.08;
            const invD = 1 / dist;
            const nx = dx * invD;
            const ny = dy * invD;
            const fall = 1 / (dist + 0.46);
            const eyeR = m0 === 2 ? 0.42 : 0.34;
            const pullMul = m0 === 2 ? 12.2 : 15.5;
            const swirlMul = m0 === 2 ? 25.5 : 14.0;
            const radial = dist > eyeR ? -(dist - eyeR) * pullMul * fall : (eyeR - dist) * 16.5;
            pvx += (nx * radial - ny * swirlMul * fall) * safeDt;
            pvy += (ny * radial + nx * swirlMul * fall) * safeDt;
          }
        } else if (m0 === 4) {
          // Initial Supernova Burst impulse
          if (d2 < 24.0) {
            const dist = Math.sqrt(d2) + 0.08;
            const invD = 1 / dist;
            const strength = (1 - dist / 4.9) * 13.5;
            if (strength > 0) {
              pvx += dx * invD * strength;
              pvy += dy * invD * strength;
            }
          }
        } else if (m0 === 1) {
          // Cohesive Wave Silk-Ribbon Entrainment around palm trajectory
          if (d2 < 20.0) {
            const dist = Math.sqrt(d2) + 0.1;
            const invD = 1 / dist;
            const nx = dx * invD;
            const ny = dy * invD;
            const fall = Math.max(0, 1 - dist / 4.45);
            const fall2 = fall * fall;
            const waveBoost = Math.min(1.0, sp0 / 4.0);
            const orbitR = 0.88 - waveBoost * 0.25;
            const radial = dist > orbitR ? -(11.8 + waveBoost * 4.5) * fall : (orbitR - dist) * 8.2;
            const swirl = (12.8 + waveBoost * 5.2) * fall;
            const wake = (5.8 + waveBoost * 3.8) * fall2;
            pvx += (nx * radial - ny * swirl + vx0 * wake) * safeDt;
            pvy += (ny * radial + nx * swirl + vy0 * wake) * safeDt;
          }
        }
      }

      // Hand 1 interaction (counter-rotating dual-hand symmetry)
      if (m1 !== 0) {
        const dx = x - x1;
        const dy = y - y1;
        const d2 = dx * dx + dy * dy;
        if (m1 === 2 || m1 === 3) {
          if (d2 < 26.0) {
            const dist = Math.sqrt(d2) + 0.08;
            const invD = 1 / dist;
            const nx = dx * invD;
            const ny = dy * invD;
            const fall = 1 / (dist + 0.46);
            const eyeR = m1 === 2 ? 0.42 : 0.34;
            const pullMul = m1 === 2 ? 12.2 : 15.5;
            const swirlMul = m1 === 2 ? 25.5 : 14.0;
            const radial = dist > eyeR ? -(dist - eyeR) * pullMul * fall : (eyeR - dist) * 16.5;
            pvx += (nx * radial + ny * swirlMul * fall) * safeDt;
            pvy += (ny * radial - nx * swirlMul * fall) * safeDt;
          }
        } else if (m1 === 4) {
          if (d2 < 24.0) {
            const dist = Math.sqrt(d2) + 0.08;
            const invD = 1 / dist;
            const strength = (1 - dist / 4.9) * 13.5;
            if (strength > 0) {
              pvx += dx * invD * strength;
              pvy += dy * invD * strength;
            }
          }
        } else if (m1 === 1) {
          if (d2 < 20.0) {
            const dist = Math.sqrt(d2) + 0.1;
            const invD = 1 / dist;
            const nx = dx * invD;
            const ny = dy * invD;
            const fall = Math.max(0, 1 - dist / 4.45);
            const fall2 = fall * fall;
            const waveBoost = Math.min(1.0, sp1 / 4.0);
            const orbitR = 0.88 - waveBoost * 0.25;
            const radial = dist > orbitR ? -(11.8 + waveBoost * 4.5) * fall : (orbitR - dist) * 8.2;
            const swirl = (12.8 + waveBoost * 5.2) * fall;
            const wake = (5.8 + waveBoost * 3.8) * fall2;
            pvx += (nx * radial + ny * swirl + vx1 * wake) * safeDt;
            pvy += (ny * radial - nx * swirl + vy1 * wake) * safeDt;
          }
        }
      }

      // Expanding circular Supernova Burst shockwave rings (pushed smoothly outward)
      if (w0Active) {
        const dx = x - w0x;
        const dy = y - w0y;
        const dist = Math.sqrt(dx * dx + dy * dy) + 0.05;
        const band = dist - w0r;
        if (band > -0.75 && band < 0.75) {
          const ringF = Math.exp(-(band * band) * 5.5) * w0Power;
          pvx += (dx / dist) * ringF * safeDt;
          pvy += (dy / dist) * ringF * safeDt;
        }
      }
      if (w1Active) {
        const dx = x - w1x;
        const dy = y - w1y;
        const dist = Math.sqrt(dx * dx + dy * dy) + 0.05;
        const band = dist - w1r;
        if (band > -0.75 && band < 0.75) {
          const ringF = Math.exp(-(band * band) * 5.5) * w1Power;
          pvx += (dx / dist) * ringF * safeDt;
          pvy += (dy / dist) * ringF * safeDt;
        }
      }

      // Soft inward elastic boundary repulsion prevents particles from piling at bottom/side edges
      if (x > softX) {
        pvx -= (x - softX) * 7.8 * safeDt;
      } else if (x < -softX) {
        pvx += (-softX - x) * 7.8 * safeDt;
      }
      if (y > softY) {
        pvy -= (y - softY) * 8.8 * safeDt;
      } else if (y < -softY) {
        pvy += (-softY - y) * 8.8 * safeDt;
      }

      pvx *= drag;
      pvy *= drag;

      const spdSq = pvx * pvx + pvy * pvy;
      if (spdSq > maxSpdSq) {
        const invS = MAX_PARTICLE_SPEED / Math.sqrt(spdSq);
        pvx *= invS;
        pvy *= invS;
      }

      x += pvx * safeDt;
      y += pvy * safeDt;

      // Clean interior respawn on boundary exit with zero velocity (zero cross-screen streaks)
      if (x > hardX || x < -hardX || y > hardY || y < -hardY) {
        x = (this.nextRand() * 2 - 1) * hw * 0.82;
        y = (this.nextRand() * 2 - 1) * hh * 0.78;
        pvx = 0;
        pvy = 0;
      }

      px[i] = x;
      py[i] = y;
      vx[i] = pvx;
      vy[i] = pvy;

      const i3 = i * 3;
      posBuf[i3] = x;
      posBuf[i3 + 1] = y;
      speedBuf[i] = Math.min(1.0, Math.sqrt(pvx * pvx + pvy * pvy) / 6.8);
    }

    this.posAttr.needsUpdate = true;
    this.speedAttr.needsUpdate = true;
    this.mat.uniforms.uOpacity.value = fadeAlpha;
    this.root.visible = true;
    return true;
  }

  public dispose(): void {
    this.geo.dispose();
    this.mat.dispose();
  }
}
