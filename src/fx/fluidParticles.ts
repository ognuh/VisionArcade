import * as THREE from 'three';
import type { GraphicsQuality } from '../types/vision';
import type { HandPose2D } from './types';

const MAX_PARTICLES = 15000;
const HIGH_PARTICLES = 9000;
const PINCH_NORM_THRESHOLD = 0.055;
const PINCH_BURST_HOLD_SEC = 0.12;
const SUPERNOVA_COOLDOWN_SEC = 0.65;

const FLUID_VERT = `
attribute float aSpeed;
attribute float aSize;
attribute float aSeed;
attribute vec2 aVel;

uniform float uOpacity;
varying float vSpeed;
varying float vSeed;
varying float vOpacity;
varying vec2 vVelDir;
varying float vStretch;

void main() {
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPos;
  vSpeed = aSpeed;
  vSeed = aSeed;
  vOpacity = uOpacity;

  // Screen-space velocity direction for Velocity Motion-Stretch meteor streaks
  vec2 vScreen = vec2(aVel.x, -aVel.y);
  float vLen = length(vScreen);
  vVelDir = vLen > 0.05 ? (vScreen / vLen) : vec2(1.0, 0.0);
  vStretch = 1.0 + smoothstep(0.14, 1.0, aSpeed) * 3.6;

  float sizeBoost = 1.0 + smoothstep(0.10, 1.0, aSpeed) * 1.45;
  gl_PointSize = clamp(aSize * sizeBoost * (11.5 / max(-mvPos.z, 1.0)), 1.8, 22.0);
}
`;

const FLUID_FRAG = `
varying float vSpeed;
varying float vSeed;
varying float vOpacity;
varying vec2 vVelDir;
varying float vStretch;

void main() {
  if (vOpacity <= 0.005) discard;
  vec2 p = gl_PointCoord - 0.5;

  // Anisotropic velocity-aligned coordinate system: compresses width perpendicular to motion
  vec2 perp = vec2(-vVelDir.y, vVelDir.x);
  float along = dot(p, vVelDir);
  float across = dot(p, perp) * vStretch;

  // Slightly forward-weighted comet head with glowing tail
  vec2 stretched = vec2(along * 1.05, across);
  float distSq = dot(stretched, stretched);
  if (distSq > 0.25) discard;

  float halo = exp(-distSq * 14.0);
  float core = exp(-distSq * 46.0);

  float s = clamp(vSpeed, 0.0, 1.0);
  vec3 cDeepBlue = vec3(0.11, 0.26, 0.92);
  vec3 cIndigo   = vec3(0.36, 0.18, 0.98);
  vec3 cCyan     = vec3(0.08, 0.96, 1.38);
  vec3 cMagenta  = vec3(1.52, 0.26, 0.98);
  vec3 cGold     = vec3(1.66, 1.44, 0.86);

  vec3 idleCol = mix(cDeepBlue, cIndigo, vSeed);
  vec3 fastCol = mix(cGold, cMagenta, smoothstep(0.35, 0.85, fract(vSeed * 3.7)));

  vec3 col = mix(idleCol, cCyan, smoothstep(0.07, 0.44, s));
  col = mix(col, fastCol, smoothstep(0.46, 0.94, s));
  col += vec3(0.75, 0.86, 1.0) * core * (0.50 + s * 0.85);

  float alpha = clamp((halo * 0.78 + core * 0.62) * vOpacity, 0.0, 1.0);
  gl_FragColor = vec4(col * (0.78 + s * 0.52), alpha);
}
`;

interface ScreenToWorldFn {
  (px: number, py: number): { x: number; y: number; pxToWorld: number };
}

interface FluidHandState {
  hasPrev: boolean;
  prevWx: number;
  prevWy: number;
  pinchHoldSec: number;
  burstCooldownSec: number;
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
 * Ultra 15,000-Particle Zero-GC GPU Fluid Sandbox (Three.js Points + Shared Half-Res UnrealBloomPass):
 * - Velocity Motion-Stretch shader turns fast sweeps & Supernova Bursts into shooting-star meteor streaks
 * - Velocity-mapped HDR palette: Deep Blue -> Electric Cyan -> White-Gold / Magenta
 * - Ambient divergence-free curl-noise drift + independent 2-hand Vortex / Repulse / Supernova physics
 */
export class FluidParticleSandbox {
  public readonly root = new THREE.Group();

  private activeCount = MAX_PARTICLES;
  private px = new Float32Array(MAX_PARTICLES);
  private py = new Float32Array(MAX_PARTICLES);
  private vx = new Float32Array(MAX_PARTICLES);
  private vy = new Float32Array(MAX_PARTICLES);
  private seeds = new Float32Array(MAX_PARTICLES);

  private posBuf = new Float32Array(MAX_PARTICLES * 3);
  private velBuf = new Float32Array(MAX_PARTICLES * 2);
  private speedBuf = new Float32Array(MAX_PARTICLES);
  private sizeBuf = new Float32Array(MAX_PARTICLES);

  private posAttr: THREE.BufferAttribute;
  private velAttr: THREE.BufferAttribute;
  private speedAttr: THREE.BufferAttribute;
  private geo: THREE.BufferGeometry;
  private mat: THREE.ShaderMaterial;
  private points: THREE.Points;

  private elapsed = 0;
  private halfW = 6.4;
  private halfH = 3.6;
  private initializedBounds = false;

  private handStates: [FluidHandState, FluidHandState] = [
    { hasPrev: false, prevWx: 0, prevWy: 0, pinchHoldSec: 0, burstCooldownSec: 0 },
    { hasPrev: false, prevWx: 0, prevWy: 0, pinchHoldSec: 0, burstCooldownSec: 0 },
  ];

  constructor() {
    this.root.visible = false;

    this.geo = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(this.posBuf, 3);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.velAttr = new THREE.BufferAttribute(this.velBuf, 2);
    this.velAttr.setUsage(THREE.DynamicDrawUsage);
    this.speedAttr = new THREE.BufferAttribute(this.speedBuf, 1);
    this.speedAttr.setUsage(THREE.DynamicDrawUsage);

    this.seedParticles(6.4, 3.6);

    this.geo.setAttribute('position', this.posAttr);
    this.geo.setAttribute('aVel', this.velAttr);
    this.geo.setAttribute('aSpeed', this.speedAttr);
    this.geo.setAttribute('aSize', new THREE.BufferAttribute(this.sizeBuf, 1));
    this.geo.setAttribute('aSeed', new THREE.BufferAttribute(this.seeds, 1));
    this.geo.setDrawRange(0, this.activeCount);

    this.mat = new THREE.ShaderMaterial({
      uniforms: {
        uOpacity: { value: 0 },
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
    const next = quality === 'ULTRA' ? MAX_PARTICLES : HIGH_PARTICLES;
    if (next === this.activeCount) return;
    this.activeCount = next;
    this.geo.setDrawRange(0, this.activeCount);
  }

  public getParticleCount(): number {
    return this.activeCount;
  }

  private seedParticles(hw: number, hh: number): void {
    let s = 133791;
    const rand = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };

    for (let i = 0; i < MAX_PARTICLES; i += 1) {
      const x = (rand() * 2 - 1) * hw * 0.98;
      const y = (rand() * 2 - 1) * hh * 0.98;
      const vx0 = (rand() - 0.5) * 0.45;
      const vy0 = (rand() - 0.5) * 0.45;
      this.px[i] = x;
      this.py[i] = y;
      this.vx[i] = vx0;
      this.vy[i] = vy0;
      this.seeds[i] = rand();
      this.sizeBuf[i] = 1.8 + rand() * 2.9;
      this.speedBuf[i] = 0.05;

      const i3 = i * 3;
      this.posBuf[i3] = x;
      this.posBuf[i3 + 1] = y;
      this.posBuf[i3 + 2] = (rand() - 0.5) * 0.4;

      const i2 = i * 2;
      this.velBuf[i2] = vx0;
      this.velBuf[i2 + 1] = vy0;
    }
    this.posAttr.needsUpdate = true;
    this.velAttr.needsUpdate = true;
    this.speedAttr.needsUpdate = true;
  }

  public reset(): void {
    this.handStates[0].hasPrev = false;
    this.handStates[0].pinchHoldSec = 0;
    this.handStates[1].hasPrev = false;
    this.handStates[1].pinchHoldSec = 0;
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
      this.halfW = hw;
      this.halfH = hh;
      this.seedParticles(hw, hh);
    } else {
      this.halfW = hw;
      this.halfH = hh;
    }

    const hMode = [0, 0];
    const hX = [0, 0];
    const hY = [0, 0];
    const hVx = [0, 0];
    const hVy = [0, 0];
    const seen = [false, false];

    const invW = 1 / Math.max(viewW, 1);
    const invH = 1 / Math.max(viewH, 1);

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
      const isOpenPalm = !isPinch && extended >= 4;

      const targetPxX = isPinch ? (h.pts[8] + h.pts[16]) * 0.5 : h.palmX;
      const targetPxY = isPinch ? (h.pts[9] + h.pts[17]) * 0.5 : h.palmY;
      const wPos = screenToWorld(targetPxX, targetPxY);

      const vxHand = st.hasPrev ? (wPos.x - st.prevWx) / safeDt : 0;
      const vyHand = st.hasPrev ? (wPos.y - st.prevWy) / safeDt : 0;
      const handSpeed = Math.hypot(vxHand, vyHand);

      hX[slot] = wPos.x;
      hY[slot] = wPos.y;
      hVx[slot] = Math.max(-20, Math.min(20, vxHand));
      hVy[slot] = Math.max(-20, Math.min(20, vyHand));

      if (isPinch) {
        st.pinchHoldSec += safeDt;
        hMode[slot] = 2; // Gravitational Vortex
      } else if (isOpenPalm) {
        if (st.pinchHoldSec >= PINCH_BURST_HOLD_SEC && st.burstCooldownSec <= 0) {
          hMode[slot] = 4; // Supernova Burst!
          st.burstCooldownSec = SUPERNOVA_COOLDOWN_SEC;
        } else {
          hMode[slot] = 3; // Radial Repulsion
        }
        st.pinchHoldSec = 0;
      } else {
        st.pinchHoldSec = Math.max(0, st.pinchHoldSec - safeDt * 2.5);
        hMode[slot] = handSpeed > 0.25 ? 1 : 0;
      }

      st.hasPrev = true;
      st.prevWx = wPos.x;
      st.prevWy = wPos.y;
    }

    for (let slot = 0; slot < 2; slot += 1) {
      if (!seen[slot]) {
        this.handStates[slot].hasPrev = false;
        this.handStates[slot].pinchHoldSec = 0;
        if (this.handStates[slot].burstCooldownSec > 0) {
          this.handStates[slot].burstCooldownSec = Math.max(
            0,
            this.handStates[slot].burstCooldownSec - safeDt,
          );
        }
      }
    }

    const count = this.activeCount;
    const px = this.px;
    const py = this.py;
    const vx = this.vx;
    const vy = this.vy;
    const posBuf = this.posBuf;
    const velBuf = this.velBuf;
    const speedBuf = this.speedBuf;
    const t = this.elapsed;
    const drag = Math.exp(-2.1 * safeDt);
    const coreDamp = Math.exp(-4.35 * safeDt);
    const maxW = this.halfW * 1.02;
    const maxH = this.halfH * 1.02;

    const m0 = hMode[0];
    const m1 = hMode[1];
    const x0 = hX[0];
    const y0 = hY[0];
    const vx0 = hVx[0];
    const vy0 = hVy[0];
    const x1 = hX[1];
    const y1 = hY[1];
    const vx1 = hVx[1];
    const vy1 = hVy[1];

    for (let i = 0; i < count; i += 1) {
      let x = px[i];
      let y = py[i];
      let pvx = vx[i];
      let pvy = vy[i];

      // Ambient divergence-free Curl-Noise drift
      const curlX = Math.cos(y * 0.85 + t * 0.65) * 0.98 + Math.sin(x * 0.55 - t * 0.42) * 0.52;
      const curlY = Math.sin(x * 0.85 - t * 0.55) * 0.98 - Math.cos(y * 0.55 + t * 0.38) * 0.52;
      pvx += curlX * safeDt;
      pvy += curlY * safeDt;

      // Hand 0 interaction
      if (m0 !== 0) {
        const dx = x - x0;
        const dy = y - y0;
        const d2 = dx * dx + dy * dy;
        if (m0 === 2) {
          if (d2 < 25.0) {
            const dist = Math.sqrt(d2) + 0.08;
            const invD = 1 / dist;
            const nx = dx * invD;
            const ny = dy * invD;
            const fall = 1 / (dist + 0.42);
            const pull = 24.0 * fall;
            const swirl = 29.5 * fall;
            pvx += (-nx * pull - ny * swirl) * safeDt;
            pvy += (-ny * pull + nx * swirl) * safeDt;
            if (dist < 0.35) {
              pvx *= coreDamp;
              pvy *= coreDamp;
            }
          }
        } else if (m0 === 4) {
          if (d2 < 28.0) {
            const dist = Math.sqrt(d2) + 0.06;
            const invD = 1 / dist;
            const strength = (1 - dist / 5.3) * 26.5;
            if (strength > 0) {
              pvx += dx * invD * strength;
              pvy += dy * invD * strength;
            }
          }
        } else if (m0 === 3) {
          if (d2 < 11.0) {
            const dist = Math.sqrt(d2) + 0.1;
            const invD = 1 / dist;
            const fall = Math.max(0, 1 - dist / 3.3);
            const repulse = 33.5 * fall * fall;
            pvx += dx * invD * repulse * safeDt;
            pvy += dy * invD * repulse * safeDt;
          }
        } else if (m0 === 1) {
          if (d2 < 6.5) {
            const dist = Math.sqrt(d2) + 0.12;
            const fall = Math.max(0, 1 - dist / 2.55);
            const w = fall * fall * safeDt;
            pvx += (vx0 * 6.8 - (dy / dist) * 5.4) * w;
            pvy += (vy0 * 6.8 + (dx / dist) * 5.4) * w;
          }
        }
      }

      // Hand 1 interaction (independent)
      if (m1 !== 0) {
        const dx = x - x1;
        const dy = y - y1;
        const d2 = dx * dx + dy * dy;
        if (m1 === 2) {
          if (d2 < 25.0) {
            const dist = Math.sqrt(d2) + 0.08;
            const invD = 1 / dist;
            const nx = dx * invD;
            const ny = dy * invD;
            const fall = 1 / (dist + 0.42);
            const pull = 24.0 * fall;
            const swirl = 29.5 * fall;
            pvx += (-nx * pull + ny * swirl) * safeDt;
            pvy += (-ny * pull - nx * swirl) * safeDt;
            if (dist < 0.35) {
              pvx *= coreDamp;
              pvy *= coreDamp;
            }
          }
        } else if (m1 === 4) {
          if (d2 < 28.0) {
            const dist = Math.sqrt(d2) + 0.06;
            const invD = 1 / dist;
            const strength = (1 - dist / 5.3) * 26.5;
            if (strength > 0) {
              pvx += dx * invD * strength;
              pvy += dy * invD * strength;
            }
          }
        } else if (m1 === 3) {
          if (d2 < 11.0) {
            const dist = Math.sqrt(d2) + 0.1;
            const invD = 1 / dist;
            const fall = Math.max(0, 1 - dist / 3.3);
            const repulse = 33.5 * fall * fall;
            pvx += dx * invD * repulse * safeDt;
            pvy += dy * invD * repulse * safeDt;
          }
        } else if (m1 === 1) {
          if (d2 < 6.5) {
            const dist = Math.sqrt(d2) + 0.12;
            const fall = Math.max(0, 1 - dist / 2.55);
            const w = fall * fall * safeDt;
            pvx += (vx1 * 6.8 + (dy / dist) * 5.4) * w;
            pvy += (vy1 * 6.8 - (dx / dist) * 5.4) * w;
          }
        }
      }

      pvx *= drag;
      pvy *= drag;

      const spdSq = pvx * pvx + pvy * pvy;
      if (spdSq > 576) {
        const invS = 24 / Math.sqrt(spdSq);
        pvx *= invS;
        pvy *= invS;
      }

      x += pvx * safeDt;
      y += pvy * safeDt;

      if (x > maxW) x -= maxW * 2;
      else if (x < -maxW) x += maxW * 2;
      if (y > maxH) y -= maxH * 2;
      else if (y < -maxH) y += maxH * 2;

      px[i] = x;
      py[i] = y;
      vx[i] = pvx;
      vy[i] = pvy;

      const i3 = i * 3;
      posBuf[i3] = x;
      posBuf[i3 + 1] = y;

      const i2 = i * 2;
      velBuf[i2] = pvx;
      velBuf[i2 + 1] = pvy;
      speedBuf[i] = Math.min(1.0, Math.sqrt(pvx * pvx + pvy * pvy) / 8.8);
    }

    this.posAttr.needsUpdate = true;
    this.velAttr.needsUpdate = true;
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
