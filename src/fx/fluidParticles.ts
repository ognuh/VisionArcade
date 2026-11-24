import * as THREE from 'three';
import type { HandPose2D } from './types';

const PARTICLE_COUNT = 8000;
const PINCH_NORM_THRESHOLD = 0.055;
const PINCH_BURST_HOLD_SEC = 0.12;
const SUPERNOVA_COOLDOWN_SEC = 0.65;

const FLUID_VERT = `
attribute float aSpeed;
attribute float aSize;
attribute float aSeed;

uniform float uOpacity;
varying float vSpeed;
varying float vSeed;
varying float vOpacity;

void main() {
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPos;
  vSpeed = aSpeed;
  vSeed = aSeed;
  vOpacity = uOpacity;
  float speedBoost = 1.0 + clamp(aSpeed, 0.0, 1.2) * 0.55;
  gl_PointSize = clamp(aSize * speedBoost * (11.2 / max(-mvPos.z, 1.0)), 1.6, 11.5);
}
`;

const FLUID_FRAG = `
varying float vSpeed;
varying float vSeed;
varying float vOpacity;

void main() {
  if (vOpacity <= 0.005) discard;
  float dist = length(gl_PointCoord - 0.5);
  if (dist > 0.5) discard;

  // Soft glowing orb with luminous pinpoint core
  float halo = exp(-dist * dist * 11.0);
  float core = exp(-dist * dist * 34.0);

  float s = clamp(vSpeed, 0.0, 1.0);
  vec3 cDeepBlue = vec3(0.14, 0.30, 0.88);
  vec3 cIndigo   = vec3(0.38, 0.22, 0.96);
  vec3 cCyan     = vec3(0.10, 0.92, 1.24);
  vec3 cMagenta  = vec3(1.42, 0.28, 0.92);
  vec3 cGold     = vec3(1.52, 1.32, 0.78);

  vec3 idleCol = mix(cDeepBlue, cIndigo, vSeed);
  vec3 fastCol = mix(cMagenta, cGold, fract(vSeed * 3.7));

  vec3 col = mix(idleCol, cCyan, smoothstep(0.08, 0.48, s));
  col = mix(col, fastCol, smoothstep(0.50, 0.96, s));
  col += vec3(0.55, 0.70, 0.85) * core * (0.45 + s * 0.75);

  float alpha = clamp((halo * 0.78 + core * 0.55) * vOpacity, 0.0, 1.0);
  gl_FragColor = vec4(col * (0.75 + s * 0.45), alpha);
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
 * 8,000-Particle Zero-GC Fluid Physics Sandbox (Three.js Points + Shared UnrealBloomPass):
 * - Velocity-mapped HDR color ramp (Deep Blue/Indigo -> Electric Cyan -> White-Gold/Magenta)
 * - Ambient curl-field drift when idle
 * - Independent 2-hand physics:
 *   1) Gentle hand sweep: directional wake + fluid swirl
 *   2) Pinch (thumb 4 & index 8 < 0.055): Gravitational Vortex spiral
 *   3) Open Palm (>= 4 fingers): Radial Repulsion + Pinch->Open Supernova Burst!
 */
export class FluidParticleSandbox {
  public readonly root = new THREE.Group();

  private px = new Float32Array(PARTICLE_COUNT);
  private py = new Float32Array(PARTICLE_COUNT);
  private vx = new Float32Array(PARTICLE_COUNT);
  private vy = new Float32Array(PARTICLE_COUNT);
  private seeds = new Float32Array(PARTICLE_COUNT);

  private posBuf = new Float32Array(PARTICLE_COUNT * 3);
  private speedBuf = new Float32Array(PARTICLE_COUNT);
  private sizeBuf = new Float32Array(PARTICLE_COUNT);

  private posAttr: THREE.BufferAttribute;
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
    this.speedAttr = new THREE.BufferAttribute(this.speedBuf, 1);
    this.speedAttr.setUsage(THREE.DynamicDrawUsage);

    this.seedParticles(6.4, 3.6);

    this.geo.setAttribute('position', this.posAttr);
    this.geo.setAttribute('aSpeed', this.speedAttr);
    this.geo.setAttribute('aSize', new THREE.BufferAttribute(this.sizeBuf, 1));
    this.geo.setAttribute('aSeed', new THREE.BufferAttribute(this.seeds, 1));

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

  private seedParticles(hw: number, hh: number): void {
    let s = 133791;
    const rand = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };

    for (let i = 0; i < PARTICLE_COUNT; i += 1) {
      const x = (rand() * 2 - 1) * hw * 0.98;
      const y = (rand() * 2 - 1) * hh * 0.98;
      this.px[i] = x;
      this.py[i] = y;
      this.vx[i] = (rand() - 0.5) * 0.4;
      this.vy[i] = (rand() - 0.5) * 0.4;
      this.seeds[i] = rand();
      this.sizeBuf[i] = 2.0 + rand() * 3.1;
      this.speedBuf[i] = 0.05;

      const i3 = i * 3;
      this.posBuf[i3] = x;
      this.posBuf[i3 + 1] = y;
      this.posBuf[i3 + 2] = (rand() - 0.5) * 0.35;
    }
    this.posAttr.needsUpdate = true;
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

    const safeDt = Math.min(Math.max(dt, 0.004), 0.04);
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

    // Prepare up to 2 independent hand force descriptors (Zero-GC fixed scalars)
    // Mode: 0=none, 1=sweep, 2=pinch vortex, 3=open palm repulse, 4=supernova burst
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

      // Pinch distance between thumb tip (4 -> idx 8,9) & index tip (8 -> idx 16,17) in normalized screen units
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
      hVx[slot] = Math.max(-18, Math.min(18, vxHand));
      hVy[slot] = Math.max(-18, Math.min(18, vyHand));

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
        hMode[slot] = handSpeed > 0.25 ? 1 : 0; // Gentle Sweep wake
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

    // Integrate 8,000 particles in one tight Zero-GC pass
    const px = this.px;
    const py = this.py;
    const vx = this.vx;
    const vy = this.vy;
    const posBuf = this.posBuf;
    const speedBuf = this.speedBuf;
    const t = this.elapsed;
    const drag = Math.exp(-2.15 * safeDt);
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

    for (let i = 0; i < PARTICLE_COUNT; i += 1) {
      let x = px[i];
      let y = py[i];
      let pvx = vx[i];
      let pvy = vy[i];

      // Ambient divergence-free curl drift keeps idle particles alive & luminous
      const curlX = Math.cos(y * 0.85 + t * 0.65) * 0.95 + Math.sin(x * 0.55 - t * 0.42) * 0.48;
      const curlY = Math.sin(x * 0.85 - t * 0.55) * 0.95 - Math.cos(y * 0.55 + t * 0.38) * 0.48;
      pvx += curlX * safeDt;
      pvy += curlY * safeDt;

      // Hand 0 interaction
      if (m0 !== 0) {
        const dx = x - x0;
        const dy = y - y0;
        const d2 = dx * dx + dy * dy;
        if (m0 === 2) {
          // Gravitational Vortex: inward pull + orbital tangential swirl
          if (d2 < 24.0) {
            const dist = Math.sqrt(d2) + 0.08;
            const invD = 1 / dist;
            const nx = dx * invD;
            const ny = dy * invD;
            const fall = 1 / (dist + 0.42);
            const pull = 23.0 * fall;
            const swirl = 28.0 * fall;
            pvx += (-nx * pull - ny * swirl) * safeDt;
            pvy += (-ny * pull + nx * swirl) * safeDt;
            // Core velocity tether prevents singularity ejection (dt-normalized for 120Hz)
            if (dist < 0.35) {
              pvx *= coreDamp;
              pvy *= coreDamp;
            }
          }
        } else if (m0 === 4) {
          // Supernova Burst: explosive radial launch of condensed galaxy!
          if (d2 < 26.0) {
            const dist = Math.sqrt(d2) + 0.06;
            const invD = 1 / dist;
            const strength = (1 - dist / 5.1) * 24.0;
            if (strength > 0) {
              pvx += dx * invD * strength;
              pvy += dy * invD * strength;
            }
          }
        } else if (m0 === 3) {
          // Open Palm Radial Repulsion
          if (d2 < 10.5) {
            const dist = Math.sqrt(d2) + 0.1;
            const invD = 1 / dist;
            const fall = Math.max(0, 1 - dist / 3.24);
            const repulse = 32.0 * fall * fall;
            pvx += dx * invD * repulse * safeDt;
            pvy += dy * invD * repulse * safeDt;
          }
        } else if (m0 === 1) {
          // Gentle Hand Sweep: directional flow + fluid wake swirl
          if (d2 < 6.25) {
            const dist = Math.sqrt(d2) + 0.12;
            const fall = Math.max(0, 1 - dist / 2.5);
            const w = fall * fall * safeDt;
            pvx += (vx0 * 6.5 - (dy / dist) * 5.2) * w;
            pvy += (vy0 * 6.5 + (dx / dist) * 5.2) * w;
          }
        }
      }

      // Hand 1 interaction (fully independent)
      if (m1 !== 0) {
        const dx = x - x1;
        const dy = y - y1;
        const d2 = dx * dx + dy * dy;
        if (m1 === 2) {
          if (d2 < 24.0) {
            const dist = Math.sqrt(d2) + 0.08;
            const invD = 1 / dist;
            const nx = dx * invD;
            const ny = dy * invD;
            const fall = 1 / (dist + 0.42);
            const pull = 23.0 * fall;
            const swirl = 28.0 * fall;
            pvx += (-nx * pull + ny * swirl) * safeDt;
            pvy += (-ny * pull - nx * swirl) * safeDt;
            if (dist < 0.35) {
              pvx *= coreDamp;
              pvy *= coreDamp;
            }
          }
        } else if (m1 === 4) {
          if (d2 < 26.0) {
            const dist = Math.sqrt(d2) + 0.06;
            const invD = 1 / dist;
            const strength = (1 - dist / 5.1) * 24.0;
            if (strength > 0) {
              pvx += dx * invD * strength;
              pvy += dy * invD * strength;
            }
          }
        } else if (m1 === 3) {
          if (d2 < 10.5) {
            const dist = Math.sqrt(d2) + 0.1;
            const invD = 1 / dist;
            const fall = Math.max(0, 1 - dist / 3.24);
            const repulse = 32.0 * fall * fall;
            pvx += dx * invD * repulse * safeDt;
            pvy += dy * invD * repulse * safeDt;
          }
        } else if (m1 === 1) {
          if (d2 < 6.25) {
            const dist = Math.sqrt(d2) + 0.12;
            const fall = Math.max(0, 1 - dist / 2.5);
            const w = fall * fall * safeDt;
            pvx += (vx1 * 6.5 + (dy / dist) * 5.2) * w;
            pvy += (vy1 * 6.5 - (dx / dist) * 5.2) * w;
          }
        }
      }

      pvx *= drag;
      pvy *= drag;

      // Clamp extreme velocities for numerical stability
      const spdSq = pvx * pvx + pvy * pvy;
      if (spdSq > 484) {
        const invS = 22 / Math.sqrt(spdSq);
        pvx *= invS;
        pvy *= invS;
      }

      x += pvx * safeDt;
      y += pvy * safeDt;

      // Soft toroidal wrap with slight damping keeps 8,000 particles evenly distributed
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
      speedBuf[i] = Math.min(1.0, Math.sqrt(pvx * pvx + pvy * pvy) / 8.5);
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
