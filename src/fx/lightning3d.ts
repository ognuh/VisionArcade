import * as THREE from 'three';
import { LightningAudioEngine } from '../audio/lightningAudio';
import type { HandPose2D } from './types';

const MAX_RIBBONS = 68;
const PTS_PER_RIBBON = 17;
const VERTS_PER_RIBBON = PTS_PER_RIBBON * 2;
const TRIS_PER_RIBBON = (PTS_PER_RIBBON - 1) * 2;
const TOTAL_VERTS = MAX_RIBBONS * VERTS_PER_RIBBON;
const TOTAL_INDICES = MAX_RIBBONS * TRIS_PER_RIBBON * 3;
const SPARK_POOL = 260;
const FINGERTIP_IDS = [4, 8, 12, 16, 20];

// Continuous 2D gradient noise for smooth organic time-evolving Catmull-Rom spline displacement
const PERM = new Uint8Array(512);
(() => {
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i += 1) p[i] = i;
  let s = 48271;
  for (let i = 255; i > 0; i -= 1) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const j = s % (i + 1);
    const tmp = p[i];
    p[i] = p[j];
    p[j] = tmp;
  }
  for (let i = 0; i < 512; i += 1) PERM[i] = p[i & 255];
})();

function hash2(ix: number, iy: number): number {
  return (PERM[(PERM[ix & 255] + iy) & 255] / 255) * 2 - 1;
}

function noise2D(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

function smoothBoltNoise(u: number, time: number, seed: number): number {
  const n1 = noise2D(u * 4.5 + seed * 3.1, time * 4.2 + seed);
  const n2 = noise2D(u * 10.5 - seed * 5.3, time * 7.8 - seed * 2.1) * 0.45;
  return n1 * 0.72 + n2 * 0.42;
}

function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    0.5 *
    (2 * p1 +
      (-p0 + p2) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
  );
}

// ============================================================================
// Volumetric HDR Plasma Ribbon, 3D Energy Orb & Gravity Spark Shaders
// ============================================================================
const RIBBON_VERT = `
attribute vec2 aUv;
attribute float aAlpha;
varying vec2 vUv;
varying float vAlpha;

void main() {
  vUv = aUv;
  vAlpha = aAlpha;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const RIBBON_FRAG = `
uniform float uTime;
varying vec2 vUv;
varying float vAlpha;

void main() {
  if (vAlpha <= 0.005) discard;
  float d = abs(vUv.y - 0.5) * 2.0; // 0 at filament center, 1 at ribbon edge
  if (d > 1.0) discard;

  float pulse = 0.90 + 0.10 * sin(vUv.x * 26.0 - uTime * 30.0);
  float core = exp(-d * d * 42.0);
  float cyanField = exp(-d * d * 8.2);
  float violetAura = exp(-d * d * 2.5);

  vec3 col =
    vec3(0.52, 0.16, 1.12) * violetAura * 0.48 +
    vec3(0.12, 0.86, 1.44) * cyanField * 0.95 +
    vec3(1.58, 1.78, 2.08) * core * 1.05;

  float alpha = clamp((violetAura * 0.36 + cyanField * 0.62 + core) * vAlpha * pulse, 0.0, 1.0);
  gl_FragColor = vec4(col * pulse, alpha);
}
`;

const ORB_VERT = `
varying vec3 vNormalView;
varying vec3 vViewDir;
varying vec3 vLocalPos;

void main() {
  vLocalPos = normalize(position);
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  vNormalView = normalize(normalMatrix * normal);
  vViewDir = normalize(-mvPos.xyz);
  gl_Position = projectionMatrix * mvPos;
}
`;

const ORB_FRAG = `
uniform float uTime;
uniform float uCharge;
uniform float uOpacity;
varying vec3 vNormalView;
varying vec3 vViewDir;
varying vec3 vLocalPos;

void main() {
  float ndotv = clamp(dot(normalize(vNormalView), normalize(vViewDir)), 0.0, 1.0);
  float bands = 0.5 + 0.5 * sin(vLocalPos.x * 16.0 + uTime * 9.0 + sin(vLocalPos.y * 14.0 - uTime * 7.0));
  float rim = pow(1.0 - ndotv, 2.2);
  float center = pow(ndotv, 1.8);

  vec3 col =
    vec3(0.42, 0.16, 0.98) * rim * (0.95 + uCharge * 0.55) +
    vec3(0.14, 0.76, 1.15) * (0.35 + bands * 0.52) +
    vec3(1.25, 1.42, 1.65) * center * (0.72 + uCharge * 0.35);

  float alpha = clamp((center * 0.88 + rim * 0.80) * uOpacity, 0.0, 1.0);
  gl_FragColor = vec4(col, alpha);
}
`;

const SPARK3D_VERT = `
attribute float aAlpha;
attribute float aSize;
attribute float aHue;
varying float vAlpha;
varying float vHue;

void main() {
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPos;
  vAlpha = aAlpha;
  vHue = aHue;
  gl_PointSize = clamp(aSize * (10.5 / max(-mvPos.z, 1.0)), 1.5, 12.0);
}
`;

const SPARK3D_FRAG = `
varying float vAlpha;
varying float vHue;

void main() {
  if (vAlpha <= 0.01) discard;
  float dist = length(gl_PointCoord - 0.5);
  if (dist > 0.5) discard;
  float g = exp(-dist * dist * 13.0);
  vec3 cCyan = vec3(0.32, 0.86, 1.45);
  vec3 cViolet = vec3(0.76, 0.42, 1.45);
  vec3 cWhite = vec3(1.75, 1.82, 1.95);
  vec3 col = vHue < 0.35 ? cWhite : (vHue < 0.72 ? cCyan : cViolet);
  gl_FragColor = vec4(col * g, g * vAlpha);
}
`;

interface ScreenToWorldFn {
  (px: number, py: number): { x: number; y: number; pxToWorld: number };
}

interface PalmOrb3D {
  root: THREE.Group;
  coreMat: THREE.ShaderMaterial;
  ringA: THREE.Mesh;
  ringB: THREE.Mesh;
  ringAMat: THREE.MeshBasicMaterial;
  ringBMat: THREE.MeshBasicMaterial;
}

/**
 * Cinema-Grade Three.js WebGL2 Volumetric Plasma Lightning Engine (Zero-GC, 60 FPS):
 * Shares EffectComposer + UnrealBloomPass with Cosmos3DEngine for authentic HDR plasma glow.
 */
export class Lightning3DEngine {
  public readonly root = new THREE.Group();

  // Single batched Volumetric Ribbon mesh for ALL lightning bolts & 2-tier branches (1 draw call!)
  private ribbonGeo: THREE.BufferGeometry;
  private ribbonMat: THREE.ShaderMaterial;
  private posBuf = new Float32Array(TOTAL_VERTS * 3);
  private uvBuf = new Float32Array(TOTAL_VERTS * 2);
  private alphaBuf = new Float32Array(TOTAL_VERTS);
  private ribbonCount = 0;

  // Scratch spline arrays (Zero-GC)
  private ctrlX = new Float32Array(5);
  private ctrlY = new Float32Array(5);
  private ptX = new Float32Array(PTS_PER_RIBBON);
  private ptY = new Float32Array(PTS_PER_RIBBON);
  private br1X = new Float32Array(PTS_PER_RIBBON);
  private br1Y = new Float32Array(PTS_PER_RIBBON);
  private br2X = new Float32Array(PTS_PER_RIBBON);
  private br2Y = new Float32Array(PTS_PER_RIBBON);

  // Two 3D Rotating Plasma Arc Energy Orbs
  private orbs: [PalmOrb3D, PalmOrb3D];

  // Expanding 3D Electromagnetic Shockwave Ring for Fist -> Open Thunder Burst
  private shockMesh: THREE.Mesh;
  private shockMat: THREE.MeshBasicMaterial;

  // 220 3D Gravity Sparks & Inertial Dissolve Particles
  private sparkGeo: THREE.BufferGeometry;
  private sparkMat: THREE.ShaderMaterial;
  private sPos = new Float32Array(SPARK_POOL * 3);
  private sAlpha = new Float32Array(SPARK_POOL);
  private sSize = new Float32Array(SPARK_POOL);
  private sHue = new Float32Array(SPARK_POOL);
  private svx = new Float32Array(SPARK_POOL);
  private svy = new Float32Array(SPARK_POOL);
  private svz = new Float32Array(SPARK_POOL);
  private sLife = new Float32Array(SPARK_POOL);
  private sMax = new Float32Array(SPARK_POOL);
  private sparkCursor = 0;

  // Fist compression, Thunder Burst & Hand-Retraction Dissolve State
  private chargeLevel = 0;
  private wasFist = false;
  private burstTimer = -1;
  private burstWorldX = 0;
  private burstWorldY = 0;
  private edgeFlashAlpha = 0;
  private elapsed = 0;
  private sparkSpawnAcc = 0;
  private lastPalms: { wx: number; wy: number; vx: number; vy: number }[] = [];
  private prevPalmWorld: { wx: number; wy: number }[] = [];
  private hadHands = false;

  private audio = new LightningAudioEngine();

  constructor() {
    this.root.visible = false;
    this.sLife.fill(99);
    this.sMax.fill(1);

    // 1. Batched Volumetric Ribbon BufferGeometry
    this.ribbonGeo = new THREE.BufferGeometry();
    const indices = new Uint32Array(TOTAL_INDICES);
    let idxCursor = 0;
    for (let r = 0; r < MAX_RIBBONS; r += 1) {
      const baseV = r * VERTS_PER_RIBBON;
      for (let s = 0; s < PTS_PER_RIBBON - 1; s += 1) {
        const v0 = baseV + s * 2;
        const v1 = v0 + 1;
        const v2 = v0 + 2;
        const v3 = v0 + 3;
        indices[idxCursor++] = v0;
        indices[idxCursor++] = v1;
        indices[idxCursor++] = v2;
        indices[idxCursor++] = v2;
        indices[idxCursor++] = v1;
        indices[idxCursor++] = v3;
      }
    }
    this.ribbonGeo.setIndex(new THREE.BufferAttribute(indices, 1));
    this.ribbonGeo.setAttribute('position', new THREE.BufferAttribute(this.posBuf, 3));
    this.ribbonGeo.setAttribute('aUv', new THREE.BufferAttribute(this.uvBuf, 2));
    this.ribbonGeo.setAttribute('aAlpha', new THREE.BufferAttribute(this.alphaBuf, 1));

    this.ribbonMat = new THREE.ShaderMaterial({
      vertexShader: RIBBON_VERT,
      fragmentShader: RIBBON_FRAG,
      uniforms: { uTime: { value: 0 } },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    const ribbonMesh = new THREE.Mesh(this.ribbonGeo, this.ribbonMat);
    ribbonMesh.frustumCulled = false;
    this.root.add(ribbonMesh);

    // 2. Two 3D Rotating Palm Energy Orbs
    const sphereGeo = new THREE.SphereGeometry(0.24, 24, 18);
    const torusGeoA = new THREE.TorusGeometry(0.42, 0.018, 8, 42);
    const torusGeoB = new THREE.TorusGeometry(0.35, 0.015, 8, 42);

    const buildOrb = (): PalmOrb3D => {
      const grp = new THREE.Group();
      const coreMat = new THREE.ShaderMaterial({
        vertexShader: ORB_VERT,
        fragmentShader: ORB_FRAG,
        uniforms: {
          uTime: { value: 0 },
          uCharge: { value: 0 },
          uOpacity: { value: 1 },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      grp.add(new THREE.Mesh(sphereGeo, coreMat));

      const ringAMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.35, 1.25, 1.85),
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const ringA = new THREE.Mesh(torusGeoA, ringAMat);
      grp.add(ringA);

      const ringBMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.88, 0.45, 1.75),
        transparent: true,
        opacity: 0.78,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const ringB = new THREE.Mesh(torusGeoB, ringBMat);
      grp.add(ringB);

      grp.visible = false;
      this.root.add(grp);
      return { root: grp, coreMat, ringA, ringB, ringAMat, ringBMat };
    };

    this.orbs = [buildOrb(), buildOrb()];

    // 3. Expanding 3D Electromagnetic Shockwave Ring
    const shockGeo = new THREE.TorusGeometry(1.0, 0.036, 10, 64);
    this.shockMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0.45, 1.35, 2.0),
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.shockMesh = new THREE.Mesh(shockGeo, this.shockMat);
    this.shockMesh.visible = false;
    this.root.add(this.shockMesh);

    // 4. 220 3D Gravity Sparks & Inertial Dissolve Particles
    this.sparkGeo = new THREE.BufferGeometry();
    this.sparkGeo.setAttribute('position', new THREE.BufferAttribute(this.sPos, 3));
    this.sparkGeo.setAttribute('aAlpha', new THREE.BufferAttribute(this.sAlpha, 1));
    this.sparkGeo.setAttribute('aSize', new THREE.BufferAttribute(this.sSize, 1));
    this.sparkGeo.setAttribute('aHue', new THREE.BufferAttribute(this.sHue, 1));
    this.sparkMat = new THREE.ShaderMaterial({
      vertexShader: SPARK3D_VERT,
      fragmentShader: SPARK3D_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const sparkPoints = new THREE.Points(this.sparkGeo, this.sparkMat);
    sparkPoints.frustumCulled = false;
    this.root.add(sparkPoints);
  }

  public setSfxEnabled(enabled: boolean): void {
    this.audio.setEnabled(enabled);
  }

  public reset(): void {
    this.chargeLevel = 0;
    this.wasFist = false;
    this.burstTimer = -1;
    this.edgeFlashAlpha = 0;
    this.hadHands = false;
    this.lastPalms.length = 0;
    this.prevPalmWorld.length = 0;
  }

  public clear(): void {
    this.root.visible = false;
    this.orbs[0].root.visible = false;
    this.orbs[1].root.visible = false;
    this.shockMesh.visible = false;
  }

  private spawnSpark3D(
    wx: number,
    wy: number,
    speed: number,
    inheritVx = 0,
    inheritVy = 0,
    longDissolve = false,
  ): void {
    const i = this.sparkCursor;
    this.sparkCursor = (this.sparkCursor + 1) % SPARK_POOL;
    const a = Math.random() * Math.PI * 2;
    const spd = (0.8 + Math.random() * 2.4) * speed;
    this.sPos[i * 3] = wx + (Math.random() - 0.5) * 0.18;
    this.sPos[i * 3 + 1] = wy + (Math.random() - 0.5) * 0.18;
    this.sPos[i * 3 + 2] = (Math.random() - 0.5) * 0.3;
    this.svx[i] = Math.cos(a) * spd + inheritVx * 0.45;
    this.svy[i] = Math.sin(a) * spd + inheritVy * 0.45 + (longDissolve ? 0.9 : 0.4);
    this.svz[i] = (Math.random() - 0.5) * 1.4;
    this.sLife[i] = 0;
    this.sMax[i] = longDissolve ? 0.42 + Math.random() * 0.25 : 0.22 + Math.random() * 0.26;
    this.sSize[i] = longDissolve ? 4.5 + Math.random() * 6.0 : 3.5 + Math.random() * 5.2;
    this.sHue[i] = Math.random();
  }

  /** Computes 17 smooth Catmull-Rom Spline points displaced by continuous 2D noise. */
  private computeSplinePoints(
    outX: Float32Array,
    outY: Float32Array,
    ax: number,
    ay: number,
    bx: number,
    by: number,
    amp: number,
    seed: number,
  ): void {
    const dx = bx - ax;
    const dy = by - ay;
    const dist = Math.hypot(dx, dy) || 0.001;
    const nx = -dy / dist;
    const ny = dx / dist;

    // 5 control points with smooth continuous 2D noise displacement
    this.ctrlX[0] = ax;
    this.ctrlY[0] = ay;
    this.ctrlX[4] = bx;
    this.ctrlY[4] = by;
    for (let c = 1; c <= 3; c += 1) {
      const frac = c * 0.25;
      const env = Math.sin(frac * Math.PI);
      const disp = smoothBoltNoise(frac, this.elapsed, seed + c * 0.7) * amp * env;
      this.ctrlX[c] = ax + dx * frac + nx * disp;
      this.ctrlY[c] = ay + dy * frac + ny * disp;
    }

    // Evaluate 17 points along the 4 Catmull-Rom spans + crisp high-frequency lightning kink
    for (let i = 0; i < PTS_PER_RIBBON; i += 1) {
      const u = i / (PTS_PER_RIBBON - 1);
      const scaled = u * 4;
      const seg = Math.min(3, Math.floor(scaled));
      const localT = scaled - seg;
      const i0 = Math.max(0, seg - 1);
      const i1 = seg;
      const i2 = Math.min(4, seg + 1);
      const i3 = Math.min(4, seg + 2);

      let px = catmullRom(this.ctrlX[i0], this.ctrlX[i1], this.ctrlX[i2], this.ctrlX[i3], localT);
      let py = catmullRom(this.ctrlY[i0], this.ctrlY[i1], this.ctrlY[i2], this.ctrlY[i3], localT);

      if (i > 0 && i < PTS_PER_RIBBON - 1) {
        const kink =
          noise2D(u * 18.0 + seed * 4.3, this.elapsed * 10.5 + seed) *
          amp *
          0.26 *
          Math.sin(u * Math.PI);
        px += nx * kink;
        py += ny * kink;
      }

      outX[i] = px;
      outY[i] = py;
    }
  }

  /** Extrudes a 17-point spline into a camera-facing Volumetric Glowing Ribbon in the shared VBO. */
  private emitRibbon(
    xs: Float32Array,
    ys: Float32Array,
    baseHalfWidth: number,
    alpha: number,
    taperEnd: boolean,
  ): void {
    if (this.ribbonCount >= MAX_RIBBONS || alpha <= 0.01) return;
    const r = this.ribbonCount;
    this.ribbonCount += 1;
    const baseV = r * VERTS_PER_RIBBON;

    for (let i = 0; i < PTS_PER_RIBBON; i += 1) {
      const u = i / (PTS_PER_RIBBON - 1);
      const prev = Math.max(0, i - 1);
      const next = Math.min(PTS_PER_RIBBON - 1, i + 1);
      const tx = xs[next] - xs[prev];
      const ty = ys[next] - ys[prev];
      const tLen = Math.hypot(tx, ty) || 0.001;
      const nx = -ty / tLen;
      const ny = tx / tLen;

      // Sharp needle taper at branch tips
      const widthScale = taperEnd ? Math.pow(1 - u, 0.72) : 0.75 + 0.25 * Math.sin(u * Math.PI);
      const hw = baseHalfWidth * Math.max(0.05, widthScale);
      const segAlpha = taperEnd ? alpha * (1 - u * 0.65) : alpha;

      const vL = baseV + i * 2;
      const vR = vL + 1;

      this.posBuf[vL * 3] = xs[i] - nx * hw;
      this.posBuf[vL * 3 + 1] = ys[i] - ny * hw;
      this.posBuf[vL * 3 + 2] = 0.1;

      this.posBuf[vR * 3] = xs[i] + nx * hw;
      this.posBuf[vR * 3 + 1] = ys[i] + ny * hw;
      this.posBuf[vR * 3 + 2] = 0.1;

      this.uvBuf[vL * 2] = u;
      this.uvBuf[vL * 2 + 1] = 0;
      this.uvBuf[vR * 2] = u;
      this.uvBuf[vR * 2 + 1] = 1;

      this.alphaBuf[vL] = segAlpha;
      this.alphaBuf[vR] = segAlpha;
    }
  }

  /** Emits a primary volumetric bolt plus 2 tiers of natural tapering sub-branches. */
  private emitBoltWith2TierBranches(
    ax: number,
    ay: number,
    bx: number,
    by: number,
    amp: number,
    halfWidth: number,
    seed: number,
    branchCount: number,
    alpha: number,
    taperMain = false,
  ): void {
    this.computeSplinePoints(this.ptX, this.ptY, ax, ay, bx, by, amp, seed);
    this.emitRibbon(this.ptX, this.ptY, halfWidth, alpha, taperMain);

    const baseAngle = Math.atan2(by - ay, bx - ax);
    const span = Math.hypot(bx - ax, by - ay);

    for (let b = 0; b < branchCount; b += 1) {
      const frac = 0.18 + ((b + 0.5) / Math.max(branchCount, 1)) * 0.64;
      const attach = Math.max(2, Math.min(PTS_PER_RIBBON - 3, Math.round(frac * (PTS_PER_RIBBON - 1))));
      const sx = this.ptX[attach];
      const sy = this.ptY[attach];

      const pulse = 0.5 + 0.5 * Math.sin(this.elapsed * 9.5 + seed * 2.3 + b * 3.1);
      if (pulse < 0.22) continue;

      const side = (b + Math.floor(seed)) % 2 === 0 ? 1 : -1;
      const angle1 = baseAngle + side * (0.42 + 0.34 * Math.sin(this.elapsed * 3.8 + b * 2.1 + seed));
      const len1 = Math.max(0.45, span * (0.22 + 0.15 * pulse));
      const ex1 = sx + Math.cos(angle1) * len1;
      const ey1 = sy + Math.sin(angle1) * len1;

      this.computeSplinePoints(this.br1X, this.br1Y, sx, sy, ex1, ey1, amp * 0.38, seed + b * 3.7);
      this.emitRibbon(this.br1X, this.br1Y, halfWidth * 0.56, alpha * (0.55 + 0.45 * pulse), true);

      // Tier-2 delicate twig off Tier-1 midpoint
      const mid = PTS_PER_RIBBON >> 1;
      const ssx = this.br1X[mid];
      const ssy = this.br1Y[mid];
      const angle2 = angle1 + side * 0.52;
      const len2 = len1 * 0.48;
      this.computeSplinePoints(
        this.br2X,
        this.br2Y,
        ssx,
        ssy,
        ssx + Math.cos(angle2) * len2,
        ssy + Math.sin(angle2) * len2,
        amp * 0.2,
        seed + b * 7.1 + 1.9,
      );
      this.emitRibbon(this.br2X, this.br2Y, halfWidth * 0.32, alpha * pulse * 0.72, true);
    }
  }

  private updateOrb(
    idx: 0 | 1,
    wx: number,
    wy: number,
    scale: number,
    charge: number,
    alpha: number,
  ): void {
    const orb = this.orbs[idx];
    orb.root.visible = alpha > 0.01;
    if (!orb.root.visible) return;

    const jitter = charge > 0.05 ? charge * 0.055 : 0;
    orb.root.position.set(
      wx + (jitter ? Math.sin(this.elapsed * 78 + idx) * jitter : 0),
      wy + (jitter ? Math.cos(this.elapsed * 92 + idx) * jitter : 0),
      0.15,
    );
    orb.root.scale.setScalar(scale * (1 + charge * 0.22));
    orb.coreMat.uniforms.uTime.value = this.elapsed;
    orb.coreMat.uniforms.uCharge.value = charge;
    orb.coreMat.uniforms.uOpacity.value = alpha;

    const spinBoost = 1 + charge * 2.2;
    orb.ringA.rotation.set(this.elapsed * 3.2 * spinBoost, this.elapsed * 2.4 * spinBoost, 0.5);
    orb.ringB.rotation.set(-this.elapsed * 2.7 * spinBoost, 0.8, this.elapsed * 3.6 * spinBoost);
    orb.ringAMat.opacity = 0.85 * alpha;
    orb.ringBMat.opacity = 0.78 * alpha;
  }

  /** Updates all 3D volumetric ribbons, orbs, shockwave, and gravity sparks; returns true if anything should render. */
  public update(
    hands: HandPose2D[],
    dt: number,
    fadeAlpha: number,
    enterProgress: number,
    viewW: number,
    viewH: number,
    screenToWorld: ScreenToWorldFn,
    overlayCtx: CanvasRenderingContext2D | null,
  ): boolean {
    this.elapsed += dt;
    this.ribbonCount = 0;
    this.orbs[0].root.visible = false;
    this.orbs[1].root.visible = false;
    this.shockMesh.visible = false;

    const nowMs = performance.now();
    const reachScale = Math.min(1, 0.25 + enterProgress * 0.75);
    let audioIntensity = 0;

    // Track hand world velocity for inertial 550ms stardust dissipation on hand retraction
    if (hands.length > 0 && fadeAlpha > 0.02) {
      this.hadHands = true;
      this.lastPalms.length = 0;
      for (let i = 0; i < Math.min(hands.length, 2); i += 1) {
        const w = screenToWorld(hands[i].palmX, hands[i].palmY);
        const prev = this.prevPalmWorld[i];
        const vx = prev ? (w.x - prev.wx) / Math.max(dt, 0.008) : 0;
        const vy = prev ? (w.y - prev.wy) / Math.max(dt, 0.008) : 0;
        this.lastPalms.push({ wx: w.x, wy: w.y, vx, vy });
        this.prevPalmWorld[i] = { wx: w.x, wy: w.y };
      }
      this.prevPalmWorld.length = Math.min(hands.length, 2);
    } else if (this.hadHands) {
      // Hand retracted -> burst 180 ionized glowing sparks inheriting hand velocity inertia!
      this.hadHands = false;
      const perPalm = Math.floor(170 / Math.max(this.lastPalms.length, 1));
      for (let p = 0; p < this.lastPalms.length; p += 1) {
        const lp = this.lastPalms[p];
        for (let s = 0; s < perPalm; s += 1) {
          this.spawnSpark3D(lp.wx, lp.wy, 1.35, lp.vx, lp.vy, true);
        }
      }
      this.lastPalms.length = 0;
      this.prevPalmWorld.length = 0;
    }

    if (fadeAlpha > 0.015 && hands.length >= 2) {
      // ---- 2 HANDS OPEN: 3 thick volumetric plasma ribbons + 5 fingertip arcs ----
      this.chargeLevel = Math.max(0, this.chargeLevel - dt * 4);
      this.wasFist = false;

      const L = hands[0];
      const R = hands[1];
      const wL = screenToWorld(L.palmX, L.palmY);
      const wR = screenToWorld(R.palmX, R.palmY);
      const worldDist = Math.hypot(wR.x - wL.x, wR.y - wL.y);
      const density = Math.max(0.7, Math.min(1.55, 3.8 / Math.max(worldDist, 0.9)));
      const amp = Math.min(1.05, 0.34 + worldDist * 0.14);
      const conf = Math.min(L.confidence, R.confidence) * fadeAlpha;
      audioIntensity = Math.min(1, 0.55 + density * 0.35);

      this.updateOrb(0, wL.x, wL.y, 0.85 * Math.min(1.25, density), 0.25, conf);
      this.updateOrb(1, wR.x, wR.y, 0.85 * Math.min(1.25, density), 0.25, conf);

      this.sparkSpawnAcc += dt * 60;
      if (this.sparkSpawnAcc >= 1) {
        this.sparkSpawnAcc -= Math.floor(this.sparkSpawnAcc);
        this.spawnSpark3D(wL.x, wL.y, density * 0.95);
        this.spawnSpark3D(wR.x, wR.y, density * 0.95);
      }

      const midX = (wL.x + wR.x) * 0.5;
      const midY = (wL.y + wR.y) * 0.5;
      const branches = worldDist > 3.2 ? 4 : 3;

      for (let m = 0; m < 3; m += 1) {
        const hw = (m === 0 ? 0.092 : 0.068) * Math.min(1.22, density);
        if (reachScale < 0.98) {
          const lx = wL.x + (midX - wL.x) * reachScale;
          const ly = wL.y + (midY - wL.y) * reachScale;
          const rx = wR.x + (midX - wR.x) * reachScale;
          const ry = wR.y + (midY - wR.y) * reachScale;
          this.emitBoltWith2TierBranches(wL.x, wL.y, lx, ly, amp * 0.65, hw, m * 3.1 + 1, 2, conf, true);
          this.emitBoltWith2TierBranches(wR.x, wR.y, rx, ry, amp * 0.65, hw, m * 3.1 + 7, 2, conf, true);
        } else {
          this.emitBoltWith2TierBranches(
            wL.x,
            wL.y,
            wR.x,
            wR.y,
            amp * (0.72 + m * 0.26),
            hw,
            m * 3.7 + 0.6,
            m < 2 ? branches : 2,
            conf,
            false,
          );
        }
      }

      // 5 opposing fingertip-to-fingertip plasma arcs when hands come close
      const palmDistPx = Math.hypot(R.palmX - L.palmX, R.palmY - L.palmY);
      if (reachScale > 0.68 && palmDistPx < viewW * 0.62) {
        const proximityAlpha = Math.min(1, (viewW * 0.62 - palmDistPx) / (viewW * 0.18)) * conf * 0.72;
        for (let f = 0; f < FINGERTIP_IDS.length; f += 1) {
          const idx = FINGERTIP_IDS[f];
          const fL = screenToWorld(L.pts[idx * 2], L.pts[idx * 2 + 1]);
          const fR = screenToWorld(R.pts[idx * 2], R.pts[idx * 2 + 1]);
          this.computeSplinePoints(this.ptX, this.ptY, fL.x, fL.y, fR.x, fR.y, amp * 0.42, 14.3 + f * 2.9);
          this.emitRibbon(this.ptX, this.ptY, 0.044, proximityAlpha, false);
        }
      }
    } else if (fadeAlpha > 0.015 && hands.length === 1) {
      // ---- 1 HAND: 3D Tesla Orb + 5 Fingertip Arcs + Fist Charge -> Thunder Burst ----
      const h = hands[0];
      const wP = screenToWorld(h.palmX, h.palmY);
      const conf = h.confidence * fadeAlpha;
      const isFist = h.gesture === 'FIST';

      if (isFist) {
        this.wasFist = true;
        this.chargeLevel = Math.min(1, this.chargeLevel + dt * 1.9);
        audioIntensity = 0.48 + this.chargeLevel * 0.52;

        this.updateOrb(0, wP.x, wP.y, 0.92 + this.chargeLevel * 0.35, this.chargeLevel, conf);

        // Crackling high-voltage containment arcs orbiting compressed fist orb
        const loops = 5 + Math.floor(this.chargeLevel * 3);
        const r = 0.48 + this.chargeLevel * 0.24;
        for (let k = 0; k < loops; k += 1) {
          const a1 = this.elapsed * 8.5 + (k * Math.PI * 2) / loops;
          const a2 = a1 + 1.35;
          this.computeSplinePoints(
            this.ptX,
            this.ptY,
            wP.x + Math.cos(a1) * r,
            wP.y + Math.sin(a1) * r,
            wP.x + Math.cos(a2) * r,
            wP.y + Math.sin(a2) * r,
            0.22 + this.chargeLevel * 0.16,
            k * 4.1 + 1.7,
          );
          this.emitRibbon(this.ptX, this.ptY, 0.056, conf, false);
        }
        this.sparkSpawnAcc += dt * 42;
        if (this.sparkSpawnAcc >= 1) {
          this.sparkSpawnAcc -= Math.floor(this.sparkSpawnAcc);
          this.spawnSpark3D(wP.x, wP.y, 0.85 + this.chargeLevel * 0.75);
        }
      } else {
        if (this.wasFist && this.chargeLevel >= 0.2) {
          // Explosive Thunder Burst!
          this.burstTimer = 0;
          this.burstWorldX = wP.x;
          this.burstWorldY = wP.y;
          this.edgeFlashAlpha = 0.85 + this.chargeLevel * 0.25;
          for (let s = 0; s < 120; s += 1) {
            this.spawnSpark3D(wP.x, wP.y, 1.65 + this.chargeLevel * 0.72);
          }
          this.audio.playThunderDischarge(0.8 + this.chargeLevel * 0.5);
        }
        this.wasFist = false;
        this.chargeLevel = Math.max(0, this.chargeLevel - dt * 3.5);
        audioIntensity = 0.45;

        this.updateOrb(0, wP.x, wP.y, 0.78, 0.05, conf);
        this.sparkSpawnAcc += dt * 60;
        if (this.sparkSpawnAcc >= 1) {
          this.sparkSpawnAcc -= Math.floor(this.sparkSpawnAcc);
          this.spawnSpark3D(wP.x, wP.y, 0.85);
        }

        for (let t = 0; t < FINGERTIP_IDS.length; t += 1) {
          const idx = FINGERTIP_IDS[t];
          const wT = screenToWorld(h.pts[idx * 2], h.pts[idx * 2 + 1]);
          const fx = wP.x + (wT.x - wP.x) * reachScale;
          const fy = wP.y + (wT.y - wP.y) * reachScale;

          this.computeSplinePoints(this.ptX, this.ptY, wP.x, wP.y, fx, fy, 0.22, t * 2.7 + 1.1);
          this.emitRibbon(this.ptX, this.ptY, 0.058, conf * 0.90, false);

          if (t < FINGERTIP_IDS.length - 1 && reachScale > 0.65) {
            const nextIdx = FINGERTIP_IDS[t + 1];
            const wN = screenToWorld(h.pts[nextIdx * 2], h.pts[nextIdx * 2 + 1]);
            this.computeSplinePoints(this.ptX, this.ptY, wT.x, wT.y, wN.x, wN.y, 0.18, t * 4.5 + 5.2);
            this.emitRibbon(this.ptX, this.ptY, 0.048, conf * 0.84, false);
          }

          // Ambient fractal bolts shooting out into open space from fingertips
          const dx = wT.x - wP.x;
          const dy = wT.y - wP.y;
          const dLen = Math.hypot(dx, dy) || 0.001;
          const reach = (0.72 + 0.65 * (0.5 + 0.5 * Math.sin(this.elapsed * 8.5 + t * 2.3))) * reachScale;
          this.emitBoltWith2TierBranches(
            wT.x,
            wT.y,
            wT.x + (dx / dLen) * reach,
            wT.y + (dy / dLen) * reach,
            0.3,
            0.054,
            t * 5.1 + 9.2,
            2,
            conf * 0.85,
            true,
          );
        }
      }
    } else {
      this.chargeLevel = Math.max(0, this.chargeLevel - dt * 3);
      this.wasFist = false;
    }

    // ---- 3D Electromagnetic Thunder Burst Shockwave (0 -> 0.36s) ----
    if (this.burstTimer >= 0) {
      this.burstTimer += dt;
      const bt = this.burstTimer / 0.36;
      if (bt >= 1) {
        this.burstTimer = -1;
      } else {
        const ease = 1 - Math.pow(1 - bt, 2.5);
        const alpha = (1 - bt) * Math.max(0.25, fadeAlpha);
        const maxR = 0.5 + ease * 3.4;

        this.shockMesh.visible = true;
        this.shockMesh.position.set(this.burstWorldX, this.burstWorldY, 0.15);
        this.shockMesh.scale.setScalar(maxR * 0.72);
        this.shockMat.opacity = alpha * 0.85;

        for (let r = 0; r < 10; r += 1) {
          const a = (r * Math.PI * 2) / 10 + 0.25;
          this.emitBoltWith2TierBranches(
            this.burstWorldX,
            this.burstWorldY,
            this.burstWorldX + Math.cos(a) * maxR,
            this.burstWorldY + Math.sin(a) * maxR,
            0.44,
            0.082,
            r * 4.3 + 27.0,
            2,
            alpha,
            true,
          );
        }
      }
    }

    // ---- Subtle Electric-Blue Edge Vignette Flash on 2D Overlay ----
    if (overlayCtx && this.edgeFlashAlpha > 0.01) {
      const cx = viewW * 0.5;
      const cy = viewH * 0.5;
      const rOuter = Math.hypot(cx, cy);
      overlayCtx.save();
      overlayCtx.globalCompositeOperation = 'lighter';
      const grad = overlayCtx.createRadialGradient(cx, cy, rOuter * 0.45, cx, cy, rOuter);
      grad.addColorStop(0, 'rgba(56, 189, 248, 0)');
      grad.addColorStop(0.76, `rgba(56, 189, 248, ${this.edgeFlashAlpha * 0.22})`);
      grad.addColorStop(1, `rgba(186, 230, 253, ${this.edgeFlashAlpha * 0.46})`);
      overlayCtx.fillStyle = grad;
      overlayCtx.fillRect(0, 0, viewW, viewH);
      overlayCtx.restore();
      this.edgeFlashAlpha = Math.max(0, this.edgeFlashAlpha - dt * 4.5);
    }

    // ---- Update 220 3D Gravity Sparks & Inertial Dissolve Particles ----
    let anySpark = false;
    for (let i = 0; i < SPARK_POOL; i += 1) {
      if (this.sLife[i] >= this.sMax[i]) {
        this.sAlpha[i] = 0;
        continue;
      }
      this.sLife[i] += dt;
      if (this.sLife[i] >= this.sMax[i]) {
        this.sAlpha[i] = 0;
        continue;
      }
      anySpark = true;
      this.svy[i] -= 4.8 * dt; // 3D world gravity downward
      this.svx[i] *= 1 - Math.min(0.35, dt * 1.3);
      this.sPos[i * 3] += this.svx[i] * dt;
      this.sPos[i * 3 + 1] += this.svy[i] * dt;
      this.sPos[i * 3 + 2] += this.svz[i] * dt;
      this.sAlpha[i] = (1 - this.sLife[i] / this.sMax[i]) * Math.max(0.35, fadeAlpha);
    }

    // Upload only the active ribbon draw range (Zero-GC, 1 draw call!)
    this.ribbonGeo.setDrawRange(0, this.ribbonCount * TRIS_PER_RIBBON * 3);
    if (this.ribbonCount > 0) {
      (this.ribbonGeo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
      (this.ribbonGeo.attributes.aUv as THREE.BufferAttribute).needsUpdate = true;
      (this.ribbonGeo.attributes.aAlpha as THREE.BufferAttribute).needsUpdate = true;
    }
    this.ribbonMat.uniforms.uTime.value = this.elapsed;

    if (anySpark) {
      (this.sparkGeo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
      (this.sparkGeo.attributes.aAlpha as THREE.BufferAttribute).needsUpdate = true;
      (this.sparkGeo.attributes.aSize as THREE.BufferAttribute).needsUpdate = true;
      (this.sparkGeo.attributes.aHue as THREE.BufferAttribute).needsUpdate = true;
    }

    if (audioIntensity > 0.15) {
      this.audio.tickCrackle(audioIntensity, nowMs);
    }

    const active =
      this.ribbonCount > 0 ||
      this.orbs[0].root.visible ||
      this.orbs[1].root.visible ||
      this.shockMesh.visible ||
      anySpark;
    this.root.visible = active;
    return active;
  }

  public dispose(): void {
    this.audio.dispose();
  }
}
