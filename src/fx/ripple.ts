import type { HandPose2D } from './types';

const SIM_W = 384;
const SIM_H = 216;
const MAX_IMPULSES = 32;
const MAX_RINGS = 2;
const FIXED_STEP_SEC = 1 / 120; // 120Hz fixed physics timestep
const DAMPING_120HZ = 0.9918; // Smooth 16-bit half-float wave persistence (~0.984^0.5)
const FIST_HOLD_MIN_SEC = 0.12;
const SHOCKWAVE_COOLDOWN_SEC = 0.7;

const SIM_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aPos;
out vec2 vUv;

void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

const SIM_FRAG = `#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uPrevSim;
uniform vec2 uTexel;
uniform float uDamping;
uniform float uBias;
uniform float uInvScale;
uniform float uScale;
uniform int uImpulseCount;
uniform vec4 uImpulses[${MAX_IMPULSES}]; // xy = UV, z = radius, w = strength
uniform vec4 uRings[${MAX_RINGS}];       // xy = UV, z = ringRadius, w = amplitude

out vec4 fragColor;

void main() {
  vec2 c = (texture(uPrevSim, vUv).rg - uBias) * uInvScale;
  float currH = c.r;
  float prevH = c.g;

  float hL = (texture(uPrevSim, clamp(vUv + vec2(-uTexel.x, 0.0), 0.002, 0.998)).r - uBias) * uInvScale;
  float hR = (texture(uPrevSim, clamp(vUv + vec2( uTexel.x, 0.0), 0.002, 0.998)).r - uBias) * uInvScale;
  float hU = (texture(uPrevSim, clamp(vUv + vec2(0.0, -uTexel.y), 0.002, 0.998)).r - uBias) * uInvScale;
  float hD = (texture(uPrevSim, clamp(vUv + vec2(0.0,  uTexel.y), 0.002, 0.998)).r - uBias) * uInvScale;

  // Isotropic 8-neighbor weighted Laplacian eliminates grid-axis diamond artifacts
  float hTL = (texture(uPrevSim, clamp(vUv + vec2(-uTexel.x, -uTexel.y), 0.002, 0.998)).r - uBias) * uInvScale;
  float hTR = (texture(uPrevSim, clamp(vUv + vec2( uTexel.x, -uTexel.y), 0.002, 0.998)).r - uBias) * uInvScale;
  float hBL = (texture(uPrevSim, clamp(vUv + vec2(-uTexel.x,  uTexel.y), 0.002, 0.998)).r - uBias) * uInvScale;
  float hBR = (texture(uPrevSim, clamp(vUv + vec2( uTexel.x,  uTexel.y), 0.002, 0.998)).r - uBias) * uInvScale;

  float avg = (hL + hR + hU + hD) * 0.20 + (hTL + hTR + hBL + hBR) * 0.05;
  float nextH = (2.0 * avg - prevH) * uDamping;

  // Continuous Catmull-Rom sub-step splats along palm & index fingertip trajectories
  vec2 aspect = vec2(16.0 / 9.0, 1.0);
  for (int i = 0; i < ${MAX_IMPULSES}; i++) {
    if (i >= uImpulseCount) break;
    vec4 imp = uImpulses[i];
    vec2 diff = (vUv - imp.xy) * aspect;
    float d2 = dot(diff, diff);
    float r2 = max(imp.z * imp.z, 0.0001);
    if (d2 < r2 * 6.5) {
      nextH += exp(-d2 / r2) * imp.w;
    }
  }

  // Fist -> Open expanding Shockwave Rings (positive crest + trailing trough)
  for (int k = 0; k < ${MAX_RINGS}; k++) {
    vec4 ring = uRings[k];
    if (ring.w > 0.002) {
      float d = length((vUv - ring.xy) * aspect);
      float crest = exp(-pow((d - ring.z) / 0.034, 2.0));
      float trough = exp(-pow((d - max(0.0, ring.z - 0.046)) / 0.028, 2.0));
      nextH += (crest * 1.0 - trough * 0.52) * ring.w;
    }
  }

  // Soft absorbing border prevents artificial box reflections
  float edgeX = smoothstep(0.0, 0.038, vUv.x) * smoothstep(1.0, 0.962, vUv.x);
  float edgeY = smoothstep(0.0, 0.038, vUv.y) * smoothstep(1.0, 0.962, vUv.y);
  nextH *= mix(0.92, 1.0, edgeX * edgeY);

  nextH = clamp(nextH, -1.8, 1.8);
  fragColor = vec4(nextH * uScale + uBias, currH * uScale + uBias, 0.0, 1.0);
}
`;

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type);
  if (!s) throw new Error('Failed to create ripple shader.');
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(s) || 'Ripple shader compile error';
    gl.deleteShader(s);
    throw new Error(info);
  }
  return s;
}

function linkProgram(gl: WebGL2RenderingContext, vert: string, frag: string): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vert);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, frag);
  const p = gl.createProgram();
  if (!p) throw new Error('Failed to link ripple program.');
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(p) || 'Ripple program link error';
    gl.deleteProgram(p);
    throw new Error(info);
  }
  return p;
}

function catmullRom1D(p0: number, p1: number, p2: number, p3: number, t: number): number {
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

interface TrailHistory4 {
  count: number;
  u0: number;
  v0: number;
  u1: number;
  v1: number;
  u2: number;
  v2: number;
  u3: number;
  v3: number;
}

function createTrailHistory(): TrailHistory4 {
  return { count: 0, u0: 0.5, v0: 0.5, u1: 0.5, v1: 0.5, u2: 0.5, v2: 0.5, u3: 0.5, v3: 0.5 };
}

function pushTrailPoint(h: TrailHistory4, u: number, v: number): void {
  if (h.count === 0) {
    h.u0 = h.u1 = h.u2 = h.u3 = u;
    h.v0 = h.v1 = h.v2 = h.v3 = v;
    h.count = 1;
    return;
  }
  h.u0 = h.u1;
  h.v0 = h.v1;
  h.u1 = h.u2;
  h.v1 = h.v2;
  h.u2 = h.u3;
  h.v2 = h.v3;
  h.u3 = u;
  h.v3 = v;
  if (h.count < 4) h.count += 1;
}

interface HandWaveState {
  palm: TrailHistory4;
  tip: TrailHistory4;
  fistHoldSec: number;
  shockCooldownSec: number;
}

interface ActiveRing {
  active: boolean;
  u: number;
  v: number;
  age: number;
  duration: number;
}

export interface RippleSimMeta {
  isHalfFloat: boolean;
  fboComplete: boolean;
  simWidth: number;
  simHeight: number;
}

/**
 * True 16-bit Half-Float WebGL2 Liquid Glass Wave Simulator (384x216, 120Hz Fixed Timestep):
 * - RGBA16F ping-pong FBOs (EXT_color_buffer_float / EXT_color_buffer_half_float + verified FBO completeness)
 * - Catmull-Rom 4–6 sub-step spline splatting for unbroken liquid wakes at 120 FPS
 * - Isotropic 8-neighbor wave PDE + Fist->Open Shockwave Ring bursts
 */
export class RippleWaveSim {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private vbo: WebGLBuffer;
  private texs: [WebGLTexture, WebGLTexture];
  private fbos: [WebGLFramebuffer, WebGLFramebuffer];
  private readIdx: 0 | 1 = 0;
  private useFloat: boolean;
  private fboComplete = false;
  private stepAcc = 0;

  private locs: {
    uPrevSim: WebGLUniformLocation | null;
    uTexel: WebGLUniformLocation | null;
    uDamping: WebGLUniformLocation | null;
    uBias: WebGLUniformLocation | null;
    uInvScale: WebGLUniformLocation | null;
    uScale: WebGLUniformLocation | null;
    uImpulseCount: WebGLUniformLocation | null;
    uImpulses: WebGLUniformLocation | null;
    uRings: WebGLUniformLocation | null;
  };

  private impulseBuf = new Float32Array(MAX_IMPULSES * 4);
  private ringBuf = new Float32Array(MAX_RINGS * 4);
  private handStates: [HandWaveState, HandWaveState] = [
    { palm: createTrailHistory(), tip: createTrailHistory(), fistHoldSec: 0, shockCooldownSec: 0 },
    { palm: createTrailHistory(), tip: createTrailHistory(), fistHoldSec: 0, shockCooldownSec: 0 },
  ];
  private rings: [ActiveRing, ActiveRing] = [
    { active: false, u: 0.5, v: 0.5, age: 0, duration: 0.50 },
    { active: false, u: 0.5, v: 0.5, age: 0, duration: 0.50 },
  ];
  private nextRingSlot: 0 | 1 = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    const hasFloatExt =
      Boolean(gl.getExtension('EXT_color_buffer_float')) ||
      Boolean(gl.getExtension('EXT_color_buffer_half_float'));
    // Enable linear filtering on half-float if extension exposed (core in WebGL2 on desktop GPUs)
    gl.getExtension('OES_texture_half_float_linear');
    gl.getExtension('OES_texture_float_linear');

    this.program = linkProgram(gl, SIM_VERT, SIM_FRAG);
    this.locs = {
      uPrevSim: gl.getUniformLocation(this.program, 'uPrevSim'),
      uTexel: gl.getUniformLocation(this.program, 'uTexel'),
      uDamping: gl.getUniformLocation(this.program, 'uDamping'),
      uBias: gl.getUniformLocation(this.program, 'uBias'),
      uInvScale: gl.getUniformLocation(this.program, 'uInvScale'),
      uScale: gl.getUniformLocation(this.program, 'uScale'),
      uImpulseCount: gl.getUniformLocation(this.program, 'uImpulseCount'),
      uImpulses: gl.getUniformLocation(this.program, 'uImpulses[0]'),
      uRings: gl.getUniformLocation(this.program, 'uRings[0]'),
    };

    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    if (!vao || !vbo) throw new Error('Failed to create ripple buffers.');
    this.vao = vao;
    this.vbo = vbo;

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Attempt RGBA16F first and verify FRAMEBUFFER_COMPLETE; fall back cleanly to RGBA8 if unsupported
    this.useFloat = hasFloatExt;
    const pair = this.allocateFboPair(this.useFloat);
    if (!pair.complete && this.useFloat) {
      pair.texs.forEach((t) => gl.deleteTexture(t));
      pair.fbos.forEach((f) => gl.deleteFramebuffer(f));
      this.useFloat = false;
      const fallback = this.allocateFboPair(false);
      this.texs = fallback.texs;
      this.fbos = fallback.fbos;
      this.fboComplete = fallback.complete;
    } else {
      this.texs = pair.texs;
      this.fbos = pair.fbos;
      this.fboComplete = pair.complete;
    }
    this.clear();
  }

  private allocateFboPair(useHalfFloat: boolean): {
    texs: [WebGLTexture, WebGLTexture];
    fbos: [WebGLFramebuffer, WebGLFramebuffer];
    complete: boolean;
  } {
    const gl = this.gl;
    const makeTex = () => {
      const tex = gl.createTexture();
      if (!tex) throw new Error('Failed to create ripple texture.');
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      if (useHalfFloat) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, SIM_W, SIM_H, 0, gl.RGBA, gl.HALF_FLOAT, null);
      } else {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, SIM_W, SIM_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      }
      gl.bindTexture(gl.TEXTURE_2D, null);
      return tex;
    };

    const makeFbo = (tex: WebGLTexture) => {
      const fbo = gl.createFramebuffer();
      if (!fbo) throw new Error('Failed to create ripple framebuffer.');
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { fbo, ok: status === gl.FRAMEBUFFER_COMPLETE };
    };

    const t0 = makeTex();
    const t1 = makeTex();
    const f0 = makeFbo(t0);
    const f1 = makeFbo(t1);
    return { texs: [t0, t1], fbos: [f0.fbo, f1.fbo], complete: f0.ok && f1.ok };
  }

  public getMeta(): RippleSimMeta {
    return {
      isHalfFloat: this.useFloat,
      fboComplete: this.fboComplete,
      simWidth: SIM_W,
      simHeight: SIM_H,
    };
  }

  public clear(): void {
    const gl = this.gl;
    const clearVal = this.useFloat ? 0.0 : 0.5;
    for (let i = 0; i < 2; i += 1) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[i]);
      gl.viewport(0, 0, SIM_W, SIM_H);
      gl.clearColor(clearVal, clearVal, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.stepAcc = 0;
    this.handStates[0].palm.count = 0;
    this.handStates[0].tip.count = 0;
    this.handStates[0].fistHoldSec = 0;
    this.handStates[1].palm.count = 0;
    this.handStates[1].tip.count = 0;
    this.handStates[1].fistHoldSec = 0;
    this.rings[0].active = false;
    this.rings[1].active = false;
  }

  public getHeightTexture(): WebGLTexture {
    return this.texs[this.readIdx];
  }

  public getBiasAndInvScale(): [number, number] {
    return this.useFloat ? [0.0, 1.0] : [0.5, 2.0];
  }

  public getTexelSize(): [number, number] {
    return [1.0 / SIM_W, 1.0 / SIM_H];
  }

  /** Fixed-timestep 120Hz GPU wave solver with Catmull-Rom 4–6 sub-step spline splatting. */
  public step(hands: HandPose2D[], dt: number, viewW: number, viewH: number): void {
    const safeDt = Math.min(Math.max(dt, 0.002), 0.05);
    this.stepAcc = Math.min(this.stepAcc + safeDt, FIXED_STEP_SEC * 2.5);

    const invW = 1 / Math.max(viewW, 1);
    const invH = 1 / Math.max(viewH, 1);
    let impulseCount = 0;

    const pushImpulse = (u: number, v: number, radius: number, strength: number) => {
      if (impulseCount >= MAX_IMPULSES) return;
      const o = impulseCount * 4;
      this.impulseBuf[o] = u;
      this.impulseBuf[o + 1] = v;
      this.impulseBuf[o + 2] = radius;
      this.impulseBuf[o + 3] = strength;
      impulseCount += 1;
    };

    const splatCatmullRom = (
      hist: TrailHistory4,
      u: number,
      v: number,
      baseRadius: number,
      maxAmp: number,
      idleAmp: number,
    ) => {
      const hadPrev = hist.count > 0;
      const prevU = hist.u3;
      const prevV = hist.v3;
      pushTrailPoint(hist, u, v);

      if (!hadPrev) {
        pushImpulse(u, v, baseRadius * 1.15, 0.16);
        return;
      }

      const dist = Math.hypot(u - prevU, v - prevV);
      const speed = dist / safeDt;

      if (dist > 0.0012) {
        // 4-6 smooth Catmull-Rom sub-steps along segment -> unbroken ribbon even at high hand speed
        const subSteps = Math.min(6, Math.max(4, Math.ceil(dist / 0.009)));
        const amp = Math.min(maxAmp, 0.045 + speed * 0.095) / Math.sqrt(subSteps);
        for (let s = 1; s <= subSteps; s += 1) {
          const f = s / subSteps;
          const su = catmullRom1D(hist.u0, hist.u1, hist.u2, hist.u3, f);
          const sv = catmullRom1D(hist.v0, hist.v1, hist.v2, hist.v3, f);
          pushImpulse(
            Math.min(0.98, Math.max(0.02, su)),
            Math.min(0.98, Math.max(0.02, sv)),
            baseRadius,
            amp,
          );
        }
      } else if (idleAmp > 0) {
        pushImpulse(u, v, baseRadius * 0.9, idleAmp * (safeDt * 120));
      }
    };

    const seenSlot = [false, false];

    for (let i = 0; i < hands.length && i < 2; i += 1) {
      const h = hands[i];
      const slot = h.id === 1 ? 1 : 0;
      seenSlot[slot] = true;
      const st = this.handStates[slot];

      if (st.shockCooldownSec > 0) {
        st.shockCooldownSec = Math.max(0, st.shockCooldownSec - safeDt);
      }

      const palmU = Math.min(0.98, Math.max(0.02, h.palmX * invW));
      const palmV = Math.min(0.98, Math.max(0.02, 1 - h.palmY * invH));
      const tipU = Math.min(0.98, Math.max(0.02, h.pts[16] * invW));
      const tipV = Math.min(0.98, Math.max(0.02, 1 - h.pts[17] * invH));

      const extended = countExtendedFingers2D(h);

      // Fist (<=1 finger >=120ms) -> Open (>=4 fingers) Shockwave Ring Burst
      if (extended <= 1 || h.gesture === 'FIST') {
        st.fistHoldSec += safeDt;
      } else {
        if (extended >= 4 && st.fistHoldSec >= FIST_HOLD_MIN_SEC && st.shockCooldownSec <= 0) {
          const rSlot = this.rings[this.nextRingSlot];
          rSlot.active = true;
          rSlot.u = palmU;
          rSlot.v = palmV;
          rSlot.age = 0;
          rSlot.duration = 0.50;
          this.nextRingSlot = this.nextRingSlot === 0 ? 1 : 0;
          st.shockCooldownSec = SHOCKWAVE_COOLDOWN_SEC;
          pushImpulse(palmU, palmV, 0.082, 0.68);
        }
        st.fistHoldSec = 0;
      }

      splatCatmullRom(st.palm, palmU, palmV, 0.044, 0.28, 0.014);
      splatCatmullRom(st.tip, tipU, tipV, 0.028, 0.24, 0.0);
    }

    for (let slot = 0; slot < 2; slot += 1) {
      if (!seenSlot[slot]) {
        this.handStates[slot].palm.count = 0;
        this.handStates[slot].tip.count = 0;
        this.handStates[slot].fistHoldSec = 0;
        if (this.handStates[slot].shockCooldownSec > 0) {
          this.handStates[slot].shockCooldownSec = Math.max(
            0,
            this.handStates[slot].shockCooldownSec - safeDt,
          );
        }
      }
    }

    // Run 1..2 Fixed-Timestep 120Hz Wave PDE Passes
    const gl = this.gl;
    let subPasses = 0;
    while (this.stepAcc >= FIXED_STEP_SEC && subPasses < 2) {
      this.stepAcc -= FIXED_STEP_SEC;

      for (let k = 0; k < MAX_RINGS; k += 1) {
        const r = this.rings[k];
        const o = k * 4;
        if (r.active) {
          r.age += FIXED_STEP_SEC;
          const t = r.age / r.duration;
          if (t >= 1) {
            r.active = false;
            this.ringBuf[o + 3] = 0;
          } else {
            const ease = 1 - Math.pow(1 - t, 1.85);
            this.ringBuf[o] = r.u;
            this.ringBuf[o + 1] = r.v;
            this.ringBuf[o + 2] = 0.04 + ease * 1.18;
            this.ringBuf[o + 3] = (1 - t) * (1 - t) * 0.24;
          }
        } else {
          this.ringBuf[o + 3] = 0;
        }
      }

      const writeIdx: 0 | 1 = this.readIdx === 0 ? 1 : 0;

      gl.useProgram(this.program);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[writeIdx]);
      gl.viewport(0, 0, SIM_W, SIM_H);
      gl.disable(gl.BLEND);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texs[this.readIdx]);
      gl.uniform1i(this.locs.uPrevSim, 0);
      gl.uniform2f(this.locs.uTexel, 1.0 / SIM_W, 1.0 / SIM_H);
      gl.uniform1f(this.locs.uDamping, DAMPING_120HZ);
      gl.uniform1f(this.locs.uBias, this.useFloat ? 0.0 : 0.5);
      gl.uniform1f(this.locs.uInvScale, this.useFloat ? 1.0 : 2.0);
      gl.uniform1f(this.locs.uScale, this.useFloat ? 1.0 : 0.5);
      // Inject hand impulses on first substep only so fast frames never double-inject
      gl.uniform1i(this.locs.uImpulseCount, subPasses === 0 ? impulseCount : 0);
      if (subPasses === 0 && impulseCount > 0) {
        gl.uniform4fv(this.locs.uImpulses, this.impulseBuf);
      }
      gl.uniform4fv(this.locs.uRings, this.ringBuf);

      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      this.readIdx = writeIdx;
      subPasses += 1;
    }
  }

  public dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.vbo);
    this.texs.forEach((t) => gl.deleteTexture(t));
    this.fbos.forEach((f) => gl.deleteFramebuffer(f));
  }
}
