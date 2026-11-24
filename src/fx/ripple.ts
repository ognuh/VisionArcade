import type { HandPose2D } from './types';

const SIM_W = 320;
const SIM_H = 180;
const MAX_IMPULSES = 24;
const MAX_RINGS = 2;
const DAMPING = 0.984;
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

  float avg = 0.25 * (hL + hR + hU + hD);
  float nextH = (2.0 * avg - prevH) * uDamping;

  // Aspect-corrected hand motion impulses (palm + index fingertip continuous wakes)
  vec2 aspect = vec2(16.0 / 9.0, 1.0);
  for (int i = 0; i < ${MAX_IMPULSES}; i++) {
    if (i >= uImpulseCount) break;
    vec4 imp = uImpulses[i];
    vec2 diff = (vUv - imp.xy) * aspect;
    float d2 = dot(diff, diff);
    float r2 = max(imp.z * imp.z, 0.0001);
    if (d2 < r2 * 6.0) {
      nextH += exp(-d2 / r2) * imp.w;
    }
  }

  // Fist -> Open expanding Shockwave Rings (positive crest + trailing trough)
  for (int k = 0; k < ${MAX_RINGS}; k++) {
    vec4 ring = uRings[k];
    if (ring.w > 0.002) {
      float d = length((vUv - ring.xy) * aspect);
      float crest = exp(-pow((d - ring.z) / 0.036, 2.0));
      float trough = exp(-pow((d - max(0.0, ring.z - 0.048)) / 0.030, 2.0));
      nextH += (crest * 1.0 - trough * 0.52) * ring.w;
    }
  }

  // Soft boundary absorption prevents harsh wall reflections
  float edgeX = smoothstep(0.0, 0.035, vUv.x) * smoothstep(1.0, 0.965, vUv.x);
  float edgeY = smoothstep(0.0, 0.035, vUv.y) * smoothstep(1.0, 0.965, vUv.y);
  nextH *= mix(0.88, 1.0, edgeX * edgeY);

  nextH = clamp(nextH, -1.6, 1.6);
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
  // Thumb vs pinky MCP
  const dThumb = Math.hypot(h.pts[8] - h.pts[34], h.pts[9] - h.pts[35]);
  const dBase = Math.hypot(h.pts[4] - h.pts[34], h.pts[5] - h.pts[35]);
  if (dThumb > dBase * 1.12) count += 1;
  return count;
}

interface HandWaveState {
  hasPrev: boolean;
  prevPalmU: number;
  prevPalmV: number;
  prevTipU: number;
  prevTipV: number;
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

/**
 * WebGL2 Ping-Pong 2D Wave Heightfield Simulator (320x180, 60 FPS):
 * - Smooth continuous hand/finger wake impulses proportional to velocity
 * - Fist (<=1 finger >=120ms) -> Open (>=4 fingers) expanding Shockwave Ring (700ms cooldown)
 * - Outputs heightfield texture consumed directly by the Liquid Glass refraction pass.
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
    {
      hasPrev: false,
      prevPalmU: 0.5,
      prevPalmV: 0.5,
      prevTipU: 0.5,
      prevTipV: 0.5,
      fistHoldSec: 0,
      shockCooldownSec: 0,
    },
    {
      hasPrev: false,
      prevPalmU: 0.5,
      prevPalmV: 0.5,
      prevTipU: 0.5,
      prevTipV: 0.5,
      fistHoldSec: 0,
      shockCooldownSec: 0,
    },
  ];
  private rings: [ActiveRing, ActiveRing] = [
    { active: false, u: 0.5, v: 0.5, age: 0, duration: 0.48 },
    { active: false, u: 0.5, v: 0.5, age: 0, duration: 0.48 },
  ];
  private nextRingSlot: 0 | 1 = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.useFloat = Boolean(gl.getExtension('EXT_color_buffer_float'));

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

    const t0 = this.createSimTexture();
    const t1 = this.createSimTexture();
    this.texs = [t0, t1];
    this.fbos = [this.createSimFbo(t0), this.createSimFbo(t1)];
    this.clear();
  }

  private createSimTexture(): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture();
    if (!tex) throw new Error('Failed to create ripple texture.');
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (this.useFloat) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, SIM_W, SIM_H, 0, gl.RGBA, gl.HALF_FLOAT, null);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, SIM_W, SIM_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  private createSimFbo(tex: WebGLTexture): WebGLFramebuffer {
    const gl = this.gl;
    const fbo = gl.createFramebuffer();
    if (!fbo) throw new Error('Failed to create ripple framebuffer.');
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
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
    this.handStates[0].hasPrev = false;
    this.handStates[0].fistHoldSec = 0;
    this.handStates[1].hasPrev = false;
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

  /** Steps the 320x180 ping-pong wave equation and injects continuous hand wakes + shockwave bursts. */
  public step(hands: HandPose2D[], dt: number, viewW: number, viewH: number): void {
    const safeDt = Math.min(Math.max(dt, 0.004), 0.05);
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

    const seenSlot = [false, false];

    for (let i = 0; i < hands.length && i < 2; i += 1) {
      const h = hands[i];
      const slot = h.id === 1 ? 1 : 0;
      seenSlot[slot] = true;
      const st = this.handStates[slot];

      if (st.shockCooldownSec > 0) {
        st.shockCooldownSec = Math.max(0, st.shockCooldownSec - safeDt);
      }

      // Screen UV (v=0 top -> FBO v=1 top, so flip Y for standard GL texture UV)
      const palmU = Math.min(0.98, Math.max(0.02, h.palmX * invW));
      const palmV = Math.min(0.98, Math.max(0.02, 1 - h.palmY * invH));
      const tipU = Math.min(0.98, Math.max(0.02, h.pts[16] * invW));
      const tipV = Math.min(0.98, Math.max(0.02, 1 - h.pts[17] * invH));

      const extended = countExtendedFingers2D(h);

      // Fist (<=1 finger >=120ms) -> Open (>=4 fingers) Shockwave Burst!
      if (extended <= 1 || h.gesture === 'FIST') {
        st.fistHoldSec += safeDt;
      } else {
        if (extended >= 4 && st.fistHoldSec >= FIST_HOLD_MIN_SEC && st.shockCooldownSec <= 0) {
          const rSlot = this.rings[this.nextRingSlot];
          rSlot.active = true;
          rSlot.u = palmU;
          rSlot.v = palmV;
          rSlot.age = 0;
          rSlot.duration = 0.48;
          this.nextRingSlot = this.nextRingSlot === 0 ? 1 : 0;
          st.shockCooldownSec = SHOCKWAVE_COOLDOWN_SEC;
          // Strong central plunge to seed inner concentric ripples
          pushImpulse(palmU, palmV, 0.085, 0.95);
        }
        st.fistHoldSec = 0;
      }

      if (st.hasPrev) {
        // Continuous interpolated wake along palm trajectory
        const pdx = palmU - st.prevPalmU;
        const pdy = palmV - st.prevPalmV;
        const pDist = Math.hypot(pdx, pdy);
        const pSpeed = pDist / safeDt;

        if (pDist > 0.0025) {
          const steps = Math.min(5, Math.max(1, Math.ceil(pDist / 0.018)));
          const amp = Math.min(0.42, 0.09 + pSpeed * 0.16);
          for (let s = 1; s <= steps; s += 1) {
            const f = s / steps;
            pushImpulse(st.prevPalmU + pdx * f, st.prevPalmV + pdy * f, 0.048, amp / Math.sqrt(steps));
          }
        } else {
          // Subtle breathing ripple even when holding hand gently in water
          pushImpulse(palmU, palmV, 0.042, 0.032);
        }

        // Continuous interpolated wake along index fingertip
        const tdx = tipU - st.prevTipU;
        const tdy = tipV - st.prevTipV;
        const tDist = Math.hypot(tdx, tdy);
        const tSpeed = tDist / safeDt;

        if (tDist > 0.0025) {
          const steps = Math.min(5, Math.max(1, Math.ceil(tDist / 0.014)));
          const amp = Math.min(0.38, 0.08 + tSpeed * 0.15);
          for (let s = 1; s <= steps; s += 1) {
            const f = s / steps;
            pushImpulse(st.prevTipU + tdx * f, st.prevTipV + tdy * f, 0.030, amp / Math.sqrt(steps));
          }
        }
      } else {
        // Gentle splash when hand first enters
        pushImpulse(palmU, palmV, 0.055, 0.24);
      }

      st.hasPrev = true;
      st.prevPalmU = palmU;
      st.prevPalmV = palmV;
      st.prevTipU = tipU;
      st.prevTipV = tipV;
    }

    for (let slot = 0; slot < 2; slot += 1) {
      if (!seenSlot[slot]) {
        this.handStates[slot].hasPrev = false;
        this.handStates[slot].fistHoldSec = 0;
        if (this.handStates[slot].shockCooldownSec > 0) {
          this.handStates[slot].shockCooldownSec = Math.max(
            0,
            this.handStates[slot].shockCooldownSec - safeDt,
          );
        }
      }
    }

    // Update active expanding Shockwave Rings
    for (let k = 0; k < MAX_RINGS; k += 1) {
      const r = this.rings[k];
      const o = k * 4;
      if (r.active) {
        r.age += safeDt;
        const t = r.age / r.duration;
        if (t >= 1) {
          r.active = false;
          this.ringBuf[o + 3] = 0;
        } else {
          const ease = 1 - Math.pow(1 - t, 1.8);
          this.ringBuf[o] = r.u;
          this.ringBuf[o + 1] = r.v;
          this.ringBuf[o + 2] = 0.04 + ease * 1.18;
          this.ringBuf[o + 3] = (1 - t) * (1 - t) * 0.42;
        }
      } else {
        this.ringBuf[o + 3] = 0;
      }
    }

    // Run 1 GPU ping-pong wave step
    const gl = this.gl;
    const writeIdx: 0 | 1 = this.readIdx === 0 ? 1 : 0;

    gl.useProgram(this.program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[writeIdx]);
    gl.viewport(0, 0, SIM_W, SIM_H);
    gl.disable(gl.BLEND);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texs[this.readIdx]);
    gl.uniform1i(this.locs.uPrevSim, 0);
    gl.uniform2f(this.locs.uTexel, 1.0 / SIM_W, 1.0 / SIM_H);
    gl.uniform1f(this.locs.uDamping, DAMPING);
    gl.uniform1f(this.locs.uBias, this.useFloat ? 0.0 : 0.5);
    gl.uniform1f(this.locs.uInvScale, this.useFloat ? 1.0 : 2.0);
    gl.uniform1f(this.locs.uScale, this.useFloat ? 1.0 : 0.5);
    gl.uniform1i(this.locs.uImpulseCount, impulseCount);
    gl.uniform4fv(this.locs.uImpulses, this.impulseBuf);
    gl.uniform4fv(this.locs.uRings, this.ringBuf);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);

    // Unbind cleanly to prevent D3D11/ANGLE feedback hazards
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.readIdx = writeIdx;
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
