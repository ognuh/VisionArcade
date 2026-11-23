import { GalaxyFx } from '../fx/galaxy';
import { LightningFx } from '../fx/lightning';
import { MysticFx } from '../fx/mystic';
import {
  buildOpticalPolygons,
  renderPrismChromaticEdges,
  renderTrailOutline,
} from '../fx/prism';
import type { HandPose2D, Point2D, RenderPolygon } from '../fx/types';
import type {
  AirDrawColorId,
  CycleModeId,
  GestureType,
  Landmark3D,
  PrismSettings,
} from '../types/vision';
import { drawCyberneticHandHud } from './cyberHud';

export type { Point2D };

interface TrailFrame {
  outlines: { pts: Point2D[]; color: string }[];
}

interface AirStroke {
  points: Point2D[];
  color: string;
}

interface NoteSparkBeam {
  sx: number;
  sy: number;
  tx: number;
  ty: number;
  cx: number;
  cy: number;
  age: number;
  duration: number;
}

interface SwipeSwoosh {
  dir: -1 | 1;
  age: number;
  duration: number;
}

export interface FrameOutput {
  activeMode: CycleModeId;
  indexTipPx: Point2D | null;
  /** Normalized screen-space [0..1] palm position for 3D Hologram Text parallax. */
  palmScreenNorm: Point2D | null;
  isLeadFist: boolean;
}

export const AIR_DRAW_COLORS: Record<AirDrawColorId, { hex: string; label: string }> = {
  CYAN: { hex: '#22d3ee', label: 'Cyan' },
  MAGENTA: { hex: '#f43f97', label: 'Magenta' },
  GREEN: { hex: '#4ade80', label: 'Neon Green' },
  GOLD: { hex: '#facc15', label: 'Gold' },
};

const GRACE_FRAMES = 5;
const TRAIL_MAX_FRAMES = 14;

const VERT_SRC = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aPosNdc;
layout(location = 1) in vec2 aScreenUv;

out vec2 vScreenUv;

void main() {
  vScreenUv = aScreenUv;
  gl_Position = vec4(aPosNdc, 0.0, 1.0);
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;

in vec2 vScreenUv;

uniform sampler2D uVideo;
uniform vec2 uCoverScale;
uniform vec2 uVideoTexel;
uniform float uMirrored;
uniform float uEnhance;
uniform int uShaderType; // 0 = GLASS, 1 = THERMAL, 2 = PRISM_BAND, 4 = WEBCAM_HD_PIPELINE
uniform vec4 uTint;

out vec4 fragColor;

vec2 screenToVideoUv(vec2 s) {
  vec2 centered = (s - 0.5) * uCoverScale + 0.5;
  float ux = (uMirrored > 0.5) ? (1.0 - centered.x) : centered.x;
  return vec2(ux, centered.y);
}

float lumaOf(vec3 c) {
  return dot(c, vec3(0.299, 0.587, 0.114));
}

vec3 thermalFalseColor(float t) {
  // High-vibrance vivid false-color ramp: Deep Cobalt -> Electric Violet -> Hot Pink -> Molten Orange -> Blazing Yellow
  vec3 c0 = vec3(0.02, 0.08, 0.42);
  vec3 c1 = vec3(0.45, 0.08, 0.92);
  vec3 c2 = vec3(1.00, 0.08, 0.64);
  vec3 c3 = vec3(1.00, 0.46, 0.02);
  vec3 c4 = vec3(1.00, 0.99, 0.22);
  if (t < 0.25) return mix(c0, c1, t / 0.25);
  if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
  if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
  return mix(c3, c4, clamp((t - 0.75) / 0.25, 0.0, 1.0));
}

vec3 enhanceWebcamHD(vec2 uv) {
  vec3 c = texture(uVideo, uv).rgb;
  if (uEnhance <= 0.005) return c;

  vec2 t = uVideoTexel;
  vec3 n  = texture(uVideo, clamp(uv + vec2( 0.0, -t.y), 0.002, 0.998)).rgb;
  vec3 s  = texture(uVideo, clamp(uv + vec2( 0.0,  t.y), 0.002, 0.998)).rgb;
  vec3 e  = texture(uVideo, clamp(uv + vec2( t.x,  0.0), 0.002, 0.998)).rgb;
  vec3 w  = texture(uVideo, clamp(uv + vec2(-t.x,  0.0), 0.002, 0.998)).rgb;
  vec3 ne = texture(uVideo, clamp(uv + vec2( t.x, -t.y), 0.002, 0.998)).rgb;
  vec3 nw = texture(uVideo, clamp(uv + vec2(-t.x, -t.y), 0.002, 0.998)).rgb;
  vec3 se = texture(uVideo, clamp(uv + vec2( t.x,  t.y), 0.002, 0.998)).rgb;
  vec3 sw = texture(uVideo, clamp(uv + vec2(-t.x,  t.y), 0.002, 0.998)).rgb;

  float lc = lumaOf(c);

  // 1. 3x3 (9-tap) Low-Light Bilateral Denoise
  float shadowMask = 1.0 - smoothstep(0.08, 0.48, lc);
  float sigmaColor = mix(0.045, 0.145, shadowMask);
  float inv2Sigma2 = 1.0 / (2.0 * sigmaColor * sigmaColor);

  vec3 acc = c;
  float wSum = 1.0;

  vec3 dN = n - c;  float wN = exp(-dot(dN, dN) * inv2Sigma2) * 0.85; acc += n * wN; wSum += wN;
  vec3 dS = s - c;  float wS = exp(-dot(dS, dS) * inv2Sigma2) * 0.85; acc += s * wS; wSum += wS;
  vec3 dE = e - c;  float wE = exp(-dot(dE, dE) * inv2Sigma2) * 0.85; acc += e * wE; wSum += wE;
  vec3 dW = w - c;  float wW = exp(-dot(dW, dW) * inv2Sigma2) * 0.85; acc += w * wW; wSum += wW;
  vec3 dNE = ne - c; float wNE = exp(-dot(dNE, dNE) * inv2Sigma2) * 0.6; acc += ne * wNE; wSum += wNE;
  vec3 dNW = nw - c; float wNW = exp(-dot(dNW, dNW) * inv2Sigma2) * 0.6; acc += nw * wNW; wSum += wNW;
  vec3 dSE = se - c; float wSE = exp(-dot(dSE, dSE) * inv2Sigma2) * 0.6; acc += se * wSE; wSum += wSE;
  vec3 dSW = sw - c; float wSW = exp(-dot(dSW, dSW) * inv2Sigma2) * 0.6; acc += sw * wSW; wSum += wSW;

  vec3 denoised = acc / wSum;

  // 2. Contrast Adaptive Sharpening (CAS / edge-gated unsharp mask)
  vec3 mn = min(min(min(n, s), min(e, w)), denoised);
  vec3 mx = max(max(max(n, s), max(e, w)), denoised);
  float localContrast = lumaOf(mx - mn);
  float edgeGate = smoothstep(0.04, 0.18, localContrast) * (1.0 - shadowMask * 0.65);
  vec3 crossAvg = (n + s + e + w) * 0.25;
  vec3 sharpened = denoised + (denoised - crossAvg) * (0.68 * edgeGate);
  sharpened = clamp(sharpened, 0.0, 1.0);

  // 3. Highlight Roll-off & Cinematic Black-Level Tone Mapping
  float lum = lumaOf(sharpened);
  float rolledLum = (lum > 0.72) ? (0.72 + (lum - 0.72) / (1.0 + (lum - 0.72) * 2.2)) : lum;
  vec3 toned = sharpened * (rolledLum / max(lum, 0.0001));
  toned = clamp((toned - 0.025) * 1.055, 0.0, 1.0);
  toned = mix(toned, toned * toned * (3.0 - 2.0 * toned), 0.14);

  return mix(c, clamp(toned, 0.0, 1.0), clamp(uEnhance, 0.0, 1.0));
}

void main() {
  vec2 uv = clamp(screenToVideoUv(vScreenUv), vec2(0.002), vec2(0.998));

  if (uShaderType == 4) {
    fragColor = vec4(enhanceWebcamHD(uv), 1.0);
    return;
  }

  vec3 cam = enhanceWebcamHD(uv);
  float luma = lumaOf(cam);

  if (uShaderType == 0) {
    // Glass Triangle: luminous crisp optical glass
    vec3 col = cam * 1.44;
    col = (col - 0.5) * 1.34 + 0.5;
    float gLuma = lumaOf(col);
    col = mix(vec3(gLuma), col, 1.25);
    col = clamp(col + vec3(0.04, 0.06, 0.09), 0.0, 1.0);
    fragColor = vec4(col, 0.96);
    return;
  }

  if (uShaderType == 1) {
    // Thermal False-Color with boosted vibrance & crisp edge contrast
    vec2 t = uVideoTexel * 2.0;
    float lR = lumaOf(texture(uVideo, clamp(uv + vec2(t.x, 0.0), 0.002, 0.998)).rgb);
    float lU = lumaOf(texture(uVideo, clamp(uv + vec2(0.0, t.y), 0.002, 0.998)).rgb);
    float edge = clamp((abs(lR - luma) + abs(lU - luma)) * 4.5, 0.0, 1.0);
    float stretched = clamp((luma - 0.08) * 1.48 + edge * 0.22, 0.0, 1.0);
    vec3 thermal = thermalFalseColor(stretched);
    fragColor = vec4(thermal, 0.98);
    return;
  }

  // Chromatic Prism Ribbon: high-contrast vivid additive glass
  float hiContrast = clamp((luma - 0.48) * 1.48 + 0.64, 0.28, 1.36);
  vec3 prismCol = uTint.rgb * hiContrast;
  fragColor = vec4(prismCol, uTint.a);
}
`;

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type);
  if (!s) throw new Error('Không thể tạo WebGL2 shader.');
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const msg = gl.getShaderInfoLog(s) || 'Shader compile error';
    gl.deleteShader(s);
    throw new Error(msg);
  }
  return s;
}

function linkProgram(gl: WebGL2RenderingContext, vert: string, frag: string): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vert);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, frag);
  const p = gl.createProgram();
  if (!p) throw new Error('Không thể tạo WebGL2 program.');
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const msg = gl.getProgramInfoLog(p) || 'Program link error';
    gl.deleteProgram(p);
    throw new Error(msg);
  }
  return p;
}

export function computeCoverScale(
  viewW: number,
  viewH: number,
  vidW: number,
  vidH: number,
): [number, number] {
  const viewAspect = Math.max(viewW, 1) / Math.max(viewH, 1);
  const vidAspect = vidW > 0 && vidH > 0 ? vidW / vidH : 16 / 9;
  if (viewAspect > vidAspect) return [1.0, vidAspect / viewAspect];
  return [viewAspect / vidAspect, 1.0];
}

export function normToScreen(
  normX: number,
  normY: number,
  viewW: number,
  viewH: number,
  coverX: number,
  coverY: number,
  mirrored: boolean,
): Point2D {
  const rawX = mirrored ? 1 - normX : normX;
  const u = (rawX - 0.5) / coverX + 0.5;
  const v = (normY - 0.5) / coverY + 0.5;
  return { x: u * viewW, y: v * viewH };
}

/** Zero-GC 21-Joint 144Hz deltaTime-aware velocity-adaptive interpolator. */
class FullHandInterpolator {
  private buffers: [Float32Array, Float32Array] = [new Float32Array(42), new Float32Array(42)];
  private activeFlags: [boolean, boolean] = [false, false];
  private missedFrames: [number, number] = [0, 0];
  private poses: [HandPose2D, HandPose2D];

  constructor() {
    this.poses = [
      { pts: this.buffers[0], palmX: 0, palmY: 0, palmScale: 80, palmSpeedMps: 0, gesture: 'IDLE' },
      { pts: this.buffers[1], palmX: 0, palmY: 0, palmScale: 80, palmSpeedMps: 0, gesture: 'IDLE' },
    ];
  }

  reset(): void {
    this.activeFlags[0] = false;
    this.activeFlags[1] = false;
    this.missedFrames[0] = 0;
    this.missedFrames[1] = 0;
  }

  update(
    rawHands: Landmark3D[][],
    rawGestures: GestureType[],
    dt: number,
    viewW: number,
    viewH: number,
    coverX: number,
    coverY: number,
    mirrored: boolean,
    outHands: HandPose2D[],
  ): void {
    outHands.length = 0;

    let h0: Landmark3D[] | undefined = rawHands[0];
    let g0: GestureType = rawGestures[0] ?? 'IDLE';
    let h1: Landmark3D[] | undefined = rawHands[1];
    let g1: GestureType = rawGestures[1] ?? 'IDLE';

    if (h0 && h0.length >= 21 && h1 && h1.length >= 21) {
      const ax = mirrored ? 1 - h0[9].x : h0[9].x;
      const bx = mirrored ? 1 - h1[9].x : h1[9].x;
      if (ax > bx) {
        const th = h0; h0 = h1; h1 = th;
        const tg = g0; g0 = g1; g1 = tg;
      }
    }

    const safeDt = Math.max(dt, 0.004);
    for (let slot = 0; slot < 2; slot += 1) {
      const hand = slot === 0 ? h0 : h1;
      const gesture = slot === 0 ? g0 : g1;
      const buf = this.buffers[slot];
      const pose = this.poses[slot];

      if (!hand || hand.length < 21) {
        if (this.activeFlags[slot] && this.missedFrames[slot] < GRACE_FRAMES) {
          this.missedFrames[slot] += 1;
          pose.palmSpeedMps *= 0.85;
          outHands.push(pose);
        } else {
          this.activeFlags[slot] = false;
          this.missedFrames[slot] = 0;
        }
        continue;
      }

      this.missedFrames[slot] = 0;
      const prevPalmX = pose.palmX;
      const prevPalmY = pose.palmY;
      const wasActive = this.activeFlags[slot];

      for (let i = 0; i < 21; i += 1) {
        const lm = hand[i];
        const rawX = mirrored ? 1 - lm.x : lm.x;
        const tx = ((rawX - 0.5) / coverX + 0.5) * viewW;
        const ty = ((lm.y - 0.5) / coverY + 0.5) * viewH;

        if (!wasActive) {
          buf[i * 2] = tx;
          buf[i * 2 + 1] = ty;
        } else {
          const dx = tx - buf[i * 2];
          const dy = ty - buf[i * 2 + 1];
          const dist = Math.hypot(dx, dy);
          const rate = 18 + Math.min(1, dist / 32) * 50;
          const alpha = 1 - Math.exp(-rate * safeDt);
          buf[i * 2] += dx * alpha;
          buf[i * 2 + 1] += dy * alpha;
        }
      }

      this.activeFlags[slot] = true;
      const palmX = (buf[0] + buf[10] + buf[18] + buf[34]) * 0.25;
      const palmY = (buf[1] + buf[11] + buf[19] + buf[35]) * 0.25;
      const palmScale = Math.hypot(buf[18] - buf[0], buf[19] - buf[1]);

      const movePx = wasActive ? Math.hypot(palmX - prevPalmX, palmY - prevPalmY) : 0;
      const speedPxPerSec = movePx / safeDt;
      const instantMps = Math.min(9.9, speedPxPerSec / Math.max(palmScale * 7.5, 220));
      pose.palmSpeedMps = pose.palmSpeedMps * 0.75 + instantMps * 0.25;

      pose.palmX = palmX;
      pose.palmY = palmY;
      pose.palmScale = Math.max(palmScale, 36);
      pose.gesture = gesture;
      outHands.push(pose);
    }
  }
}

export class OpticalPrismEngine {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private vbo: WebGLBuffer;
  private videoTex: WebGLTexture;
  private interpolator = new FullHandInterpolator();

  // Pre-allocated scratch buffers (Zero-GC)
  private reusableHands: HandPose2D[] = [];
  private fullQuadVerts = new Float32Array([
    -1, 1, 0, 0,
     1, 1, 1, 0,
     1, -1, 1, 1,
    -1, 1, 0, 0,
     1, -1, 1, 1,
    -1, -1, 0, 1,
  ]);
  private polyVertScratch = new Float32Array(24 * 4);

  // Modular high-detail graphics engines (src/fx/)
  private galaxyFx = new GalaxyFx();
  private lightningFx = new LightningFx();
  private mysticFx = new MysticFx();

  // Motion Trail ring buffer
  private trailBuffer: TrailFrame[] = [];

  // Air-Canvas strokes
  private airStrokes: AirStroke[] = [];
  private activeStroke: AirStroke | null = null;

  // Flying light streaks from fingertip into centered 3D Hologram Text
  private noteBeams: NoteSparkBeam[] = [];
  // Motion swoosh wave when swiping horizontally to switch modes
  private swipeSwoosh: SwipeSwoosh | null = null;
  // Smooth 200ms crossfade easing when entering a new Mode
  private prevRenderedMode: CycleModeId | null = null;
  private modeFade = 1;
  // Smooth Lerp for HD+ Denoise shader uniform (no abrupt sharp/blur snaps)
  private smoothEnhance = 1;
  private elapsed = 0;

  private locs: {
    uVideo: WebGLUniformLocation | null;
    uCoverScale: WebGLUniformLocation | null;
    uVideoTexel: WebGLUniformLocation | null;
    uMirrored: WebGLUniformLocation | null;
    uEnhance: WebGLUniformLocation | null;
    uShaderType: WebGLUniformLocation | null;
    uTint: WebGLUniformLocation | null;
  };

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('Trình duyệt không hỗ trợ WebGL2.');
    this.gl = gl;

    this.program = linkProgram(gl, VERT_SRC, FRAG_SRC);
    this.locs = {
      uVideo: gl.getUniformLocation(this.program, 'uVideo'),
      uCoverScale: gl.getUniformLocation(this.program, 'uCoverScale'),
      uVideoTexel: gl.getUniformLocation(this.program, 'uVideoTexel'),
      uMirrored: gl.getUniformLocation(this.program, 'uMirrored'),
      uEnhance: gl.getUniformLocation(this.program, 'uEnhance'),
      uShaderType: gl.getUniformLocation(this.program, 'uShaderType'),
      uTint: gl.getUniformLocation(this.program, 'uTint'),
    };

    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    const tex = gl.createTexture();
    if (!vao || !vbo || !tex) throw new Error('Không thể khởi tạo tài nguyên WebGL2.');

    this.vao = vao;
    this.vbo = vbo;
    this.videoTex = tex;

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.polyVertScratch.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 255]),
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  resetSmoothing(): void {
    this.interpolator.reset();
    this.galaxyFx.reset();
    this.trailBuffer = [];
  }

  clearDrawing(): void {
    this.airStrokes = [];
    this.activeStroke = null;
  }

  /** Spawns a sweeping light ribbon across the viewport in the exact Air Swipe direction. */
  spawnSwipeSwoosh(dir: -1 | 1): void {
    this.swipeSwoosh = { dir, age: 0, duration: 0.28 };
    this.modeFade = 0;
  }

  spawnNoteSpark(
    normX: number,
    normY: number,
    targetX: number,
    targetY: number,
    viewW: number,
    viewH: number,
    videoW: number,
    videoH: number,
    mirrored: boolean,
  ): void {
    const [coverX, coverY] = computeCoverScale(viewW, viewH, videoW, videoH);
    const start = normToScreen(normX, normY, viewW, viewH, coverX, coverY, mirrored);
    this.noteBeams.push({
      sx: start.x,
      sy: start.y,
      tx: targetX,
      ty: targetY,
      cx: (start.x + targetX) * 0.5 + (Math.random() - 0.5) * 90,
      cy: Math.min(start.y, targetY) - 65 - Math.random() * 55,
      age: 0,
      duration: 0.26,
    });
  }

  uploadVideoFrame(video: HTMLVideoElement): void {
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  private drawFullscreenWebcamPass(): void {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    gl.uniform1i(this.locs.uShaderType, 4);
    gl.uniform4f(this.locs.uTint, 1, 1, 1, 1);

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.fullQuadVerts);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  private drawPolygonWebGL(poly: RenderPolygon, viewW: number, viewH: number): void {
    const pts = poly.pts;
    if (pts.length < 3) return;
    const gl = this.gl;
    const triCount = pts.length - 2;
    const data = this.polyVertScratch;
    let offset = 0;

    const pushVertex = (p: Point2D) => {
      const u = p.x / viewW;
      const v = p.y / viewH;
      data[offset++] = u * 2 - 1;
      data[offset++] = 1 - v * 2;
      data[offset++] = u;
      data[offset++] = v;
    };

    for (let i = 1; i < pts.length - 1; i += 1) {
      pushVertex(pts[0]);
      pushVertex(pts[i]);
      pushVertex(pts[i + 1]);
    }

    gl.uniform1i(this.locs.uShaderType, poly.shaderType);
    gl.uniform4f(this.locs.uTint, poly.tint[0], poly.tint[1], poly.tint[2], poly.tint[3]);

    gl.enable(gl.BLEND);
    if (poly.additive) {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    } else {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data.subarray(0, offset));
    gl.drawArrays(gl.TRIANGLES, 0, triCount * 3);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  private drawAirStrokes(ctx: CanvasRenderingContext2D): void {
    if (this.airStrokes.length === 0) return;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const stroke of this.airStrokes) {
      const pts = stroke.points;
      if (pts.length < 2) continue;

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length - 1; i += 1) {
        const midX = (pts[i].x + pts[i + 1].x) * 0.5;
        const midY = (pts[i].y + pts[i + 1].y) * 0.5;
        ctx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
      }
      const last = pts[pts.length - 1];
      ctx.lineTo(last.x, last.y);

      ctx.strokeStyle = stroke.color;
      ctx.lineWidth = 9;
      ctx.globalAlpha = 0.22;
      ctx.stroke();

      ctx.lineWidth = 4;
      ctx.globalAlpha = 0.72;
      ctx.stroke();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.95;
      ctx.stroke();
    }

    ctx.restore();
  }

  private drawNoteBeams(ctx: CanvasRenderingContext2D, dt: number): void {
    if (this.noteBeams.length === 0) return;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    const alive: NoteSparkBeam[] = [];

    for (const b of this.noteBeams) {
      b.age += dt;
      if (b.age >= b.duration) continue;

      const t = b.age / b.duration;
      const tailT = Math.max(0, t - 0.28);
      const evalPt = (u: number) => {
        const inv = 1 - u;
        return {
          x: inv * inv * b.sx + 2 * inv * u * b.cx + u * u * b.tx,
          y: inv * inv * b.sy + 2 * inv * u * b.cy + u * u * b.ty,
        };
      };

      const pHead = evalPt(t);
      const pMid = evalPt((t + tailT) * 0.5);
      const pTail = evalPt(tailT);
      const alpha = Math.sin(t * Math.PI);

      ctx.strokeStyle = `rgba(34, 211, 238, ${alpha * 0.9})`;
      ctx.lineWidth = 4.5;
      ctx.beginPath();
      ctx.moveTo(pTail.x, pTail.y);
      ctx.quadraticCurveTo(pMid.x, pMid.y, pHead.x, pHead.y);
      ctx.stroke();

      ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
      ctx.lineWidth = 1.6;
      ctx.stroke();

      ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
      ctx.beginPath();
      ctx.arc(pHead.x, pHead.y, 3.2, 0, Math.PI * 2);
      ctx.fill();

      alive.push(b);
    }

    this.noteBeams = alive;
    ctx.restore();
  }

  private drawSwipeSwoosh(
    ctx: CanvasRenderingContext2D,
    viewW: number,
    viewH: number,
    dt: number,
  ): void {
    if (!this.swipeSwoosh) return;
    const sw = this.swipeSwoosh;
    sw.age += dt;
    if (sw.age >= sw.duration) {
      this.swipeSwoosh = null;
      return;
    }

    const t = sw.age / sw.duration;
    const ease = 1 - Math.pow(1 - t, 2.6);
    const alpha = Math.sin(t * Math.PI);
    const bandW = viewW * 0.26;
    const xPos =
      sw.dir === 1 ? -bandW + (viewW + bandW * 2) * ease : viewW + bandW - (viewW + bandW * 2) * ease;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const grad = ctx.createLinearGradient(xPos - bandW * 0.5, 0, xPos + bandW * 0.5, 0);
    grad.addColorStop(0, 'rgba(34, 211, 238, 0)');
    grad.addColorStop(0.42, `rgba(34, 211, 238, ${alpha * 0.26})`);
    grad.addColorStop(0.5, `rgba(255, 255, 255, ${alpha * 0.45})`);
    grad.addColorStop(0.58, `rgba(168, 85, 247, ${alpha * 0.24})`);
    grad.addColorStop(1, 'rgba(168, 85, 247, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(xPos - bandW * 0.5, 0, bandW, viewH);
    ctx.restore();
  }

  renderFrame(
    dt: number,
    viewW: number,
    viewH: number,
    videoW: number,
    videoH: number,
    rawHands: Landmark3D[][],
    gestures: GestureType[],
    settings: PrismSettings,
    isHoveringUi: boolean,
    overlayCtx: CanvasRenderingContext2D | null,
  ): FrameOutput {
    this.elapsed += dt;
    const gl = this.gl;
    gl.viewport(0, 0, viewW, viewH);

    if (overlayCtx) {
      overlayCtx.clearRect(0, 0, viewW, viewH);
    }

    const [coverX, coverY] = computeCoverScale(viewW, viewH, videoW, videoH);

    this.interpolator.update(
      rawHands,
      gestures,
      dt,
      viewW,
      viewH,
      coverX,
      coverY,
      settings.mirrored,
      this.reusableHands,
    );
    const hands = this.reusableHands;

    const leadHand = hands[0] ?? null;
    const indexTipPx: Point2D | null = leadHand ? { x: leadHand.pts[16], y: leadHand.pts[17] } : null;
    const palmScreenNorm: Point2D | null = leadHand
      ? { x: leadHand.palmX / Math.max(viewW, 1), y: leadHand.palmY / Math.max(viewH, 1) }
      : null;
    const isLeadFist = leadHand ? leadHand.gesture === 'FIST' : false;

    // Resolve active mode
    let activeMode: CycleModeId = 'GLASS';
    if (settings.mode !== 'AUTO') {
      activeMode = settings.mode;
    } else if (hands.length <= 1) {
      activeMode = 'GLASS';
    } else {
      const g0 = hands[0].gesture;
      const g1 = hands[1].gesture;
      const pointingOrPinched =
        g0 === 'PINCH' || g0 === 'POINT' || g1 === 'PINCH' || g1 === 'POINT';
      activeMode = pointingOrPinched ? 'THERMAL' : 'PRISM';
    }

    // Track smooth 200ms fade-in/easing whenever switching between Modes
    if (this.prevRenderedMode !== activeMode) {
      this.prevRenderedMode = activeMode;
      this.modeFade = 0;
      if (activeMode === 'GALAXY') this.galaxyFx.reset();
    } else if (this.modeFade < 1) {
      this.modeFade = Math.min(1, this.modeFade + dt / 0.22);
    }
    const modeEase = 1 - Math.pow(1 - this.modeFade, 2.4);

    // Air-Canvas drawing
    if (settings.airDrawEnabled && leadHand && !isHoveringUi && leadHand.pts[17] < viewH - 92) {
      if (leadHand.gesture === 'POINT') {
        const tx = leadHand.pts[16];
        const ty = leadHand.pts[17];
        const colorHex = AIR_DRAW_COLORS[settings.drawColor].hex;
        if (!this.activeStroke || this.activeStroke.color !== colorHex) {
          this.activeStroke = { points: [{ x: tx, y: ty }], color: colorHex };
          this.airStrokes.push(this.activeStroke);
        } else {
          const pts = this.activeStroke.points;
          const prev = pts[pts.length - 1];
          if (Math.hypot(tx - prev.x, ty - prev.y) >= 3) {
            pts.push({ x: tx, y: ty });
          }
        }
      } else {
        this.activeStroke = null;
      }
    } else {
      this.activeStroke = null;
    }

    const livePolygons = settings.airDrawEnabled ? [] : buildOpticalPolygons(hands, activeMode);

    // Motion Trail ring buffer
    if (settings.trailEnabled && livePolygons.length > 0) {
      this.trailBuffer.push({
        outlines: livePolygons.map((p) => ({
          pts: p.pts.map((v) => ({ x: v.x, y: v.y })),
          color: p.glow,
        })),
      });
      if (this.trailBuffer.length > TRAIL_MAX_FRAMES) {
        this.trailBuffer.shift();
      }
    } else if (!settings.trailEnabled) {
      this.trailBuffer = [];
    } else if (this.trailBuffer.length > 0) {
      this.trailBuffer.shift();
    }

    // Smoothly Lerp HD+ Denoise shader uniform so transitions never snap harshly
    const targetEnhance = settings.hdDenoise ? 1.0 : 0.0;
    this.smoothEnhance += (targetEnhance - this.smoothEnhance) * (1 - Math.exp(-5.0 * dt));

    // --- WebGL2 Pass 1: ALWAYS draw latest videoTex every rAF frame (never blank frame) ---
    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.uniform1i(this.locs.uVideo, 0);
    gl.uniform2f(this.locs.uCoverScale, coverX, coverY);
    gl.uniform2f(this.locs.uVideoTexel, 1.0 / Math.max(videoW, 1), 1.0 / Math.max(videoH, 1));
    gl.uniform1f(this.locs.uMirrored, settings.mirrored ? 1.0 : 0.0);
    gl.uniform1f(this.locs.uEnhance, this.smoothEnhance);

    this.drawFullscreenWebcamPass();

    // --- WebGL2 Pass 2: Vivid Glass / Thermal / Prism Shader Polygons ---
    for (let i = 0; i < livePolygons.length; i += 1) {
      this.drawPolygonWebGL(livePolygons[i], viewW, viewH);
    }

    gl.bindTexture(gl.TEXTURE_2D, null);

    // --- Canvas 2D Overlay (Zero-Copy, dpr=1, 0 shadowBlur) ---
    if (overlayCtx) {
      if (settings.trailEnabled && this.trailBuffer.length > 1) {
        overlayCtx.save();
        overlayCtx.globalCompositeOperation = 'lighter';
        const len = this.trailBuffer.length;
        for (let i = 0; i < len - 1; i += 1) {
          const frac = (i + 1) / len;
          const alpha = frac * frac * 0.52;
          const frame = this.trailBuffer[i];
          for (let j = 0; j < frame.outlines.length; j += 1) {
            renderTrailOutline(overlayCtx, frame.outlines[j].pts, frame.outlines[j].color, alpha);
          }
        }
        overlayCtx.restore();
      }

      // Chromatic Aberration Edge Split around Glass / Thermal / Prism polygons
      if (livePolygons.length > 0) {
        renderPrismChromaticEdges(overlayCtx, livePolygons, modeEase);
      }

      // Upgraded Modular VFX with smooth 200ms intro easing
      if (!settings.airDrawEnabled) {
        if (activeMode === 'GALAXY') {
          this.galaxyFx.updateAndRender(overlayCtx, hands, dt, this.modeFade);
        } else if (activeMode === 'LIGHTNING') {
          this.lightningFx.updateAndRender(overlayCtx, hands, dt, modeEase);
        } else if (activeMode === 'MYSTIC') {
          this.mysticFx.updateAndRender(overlayCtx, hands, dt, this.modeFade);
        }
      }

      // Cybernetic HUD Skeleton
      if (settings.cyberHud && hands.length > 0) {
        drawCyberneticHandHud(overlayCtx, hands, activeMode, this.elapsed);
      }

      // Air-Canvas Neon Strokes
      this.drawAirStrokes(overlayCtx);

      // Flying Note Sparks into centered 3D Hologram Text
      this.drawNoteBeams(overlayCtx, dt);

      // Air Swipe Motion Swoosh Wave
      this.drawSwipeSwoosh(overlayCtx, viewW, viewH, dt);
    }

    return {
      activeMode,
      indexTipPx,
      palmScreenNorm,
      isLeadFist,
    };
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.vbo);
    gl.deleteTexture(this.videoTex);
  }
}
