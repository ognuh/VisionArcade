import { Cosmos3DEngine } from '../fx/cosmos';
import { MysticFx } from '../fx/mystic';
import {
  buildOpticalPolygons,
  renderPrismChromaticEdges,
  renderTrailOutline,
} from '../fx/prism';
import { TransitionManager } from '../fx/transitionManager';
import type { HandPose2D, Point2D, RenderPolygon } from '../fx/types';
import type {
  AirDrawColorId,
  CycleModeId,
  GestureType,
  Landmark3D,
  PrismSettings,
} from '../types/vision';
import type { SwipeIndicatorState } from '../vision/gestureManager';
import { OneEuroAxis } from '../vision/smoothing';
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

interface PolyDissolveGhost {
  pts: Point2D[];
  cx: number;
  cy: number;
  color: string;
  age: number;
  duration: number;
  motes: { x: number; y: number; vx: number; vy: number }[];
}

export interface FrameOutput {
  activeMode: CycleModeId;
  indexTipPx: Point2D | null;
  palmScreenNorm: Point2D | null;
  isLeadFist: boolean;
}

export const AIR_DRAW_COLORS: Record<AirDrawColorId, { hex: string; label: string }> = {
  CYAN: { hex: '#22d3ee', label: 'Cyan' },
  MAGENTA: { hex: '#f43f97', label: 'Magenta' },
  GREEN: { hex: '#4ade80', label: 'Neon Green' },
  GOLD: { hex: '#facc15', label: 'Gold' },
};

const HAND_PERSISTENCE_SEC = 0.18;
const ONE_EURO_MIN_CUTOFF = 1.2;
const ONE_EURO_BETA = 0.04;
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
uniform sampler2D uPrevFrame;
uniform vec2 uCoverScale;
uniform vec2 uVideoTexel;
uniform float uMirrored;
uniform float uEnhance;
uniform float uHasPrev;
uniform int uShaderType; // 0=GLASS, 1=THERMAL, 2=PRISM, 4=CINEMA_CLEAN_PASS1, 5=BLIT_FBO
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

vec3 rgbToYCbCr(vec3 c) {
  float y  =  0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
  float cb = -0.168736 * c.r - 0.331264 * c.g + 0.5 * c.b;
  float cr =  0.5 * c.r - 0.418688 * c.g - 0.081312 * c.b;
  return vec3(y, cb, cr);
}

vec3 yCbCrToRgb(vec3 ycc) {
  float r = ycc.x + 1.402 * ycc.z;
  float g = ycc.x - 0.344136 * ycc.y - 0.714136 * ycc.z;
  float b = ycc.x + 1.772 * ycc.y;
  return clamp(vec3(r, g, b), 0.0, 1.0);
}

vec3 acesFilmic(vec3 x) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 thermalFalseColor(float t) {
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

/**
 * Cinema Clean GPU Pipeline:
 * 1. Motion-Adaptive Temporal Denoise (mix 35% prev frame when diff < 0.08 — zero ghosting)
 * 2. YCbCr Chroma Smoothing (erases red/green low-light blotches while keeping Luma sharp)
 * 3. Edge-Gated CAS Sharpening
 * 4. ACES Filmic Highlight Compression (luma > 0.72) + Cool Shadow Cinema Grade + 0.25 Vignette
 */
vec3 cinemaCleanPass(vec2 uv, vec2 fboUv) {
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
  float shadowMask = 1.0 - smoothstep(0.08, 0.50, lc);

  // 1. Spatial 3x3 Bilateral Denoise + Separate Chroma (Cb/Cr) Smoothing
  float sigmaLuma = mix(0.04, 0.12, shadowMask);
  float inv2SigmaL = 1.0 / (2.0 * sigmaLuma * sigmaLuma);
  float inv2SigmaC = 1.0 / (2.0 * 0.19 * 0.19); // wider chroma filter wipes color noise on skin/walls

  vec3 yccC = rgbToYCbCr(c);
  float yAcc = yccC.x;
  float yW = 1.0;
  vec2 cbcrAcc = yccC.yz;
  float cW = 1.0;

  vec3 nbrs[8];
  nbrs[0] = n; nbrs[1] = s; nbrs[2] = e; nbrs[3] = w;
  nbrs[4] = ne; nbrs[5] = nw; nbrs[6] = se; nbrs[7] = sw;

  for (int i = 0; i < 8; i++) {
    vec3 nb = nbrs[i];
    vec3 yccN = rgbToYCbCr(nb);
    float spW = (i < 4) ? 0.85 : 0.6;
    float dy = yccN.x - yccC.x;
    float wl = exp(-(dy * dy) * inv2SigmaL) * spW;
    yAcc += yccN.x * wl;
    yW += wl;

    vec3 diffRgb = nb - c;
    float wc = exp(-dot(diffRgb, diffRgb) * inv2SigmaC) * spW;
    cbcrAcc += yccN.yz * wc;
    cW += wc;
  }

  float denoisedY = yAcc / yW;
  vec2 smoothCbCr = cbcrAcc / cW;

  // 2. Edge-Gated CAS Sharpening on Luma channel ONLY (preserves crisp eyes/hair without chroma noise)
  float minY = min(min(min(lumaOf(n), lumaOf(s)), min(lumaOf(e), lumaOf(w))), denoisedY);
  float maxY = max(max(max(lumaOf(n), lumaOf(s)), max(lumaOf(e), lumaOf(w))), denoisedY);
  float edgeGate = smoothstep(0.04, 0.18, maxY - minY) * (1.0 - shadowMask * 0.6);
  float crossY = (lumaOf(n) + lumaOf(s) + lumaOf(e) + lumaOf(w)) * 0.25;
  float sharpY = clamp(denoisedY + (denoisedY - crossY) * (0.65 * edgeGate), 0.0, 1.0);

  vec3 clean = yCbCrToRgb(vec3(sharpY, smoothCbCr));

  // 3. Motion-Adaptive Temporal Denoise (35% prev frame blend where diff < 0.08)
  if (uHasPrev > 0.5) {
    vec3 prev = texture(uPrevFrame, fboUv).rgb;
    float diff = length(clean - prev);
    float temporalWeight = 0.35 * (1.0 - smoothstep(0.025, 0.08, diff));
    clean = mix(clean, prev, temporalWeight);
  }

  // 4. ACES Filmic Highlight Compression (luma > 0.72) + Cinema Grade + Soft Vignette (0.25)
  float lum = lumaOf(clean);
  if (lum > 0.72) {
    vec3 filmic = acesFilmic(clean * 1.15);
    float hiBlend = smoothstep(0.72, 0.98, lum);
    clean = mix(clean, filmic, hiBlend * 0.78);
  }

  // Subtle cool-slate deep shadows & natural skin midtones
  vec3 coolShadow = vec3(0.95, 0.98, 1.04);
  clean *= mix(vec3(1.0), coolShadow, shadowMask * 0.35);
  clean = clamp((clean - 0.02) * 1.04, 0.0, 1.0);

  // Soft radial vignette (0.25)
  vec2 vig = vScreenUv - 0.5;
  float vigFactor = 1.0 - dot(vig, vig) * 0.92;
  clean *= mix(1.0, clamp(vigFactor, 0.75, 1.0), 0.95);

  return mix(c, clamp(clean, 0.0, 1.0), clamp(uEnhance, 0.0, 1.0));
}

void main() {
  if (uShaderType == 5) {
    // Pass 2: Fast blit of Cinema Clean FBO
    vec2 fboUv = vec2(vScreenUv.x, 1.0 - vScreenUv.y);
    fragColor = vec4(texture(uPrevFrame, fboUv).rgb, 1.0);
    return;
  }

  vec2 uv = clamp(screenToVideoUv(vScreenUv), vec2(0.002), vec2(0.998));

  if (uShaderType == 4) {
    // Pass 1: Cinema Clean into FBO
    vec2 fboUv = vec2(vScreenUv.x, 1.0 - vScreenUv.y);
    fragColor = vec4(cinemaCleanPass(uv, fboUv), 1.0);
    return;
  }

  // Optical Polygons read cleanly from the already Cinema-Clean FBO texture
  vec2 fboUv = vec2(vScreenUv.x, 1.0 - vScreenUv.y);
  vec3 cam = texture(uPrevFrame, fboUv).rgb;
  float luma = lumaOf(cam);

  if (uShaderType == 0) {
    vec3 col = cam * 1.44;
    col = (col - 0.5) * 1.34 + 0.5;
    float gLuma = lumaOf(col);
    col = mix(vec3(gLuma), col, 1.25);
    col = clamp(col + vec3(0.04, 0.06, 0.09), 0.0, 1.0);
    fragColor = vec4(col, 0.96);
    return;
  }

  if (uShaderType == 1) {
    vec2 t = uVideoTexel * 2.0;
    float lR = lumaOf(texture(uVideo, clamp(uv + vec2(t.x, 0.0), 0.002, 0.998)).rgb);
    float lU = lumaOf(texture(uVideo, clamp(uv + vec2(0.0, t.y), 0.002, 0.998)).rgb);
    float edge = clamp((abs(lR - luma) + abs(lU - luma)) * 4.5, 0.0, 1.0);
    float stretched = clamp((luma - 0.08) * 1.48 + edge * 0.22, 0.0, 1.0);
    vec3 thermal = thermalFalseColor(stretched);
    fragColor = vec4(thermal, 0.98);
    return;
  }

  // Chromatic Prism Ribbon
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

/**
 * Zero-GC 21-Joint Hand Tracker Smoother:
 * - Nearest-distance stable slot ID matching (never swaps hand IDs based on MediaPipe array order)
 * - Synced One-Euro Filter (minCutoff: 1.2, beta: 0.04) across all 21 joints
 * - 180ms Hand Persistence grace period with gentle confidence taper on transient frame drops
 */
class FullHandInterpolator {
  private buffers: [Float32Array, Float32Array] = [new Float32Array(42), new Float32Array(42)];
  private filters: [OneEuroAxis[], OneEuroAxis[]];
  private activeFlags: [boolean, boolean] = [false, false];
  private missedSec: [number, number] = [0, 0];
  private poses: [HandPose2D, HandPose2D];

  constructor() {
    this.filters = [
      Array.from({ length: 42 }, () => new OneEuroAxis()),
      Array.from({ length: 42 }, () => new OneEuroAxis()),
    ];
    this.poses = [
      {
        id: 0,
        pts: this.buffers[0],
        palmX: 0,
        palmY: 0,
        palmScale: 80,
        palmSpeedMps: 0,
        confidence: 1,
        gesture: 'IDLE',
      },
      {
        id: 1,
        pts: this.buffers[1],
        palmX: 0,
        palmY: 0,
        palmScale: 80,
        palmSpeedMps: 0,
        confidence: 1,
        gesture: 'IDLE',
      },
    ];
  }

  reset(): void {
    this.activeFlags[0] = false;
    this.activeFlags[1] = false;
    this.missedSec[0] = 0;
    this.missedSec[1] = 0;
    for (let s = 0; s < 2; s += 1) {
      for (let i = 0; i < 42; i += 1) this.filters[s][i].reset();
    }
  }

  private getScreenPalm(
    hand: Landmark3D[],
    viewW: number,
    viewH: number,
    coverX: number,
    coverY: number,
    mirrored: boolean,
  ): { x: number; y: number } {
    const rawX = (hand[0].x + hand[5].x + hand[9].x + hand[17].x) * 0.25;
    const rawY = (hand[0].y + hand[5].y + hand[9].y + hand[17].y) * 0.25;
    const mx = mirrored ? 1 - rawX : rawX;
    return {
      x: ((mx - 0.5) / coverX + 0.5) * viewW,
      y: ((rawY - 0.5) / coverY + 0.5) * viewH,
    };
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

    // Collect valid incoming hands (up to 2)
    const validHands: { hand: Landmark3D[]; gesture: GestureType; px: number; py: number }[] = [];
    for (let i = 0; i < Math.min(rawHands.length, 2); i += 1) {
      const h = rawHands[i];
      if (h && h.length >= 21) {
        const palm = this.getScreenPalm(h, viewW, viewH, coverX, coverY, mirrored);
        validHands.push({ hand: h, gesture: rawGestures[i] ?? 'IDLE', px: palm.x, py: palm.y });
      }
    }

    // Nearest-distance stable slot assignment (prevents ID swapping across frames)
    const assignedHand: [Landmark3D[] | null, Landmark3D[] | null] = [null, null];
    const assignedGesture: [GestureType, GestureType] = ['IDLE', 'IDLE'];

    if (validHands.length === 1) {
      const vh = validHands[0];
      let targetSlot: 0 | 1 = 0;
      if (this.activeFlags[0] && this.activeFlags[1]) {
        const d0 = Math.hypot(vh.px - this.poses[0].palmX, vh.py - this.poses[0].palmY);
        const d1 = Math.hypot(vh.px - this.poses[1].palmX, vh.py - this.poses[1].palmY);
        targetSlot = d1 < d0 ? 1 : 0;
      } else if (this.activeFlags[1] && !this.activeFlags[0]) {
        targetSlot = 1;
      }
      assignedHand[targetSlot] = vh.hand;
      assignedGesture[targetSlot] = vh.gesture;
    } else if (validHands.length === 2) {
      const a = validHands[0];
      const b = validHands[1];
      let swap = false;
      if (this.activeFlags[0] && this.activeFlags[1]) {
        const costKeep =
          Math.hypot(a.px - this.poses[0].palmX, a.py - this.poses[0].palmY) +
          Math.hypot(b.px - this.poses[1].palmX, b.py - this.poses[1].palmY);
        const costSwap =
          Math.hypot(a.px - this.poses[1].palmX, a.py - this.poses[1].palmY) +
          Math.hypot(b.px - this.poses[0].palmX, b.py - this.poses[0].palmY);
        swap = costSwap < costKeep;
      } else if (this.activeFlags[0]) {
        const dA = Math.hypot(a.px - this.poses[0].palmX, a.py - this.poses[0].palmY);
        const dB = Math.hypot(b.px - this.poses[0].palmX, b.py - this.poses[0].palmY);
        swap = dB < dA;
      } else if (this.activeFlags[1]) {
        const dA = Math.hypot(a.px - this.poses[1].palmX, a.py - this.poses[1].palmY);
        const dB = Math.hypot(b.px - this.poses[1].palmX, b.py - this.poses[1].palmY);
        swap = dA < dB;
      } else {
        // Fresh 2-hand appearance: deterministic left=0, right=1 initial binding
        swap = a.px > b.px;
      }

      if (swap) {
        assignedHand[0] = b.hand;
        assignedGesture[0] = b.gesture;
        assignedHand[1] = a.hand;
        assignedGesture[1] = a.gesture;
      } else {
        assignedHand[0] = a.hand;
        assignedGesture[0] = a.gesture;
        assignedHand[1] = b.hand;
        assignedGesture[1] = b.gesture;
      }
    }

    const safeDt = Math.min(Math.max(dt, 0.004), 0.08);
    for (let slot = 0; slot < 2; slot += 1) {
      const hand = assignedHand[slot];
      const gesture = assignedGesture[slot];
      const buf = this.buffers[slot];
      const flt = this.filters[slot];
      const pose = this.poses[slot];

      if (!hand) {
        // Hand Persistence: retain last smoothed pose up to 180ms with gentle confidence decay
        if (this.activeFlags[slot] && this.missedSec[slot] < HAND_PERSISTENCE_SEC) {
          this.missedSec[slot] += safeDt;
          const dropFrac = Math.min(1, this.missedSec[slot] / HAND_PERSISTENCE_SEC);
          pose.confidence = Math.max(0.35, 1 - dropFrac * 0.65);
          pose.palmSpeedMps *= 0.85;
          outHands.push(pose);
        } else {
          this.activeFlags[slot] = false;
          this.missedSec[slot] = 0;
          pose.confidence = 0;
          for (let i = 0; i < 42; i += 1) flt[i].reset();
        }
        continue;
      }

      const wasActive = this.activeFlags[slot];
      this.missedSec[slot] = 0;
      const prevPalmX = pose.palmX;
      const prevPalmY = pose.palmY;

      for (let i = 0; i < 21; i += 1) {
        const lm = hand[i];
        const rawX = mirrored ? 1 - lm.x : lm.x;
        const tx = ((rawX - 0.5) / coverX + 0.5) * viewW;
        const ty = ((lm.y - 0.5) / coverY + 0.5) * viewH;

        if (!wasActive) {
          flt[i * 2].reset();
          flt[i * 2 + 1].reset();
          buf[i * 2] = flt[i * 2].filter(tx, safeDt, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA);
          buf[i * 2 + 1] = flt[i * 2 + 1].filter(ty, safeDt, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA);
        } else {
          buf[i * 2] = flt[i * 2].filter(tx, safeDt, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA);
          buf[i * 2 + 1] = flt[i * 2 + 1].filter(ty, safeDt, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA);
        }
      }

      this.activeFlags[slot] = true;
      pose.confidence = 1;
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

  // 2-Pass Cinema Clean Ping-Pong FBOs for Motion-Adaptive Temporal Denoise
  private prevTexs: [WebGLTexture, WebGLTexture];
  private prevFbos: [WebGLFramebuffer, WebGLFramebuffer];
  private fboReadIdx: 0 | 1 = 0;
  private fboW = 0;
  private fboH = 0;
  private hasPrevFrame = false;

  private interpolator = new FullHandInterpolator();

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
  private cosmosFx: Cosmos3DEngine | null = null;
  private mysticFx = new MysticFx();

  private trailBuffer: TrailFrame[] = [];
  private airStrokes: AirStroke[] = [];
  private activeStroke: AirStroke | null = null;
  private noteBeams: NoteSparkBeam[] = [];
  private swipeSwoosh: SwipeSwoosh | null = null;
  private lastPolyCache: { pts: Point2D[]; color: string }[] = [];
  private polyDissolves: PolyDissolveGhost[] = [];

  private transitionMgr = new TransitionManager();
  private smoothEnhance = 1;
  private elapsed = 0;

  private locs: {
    uVideo: WebGLUniformLocation | null;
    uPrevFrame: WebGLUniformLocation | null;
    uCoverScale: WebGLUniformLocation | null;
    uVideoTexel: WebGLUniformLocation | null;
    uMirrored: WebGLUniformLocation | null;
    uEnhance: WebGLUniformLocation | null;
    uHasPrev: WebGLUniformLocation | null;
    uShaderType: WebGLUniformLocation | null;
    uTint: WebGLUniformLocation | null;
  };

  constructor(canvas: HTMLCanvasElement, cosmosCanvas?: HTMLCanvasElement | null) {
    if (cosmosCanvas) {
      this.cosmosFx = new Cosmos3DEngine(cosmosCanvas);
    }
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is not supported by this browser.');
    this.gl = gl;

    this.program = linkProgram(gl, VERT_SRC, FRAG_SRC);
    this.locs = {
      uVideo: gl.getUniformLocation(this.program, 'uVideo'),
      uPrevFrame: gl.getUniformLocation(this.program, 'uPrevFrame'),
      uCoverScale: gl.getUniformLocation(this.program, 'uCoverScale'),
      uVideoTexel: gl.getUniformLocation(this.program, 'uVideoTexel'),
      uMirrored: gl.getUniformLocation(this.program, 'uMirrored'),
      uEnhance: gl.getUniformLocation(this.program, 'uEnhance'),
      uHasPrev: gl.getUniformLocation(this.program, 'uHasPrev'),
      uShaderType: gl.getUniformLocation(this.program, 'uShaderType'),
      uTint: gl.getUniformLocation(this.program, 'uTint'),
    };

    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    if (!vao || !vbo) throw new Error('Không thể khởi tạo tài nguyên WebGL2.');

    this.vao = vao;
    this.vbo = vbo;
    this.videoTex = this.createLinearTex();
    this.prevTexs = [this.createLinearTex(), this.createLinearTex()];
    this.prevFbos = [this.createFbo(this.prevTexs[0]), this.createFbo(this.prevTexs[1])];

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.polyVertScratch.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
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

    this.ensureFboSize(canvas.width || 1280, canvas.height || 720);
  }

  private createLinearTex(): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture();
    if (!tex) throw new Error('Không thể tạo WebGL2 texture.');
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  private createFbo(tex: WebGLTexture): WebGLFramebuffer {
    const gl = this.gl;
    const fbo = gl.createFramebuffer();
    if (!fbo) throw new Error('Không thể tạo WebGL2 framebuffer.');
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
  }

  private ensureFboSize(w: number, h: number): void {
    const pw = Math.max(1, Math.floor(w));
    const ph = Math.max(1, Math.floor(h));
    if (pw === this.fboW && ph === this.fboH) return;
    this.fboW = pw;
    this.fboH = ph;
    this.hasPrevFrame = false;

    const gl = this.gl;
    for (let i = 0; i < 2; i += 1) {
      gl.bindTexture(gl.TEXTURE_2D, this.prevTexs[i]);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, pw, ph, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  resetSmoothing(): void {
    this.interpolator.reset();
    this.cosmosFx?.reset();
    this.cosmosFx?.resetLightning();
    this.mysticFx.reset();
    this.trailBuffer = [];
    this.hasPrevFrame = false;
  }

  isCosmosBlackHoleBusy(): boolean {
    return this.cosmosFx?.isBlackHoleBusy() ?? false;
  }

  prewarmCosmosNow(): void {
    this.cosmosFx?.prewarmAllNow();
  }

  clearDrawing(): void {
    this.airStrokes = [];
    this.activeStroke = null;
  }

  spawnSwipeSwoosh(dir: -1 | 1): void {
    this.swipeSwoosh = { dir, age: 0, duration: 0.32 };
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

  private drawFullQuad(shaderType: number): void {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    gl.uniform1i(this.locs.uShaderType, shaderType);
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

  /** Air Swipe 3.0 Visual Feedback: Curved luminous arc + motion streak sweeping across viewport. */
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
    const ease = 1 - Math.pow(1 - t, 2.5);
    const alpha = Math.sin(t * Math.PI);
    const bandW = viewW * 0.24;
    const xPos =
      sw.dir === 1 ? -bandW + (viewW + bandW * 2) * ease : viewW + bandW - (viewW + bandW * 2) * ease;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    // Soft sweeping ambient light wash
    const grad = ctx.createLinearGradient(xPos - bandW * 0.5, 0, xPos + bandW * 0.5, 0);
    grad.addColorStop(0, 'rgba(34, 211, 238, 0)');
    grad.addColorStop(0.45, `rgba(34, 211, 238, ${alpha * 0.24})`);
    grad.addColorStop(0.5, `rgba(255, 255, 255, ${alpha * 0.42})`);
    grad.addColorStop(0.55, `rgba(168, 85, 247, ${alpha * 0.22})`);
    grad.addColorStop(1, 'rgba(168, 85, 247, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(xPos - bandW * 0.5, 0, bandW, viewH);

    // Crisp curved Swipe Arc Indicator bowing forward in swipe direction
    const bow = sw.dir * viewW * 0.08;
    ctx.strokeStyle = `rgba(103, 232, 249, ${alpha * 0.72})`;
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(xPos - bow * 0.4, viewH * 0.14);
    ctx.quadraticCurveTo(xPos + bow, viewH * 0.5, xPos - bow * 0.4, viewH * 0.86);
    ctx.stroke();

    ctx.strokeStyle = `rgba(255, 255, 255, ${alpha * 0.92})`;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.restore();
  }

  /** Ethereal Prismatic Mist Dissolve when retracting hands in Glass / Thermal / Prism modes. */
  private drawPolyDissolves(ctx: CanvasRenderingContext2D, dt: number): void {
    if (this.polyDissolves.length === 0) return;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const alive: PolyDissolveGhost[] = [];

    for (let i = 0; i < this.polyDissolves.length; i += 1) {
      const g = this.polyDissolves[i];
      g.age += dt;
      if (g.age >= g.duration) continue;
      const t = g.age / g.duration;
      const alpha = Math.pow(1 - t, 1.7);
      const expand = 1 + t * 0.16;
      const liftY = -t * 18;

      // Expanding translucent chromatic silhouette
      ctx.beginPath();
      for (let k = 0; k < g.pts.length; k += 1) {
        const px = g.cx + (g.pts[k].x - g.cx) * expand;
        const py = g.cy + (g.pts[k].y - g.cy) * expand + liftY;
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.strokeStyle = g.color;
      ctx.lineWidth = 4.5 * (1 - t) + 0.8;
      ctx.globalAlpha = alpha * 0.55;
      ctx.stroke();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.2;
      ctx.globalAlpha = alpha * 0.72;
      ctx.stroke();

      // Rising shimmering prismatic dust motes
      for (let m = 0; m < g.motes.length; m += 1) {
        const pt = g.motes[m];
        pt.x += pt.vx * dt;
        pt.y += pt.vy * dt;
        pt.vy -= 36 * dt;
        ctx.fillStyle = m % 2 === 0 ? '#e0f2fe' : g.color;
        ctx.globalAlpha = alpha * 0.85;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 1.5 + (1 - t) * 1.8, 0, Math.PI * 2);
        ctx.fill();
      }

      alive.push(g);
    }

    this.polyDissolves = alive;
    ctx.restore();
  }
  /** Directional Swipe Indicator hugging the swiping hand during Phase 1 (Arm -> Commit). */
  private drawDirectionalSwipeIndicator(
    ctx: CanvasRenderingContext2D,
    ind: SwipeIndicatorState,
    viewW: number,
    viewH: number,
  ): void {
    if (!ind.active || ind.dir === 0 || ind.progress <= 0.04) return;
    const cx = ind.normX * viewW;
    const cy = ind.normY * viewH - 52;
    const dir = ind.dir;
    const p = Math.min(1, ind.progress);

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Subtle directional progress rail
    const barW = 74;
    const x0 = cx - barW * 0.5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x0, cy);
    ctx.lineTo(x0 + barW, cy);
    ctx.stroke();

    const fillStart = dir === 1 ? x0 : x0 + barW;
    const fillEnd = dir === 1 ? x0 + barW * p : x0 + barW * (1 - p);
    ctx.strokeStyle = `rgba(56, 189, 248, ${0.55 + p * 0.45})`;
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.moveTo(fillStart, cy);
    ctx.lineTo(fillEnd, cy);
    ctx.stroke();

    // Animated directional chevron pair pointing in swipe direction
    for (let k = 0; k < 2; k += 1) {
      const tipX = fillEnd + dir * (10 + k * 9);
      const a = (0.45 + p * 0.55) * (k === 0 ? 1 : 0.6);
      ctx.strokeStyle = `rgba(224, 242, 254, ${a})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(tipX - dir * 7, cy - 7);
      ctx.lineTo(tipX, cy);
      ctx.lineTo(tipX - dir * 7, cy + 7);
      ctx.stroke();
    }

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
    swipeIndicator?: SwipeIndicatorState,
  ): FrameOutput {
    this.elapsed += dt;
    this.ensureFboSize(viewW, viewH);
    this.cosmosFx?.setLightningSfxEnabled(settings.sfxEnabled);

    const gl = this.gl;
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

    // Resolve target mode
    let targetMode: CycleModeId = 'GLASS';
    if (settings.mode !== 'AUTO') {
      targetMode = settings.mode;
    } else if (hands.length <= 1) {
      targetMode = 'GLASS';
    } else {
      const g0 = hands[0].gesture;
      const g1 = hands[1].gesture;
      const pointingOrPinched =
        g0 === 'PINCH' || g0 === 'POINT' || g1 === 'PINCH' || g1 === 'POINT';
      targetMode = pointingOrPinched ? 'THERMAL' : 'PRISM';
    }

    // Unified 3-Phase Scene Transition Lifecycle (ENTER 280ms -> ACTIVE -> EXIT 220ms)
    const didSwitch = this.transitionMgr.setMode(targetMode);
    if (didSwitch) {
      if (targetMode === 'COSMOS') this.cosmosFx?.reset();
      if (targetMode === 'MYSTIC') this.mysticFx.reset();
      if (targetMode === 'LIGHTNING') this.cosmosFx?.resetLightning();
    }
    const trans = this.transitionMgr.update(dt);
    const activeMode = trans.activeMode;

    // Air-Canvas drawing
    if (
      settings.airDrawEnabled &&
      leadHand &&
      !isHoveringUi &&
      leadHand.pts[17] < viewH - 92
    ) {
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

    // Build incoming + outgoing optical polygons (skipped when Air Draw active)
    const livePolygons: RenderPolygon[] = [];
    if (!settings.airDrawEnabled) {
      if (trans.outgoingMode && trans.outgoingAlpha > 0.02) {
        const outPolys = buildOpticalPolygons(hands, trans.outgoingMode);
        for (let i = 0; i < outPolys.length; i += 1) {
          outPolys[i].tint[3] *= trans.outgoingAlpha;
          livePolygons.push(outPolys[i]);
        }
      }
      const inPolys = buildOpticalPolygons(hands, activeMode);
      for (let i = 0; i < inPolys.length; i += 1) {
        inPolys[i].tint[3] *= trans.activeAlpha;
        livePolygons.push(inPolys[i]);
      }
    }

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

    const targetEnhance = settings.hdDenoise ? 1.0 : 0.0;
    this.smoothEnhance += (targetEnhance - this.smoothEnhance) * (1 - Math.exp(-5.0 * dt));

    const writeIdx: 0 | 1 = this.fboReadIdx === 0 ? 1 : 0;

    // --- WebGL2 Pass 1: Cinema Clean (Temporal Denoise + Chroma Smooth + ACES + Vignette) into FBO ---
    gl.useProgram(this.program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.prevFbos[writeIdx]);
    gl.viewport(0, 0, this.fboW, this.fboH);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.uniform1i(this.locs.uVideo, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.prevTexs[this.fboReadIdx]);
    gl.uniform1i(this.locs.uPrevFrame, 1);

    gl.uniform2f(this.locs.uCoverScale, coverX, coverY);
    gl.uniform2f(this.locs.uVideoTexel, 1.0 / Math.max(videoW, 1), 1.0 / Math.max(videoH, 1));
    gl.uniform1f(this.locs.uMirrored, settings.mirrored ? 1.0 : 0.0);
    gl.uniform1f(this.locs.uEnhance, this.smoothEnhance);
    gl.uniform1f(this.locs.uHasPrev, this.hasPrevFrame ? 1.0 : 0.0);

    this.drawFullQuad(4);

    // --- WebGL2 Pass 2: Blit Cinema Clean FBO + Optical Polygons ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, viewW, viewH);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.prevTexs[writeIdx]);
    gl.uniform1i(this.locs.uPrevFrame, 1);

    this.drawFullQuad(5);

    for (let i = 0; i < livePolygons.length; i += 1) {
      this.drawPolygonWebGL(livePolygons[i], viewW, viewH);
    }

    // Unbind both texture units cleanly (ANGLE D3D11 zero feedback hazard)
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.fboReadIdx = writeIdx;
    this.hasPrevFrame = true;

    // --- Canvas 2D & Shared Three.js WebGL2 + UnrealBloomPass Overlay ---
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

      if (livePolygons.length > 0) {
        this.lastPolyCache = livePolygons.map((p) => ({
          pts: p.pts.map((v) => ({ x: v.x, y: v.y })),
          color: p.glow,
        }));
      } else if (this.lastPolyCache.length > 0) {
        // Hand retracted or swiped away -> shatter polygon into ~180 inertial particles over 550ms
        const perPoly = Math.max(80, Math.floor(190 / this.lastPolyCache.length));
        for (let i = 0; i < this.lastPolyCache.length; i += 1) {
          const cached = this.lastPolyCache[i];
          if (cached.pts.length < 3) continue;
          let cx = 0;
          let cy = 0;
          for (let k = 0; k < cached.pts.length; k += 1) {
            cx += cached.pts[k].x;
            cy += cached.pts[k].y;
          }
          cx /= cached.pts.length;
          cy /= cached.pts.length;
          const motes: { x: number; y: number; vx: number; vy: number }[] = [];
          for (let m = 0; m < perPoly; m += 1) {
            const pA = cached.pts[m % cached.pts.length];
            const pB = cached.pts[(m + 1) % cached.pts.length];
            const f = Math.random();
            const mx = pA.x + (pB.x - pA.x) * f;
            const my = pA.y + (pB.y - pA.y) * f;
            motes.push({
              x: mx,
              y: my,
              vx: (mx - cx) * 0.55 + (Math.random() - 0.5) * 85,
              vy: (my - cy) * 0.4 - 42 - Math.random() * 75,
            });
          }
          this.polyDissolves.push({
            pts: cached.pts,
            cx,
            cy,
            color: cached.color,
            age: 0,
            duration: 0.55,
            motes,
          });
        }
        this.lastPolyCache = [];
      }

      if (livePolygons.length > 0) {
        renderPrismChromaticEdges(
          overlayCtx,
          livePolygons,
          Math.max(trans.activeAlpha, trans.outgoingAlpha),
        );
      }
      this.drawPolyDissolves(overlayCtx, dt);

      if (!settings.airDrawEnabled) {
        // Shared Three.js Stage renders Cosmos AND/OR 3D Volumetric Lightning in ONE UnrealBloomPass
        const cosmosActive =
          activeMode === 'COSMOS' || (trans.outgoingMode === 'COSMOS' && trans.outgoingAlpha > 0.02);
        const cosmosFade =
          activeMode === 'COSMOS' ? trans.enterProgress : trans.outgoingAlpha;

        const lightningActive =
          activeMode === 'LIGHTNING' ||
          (trans.outgoingMode === 'LIGHTNING' && trans.outgoingAlpha > 0.02);
        const lightningFade =
          activeMode === 'LIGHTNING' ? trans.activeAlpha : trans.outgoingAlpha;
        const lightningEnter = activeMode === 'LIGHTNING' ? trans.enterProgress : 1;

        if (cosmosActive || lightningActive) {
          this.cosmosFx?.renderStage({
            cosmosActive,
            cosmosFade,
            lightningActive,
            lightningFade,
            lightningEnter,
            hands,
            dt,
            viewW,
            viewH,
            overlayCtx,
          });
        } else {
          this.cosmosFx?.clear();
        }

        // Mystic 2D Eldritch Mandala (supports parallel ENTER/EXIT crossfade)
        if (trans.outgoingMode === 'MYSTIC' && trans.outgoingAlpha > 0.02) {
          this.mysticFx.updateAndRender(overlayCtx, hands, dt, trans.outgoingAlpha);
        }
        if (activeMode === 'MYSTIC') {
          this.mysticFx.updateAndRender(overlayCtx, hands, dt, trans.enterProgress);
        }
      } else {
        this.cosmosFx?.clear();
      }

      if (settings.cyberHud && hands.length > 0) {
        drawCyberneticHandHud(overlayCtx, hands, activeMode, this.elapsed);
      }

      this.drawAirStrokes(overlayCtx);
      this.drawNoteBeams(overlayCtx, dt);
      if (swipeIndicator && swipeIndicator.active) {
        this.drawDirectionalSwipeIndicator(overlayCtx, swipeIndicator, viewW, viewH);
      }
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
    this.cosmosFx?.dispose();
    this.cosmosFx = null;
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.vbo);
    gl.deleteTexture(this.videoTex);
    this.prevTexs.forEach((t) => gl.deleteTexture(t));
    this.prevFbos.forEach((f) => gl.deleteFramebuffer(f));
  }
}
