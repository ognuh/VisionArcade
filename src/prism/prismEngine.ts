import { Cosmos3DEngine } from '../fx/cosmos';
import { MysticFx } from '../fx/mystic';
import {
  buildOpticalPolygons,
  renderPrismChromaticEdges,
  renderTrailOutline,
} from '../fx/prism';
import { RippleWaveSim, type RippleSimMeta } from '../fx/ripple';
import type { RunnerHudSnapshot } from '../fx/runnerGame';
import { TransitionManager } from '../fx/transitionManager';
import type { HandPose2D, Point2D, RenderPolygon } from '../fx/types';
import type {
  AirDrawColorId,
  CycleModeId,
  GestureType,
  GraphicsQuality,
  Landmark3D,
  PrismSettings,
} from '../types/vision';
import { BodyController, drawRunnerPipGuide } from '../vision/bodyController';
import type { SwipeIndicatorState } from '../vision/gestureManager';
import { OneEuroAxis } from '../vision/smoothing';
import { drawCyberneticHandHud } from './cyberHud';

export type { Point2D };

interface TrailFrame {
  outlines: { pts: Point2D[]; color: string }[];
  age: number;
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
const TRAIL_MAX_AGE_SEC = 0.23; // FPS-invariant motion trail window (identical at 60 & 120 FPS)

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
uniform sampler2D uRippleTex;
uniform vec2 uCoverScale;
uniform vec2 uVideoTexel;
uniform vec2 uRippleTexel;
uniform float uMirrored;
uniform float uRippleStrength;
uniform float uRippleBias;
uniform float uRippleInvScale;
uniform int uShaderType; // 0=BEVELED_GLASS_LENS, 1=THERMAL, 2=PRISM, 4=PASSTHROUGH_RIPPLE
uniform vec4 uTint;
uniform vec2 uLensVerts[4];
uniform int uLensCount;
uniform vec2 uLensCenter;
uniform float uViewAspect;

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

float sampleWaveHeight(vec2 rUv) {
  return (texture(uRippleTex, clamp(rUv, 0.002, 0.998)).r - uRippleBias) * uRippleInvScale;
}

void main() {
  vec2 uv = clamp(screenToVideoUv(vScreenUv), vec2(0.002), vec2(0.998));

  if (uShaderType == 4) {
    // Ultra-light pass-through webcam + True 16-bit Float Liquid Glass Reality Ripple
    if (uRippleStrength > 0.002) {
      vec2 rUv = vec2(vScreenUv.x, 1.0 - vScreenUv.y);
      vec2 t = uRippleTexel;

      // 9-tap smooth Sobel gradient for silky optical refraction normals
      float hTL = sampleWaveHeight(rUv + vec2(-t.x, -t.y));
      float hTC = sampleWaveHeight(rUv + vec2( 0.0, -t.y));
      float hTR = sampleWaveHeight(rUv + vec2( t.x, -t.y));
      float hML = sampleWaveHeight(rUv + vec2(-t.x,  0.0));
      float hC  = sampleWaveHeight(rUv);
      float hMR = sampleWaveHeight(rUv + vec2( t.x,  0.0));
      float hBL = sampleWaveHeight(rUv + vec2(-t.x,  t.y));
      float hBC = sampleWaveHeight(rUv + vec2( 0.0,  t.y));
      float hBR = sampleWaveHeight(rUv + vec2( t.x,  t.y));

      float gx = (hTR + 2.0 * hMR + hBR) - (hTL + 2.0 * hML + hBL);
      float gy = (hBL + 2.0 * hBC + hBR) - (hTL + 2.0 * hTC + hTR);
      vec2 grad = vec2(gx, gy) * 0.25;

      // Smooth deep UV refraction + soft R/G/B Chromatic Aberration along wave slope
      vec2 disp = grad * 0.122 * uRippleStrength;
      if (uMirrored > 0.5) disp.x = -disp.x;

      float rCh = texture(uVideo, clamp(uv + disp * 1.18, 0.002, 0.998)).r;
      float gCh = texture(uVideo, clamp(uv + disp,        0.002, 0.998)).g;
      float bCh = texture(uVideo, clamp(uv + disp * 0.82, 0.002, 0.998)).b;
      vec3 refracted = vec3(rCh, gCh, bCh);

      // Glossy Fresnel specular highlights + shimmering aquatic caustics on convex crests
      vec3 N = normalize(vec3(-grad.x * 9.2, -grad.y * 9.2, 1.0));
      vec3 V = vec3(0.0, 0.0, 1.0);
      vec3 L1 = normalize(vec3(-0.38, 0.48, 0.79));
      vec3 L2 = normalize(vec3(0.45, -0.35, 0.82));

      float fresnel = pow(1.0 - max(dot(N, V), 0.0), 2.6) * uRippleStrength;
      float spec1 = pow(max(dot(N, normalize(L1 + V)), 0.0), 72.0) * 0.92;
      float spec2 = pow(max(dot(N, normalize(L2 + V)), 0.0), 38.0) * 0.42;
      float spec = (spec1 + spec2) * uRippleStrength;

      float lap = max(0.0, hC * 4.0 - (hML + hMR + hTC + hBC));
      float caustics = (smoothstep(0.009, 0.11, lap) * 0.34 + smoothstep(0.035, 0.20, lap) * 0.22) * uRippleStrength;

      vec3 glassLight =
        vec3(0.96, 0.99, 1.0) * spec +
        vec3(0.42, 0.88, 1.0) * (fresnel * 0.44 + caustics);

      fragColor = vec4(clamp(refracted + glassLight, 0.0, 1.0), 1.0);
      return;
    }

    fragColor = vec4(texture(uVideo, uv).rgb, 1.0);
    return;
  }

  // Mode 0: 3D Beveled Liquid Glass Crystal Lens (Convex magnification + Frosted micro-blur + Beveled edge refraction)
  if (uShaderType == 0) {
    vec2 aspect = vec2(uViewAspect, 1.0);
    vec2 pA = vScreenUv * aspect;
    vec2 cA = uLensCenter * aspect;

    float minDist = 10.0;
    vec2 bestInNormal = vec2(0.0, 1.0);

    for (int i = 0; i < 4; i++) {
      if (i >= uLensCount) break;
      int nextIdx = (i + 1 == uLensCount) ? 0 : (i + 1);
      vec2 a = uLensVerts[i] * aspect;
      vec2 b = uLensVerts[nextIdx] * aspect;
      vec2 ab = b - a;
      float lenSq = max(dot(ab, ab), 0.00001);
      float proj = clamp(dot(pA - a, ab) / lenSq, 0.0, 1.0);
      vec2 closest = a + ab * proj;
      float d = length(pA - closest);

      if (d < minDist) {
        minDist = d;
        vec2 edgeDir = ab / sqrt(lenSq);
        vec2 n = vec2(-edgeDir.y, edgeDir.x);
        if (dot(cA - closest, n) < 0.0) n = -n;
        bestInNormal = n;
      }
    }

    // Smooth optical bevel ramp along the crystal borders (0 = clear convex dome, 1 = sharp beveled edge)
    float bevel = 1.0 - smoothstep(0.0, 0.052, minDist);
    float bevelSq = bevel * bevel;

    // 1. Subtle convex crystal magnification in the clear center
    vec2 toCenter = vScreenUv - uLensCenter;
    vec2 convexShift = -toCenter * 0.115 * (1.0 - bevel * 0.45);

    // 2. Strong beveled liquid-glass refraction bending light around the 4 edges
    vec2 bevelShift = (-bestInNormal / aspect) * bevelSq * 0.054;
    vec2 totalShift = convexShift + bevelShift;
    if (uMirrored > 0.5) totalShift.x = -totalShift.x;

    // 3. Prismatic chromatic aberration along the beveled crystal facets
    vec2 caShift = (-bestInNormal / aspect) * bevelSq * 0.013;
    if (uMirrored > 0.5) caShift.x = -caShift.x;

    vec2 baseUv = clamp(uv + totalShift, 0.002, 0.998);
    float rCh = texture(uVideo, clamp(uv + totalShift + caShift, 0.002, 0.998)).r;
    float gCh = texture(uVideo, baseUv).g;
    float bCh = texture(uVideo, clamp(uv + totalShift - caShift, 0.002, 0.998)).b;
    vec3 refracted = vec3(rCh, gCh, bCh);

    // 4. Crisp 9-tap circular VisionOS frosted optical bokeh micro-blur
    vec2 bOff = uVideoTexel * (1.8 + bevel * 1.6);
    vec2 bDiag = bOff * 0.7071;
    vec3 frosted =
      refracted * 0.28 +
      (texture(uVideo, clamp(baseUv + vec2( bOff.x,  0.0),     0.002, 0.998)).rgb +
       texture(uVideo, clamp(baseUv + vec2(-bOff.x,  0.0),     0.002, 0.998)).rgb +
       texture(uVideo, clamp(baseUv + vec2( 0.0,     bOff.y),  0.002, 0.998)).rgb +
       texture(uVideo, clamp(baseUv + vec2( 0.0,    -bOff.y),  0.002, 0.998)).rgb +
       texture(uVideo, clamp(baseUv + vec2( bDiag.x,  bDiag.y), 0.002, 0.998)).rgb +
       texture(uVideo, clamp(baseUv + vec2(-bDiag.x,  bDiag.y), 0.002, 0.998)).rgb +
       texture(uVideo, clamp(baseUv + vec2( bDiag.x, -bDiag.y), 0.002, 0.998)).rgb +
       texture(uVideo, clamp(baseUv + vec2(-bDiag.x, -bDiag.y), 0.002, 0.998)).rgb) * 0.09;
    vec3 glassSample = mix(refracted, frosted, 0.36 + bevel * 0.24);

    // 5. 3D bevel surface lighting, top-left specular catchlight & crisp white crystal rim sheen
    vec3 bevelN = normalize(vec3(-bestInNormal * bevel * 1.65, 1.0));
    vec3 lightDir = normalize(vec3(-0.45, -0.55, 0.70));
    float spec = pow(max(dot(bevelN, lightDir), 0.0), 42.0) * bevel * 0.62;
    float rimGlow = smoothstep(0.64, 0.99, bevel) * 0.42;
    float domeSheen = pow(max(0.0, 1.0 - length((vScreenUv - uLensCenter) * 2.3)), 2.0) * 0.075;

    vec3 crystal = glassSample * 1.06 + vec3(0.018, 0.032, 0.056) * (0.40 + bevel * 0.60);
    crystal += vec3(0.96, 0.99, 1.0) * (spec + rimGlow + domeSheen);
    fragColor = vec4(clamp(crystal, 0.0, 1.0), 0.98 * uTint.a);
    return;
  }

  // Optical Polygons read directly from raw mirrored webcam texture
  vec3 cam = texture(uVideo, uv).rgb;
  float luma = lumaOf(cam);

  if (uShaderType == 1) {
    // Pro Studio FLIR Thermal: Sobel edge contour + crisp isotherm shimmer
    vec2 t = uVideoTexel * 2.0;
    float lR = lumaOf(texture(uVideo, clamp(uv + vec2( t.x, 0.0), 0.002, 0.998)).rgb);
    float lL = lumaOf(texture(uVideo, clamp(uv + vec2(-t.x, 0.0), 0.002, 0.998)).rgb);
    float lU = lumaOf(texture(uVideo, clamp(uv + vec2(0.0,  t.y), 0.002, 0.998)).rgb);
    float lD = lumaOf(texture(uVideo, clamp(uv + vec2(0.0, -t.y), 0.002, 0.998)).rgb);
    float edge = clamp((abs(lR - lL) + abs(lU - lD)) * 3.4, 0.0, 1.0);
    float stretched = clamp((luma - 0.07) * 1.52 + edge * 0.24, 0.0, 1.0);
    float iso = smoothstep(0.93, 0.99, abs(sin(stretched * 25.1327))) * 0.08;
    vec3 thermal = clamp(thermalFalseColor(stretched) + vec3(iso * 0.6, iso * 0.9, iso), 0.0, 1.0);
    fragColor = vec4(thermal, 0.98 * uTint.a);
    return;
  }

  // Chromatic Prism Ribbon with subtle optical RGB spectral dispersion
  vec2 pShift = uVideoTexel * 3.0;
  float lR = lumaOf(texture(uVideo, clamp(uv + vec2( pShift.x, 0.0), 0.002, 0.998)).rgb);
  float lB = lumaOf(texture(uVideo, clamp(uv + vec2(-pShift.x, 0.0), 0.002, 0.998)).rgb);
  vec3 hiContrast = clamp((vec3(lR, luma, lB) - 0.46) * 1.52 + 0.66, 0.28, 1.38);
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
  private targetBuffers: [Float32Array, Float32Array] = [new Float32Array(42), new Float32Array(42)];
  private velBuffers: [Float32Array, Float32Array] = [new Float32Array(42), new Float32Array(42)];
  private filters: [OneEuroAxis[], OneEuroAxis[]];
  private activeFlags: [boolean, boolean] = [false, false];
  private missedSec: [number, number] = [0, 0];
  private lastPacketSeq = -1;
  private sincePacketSec = 0;
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
    this.lastPacketSeq = -1;
    this.sincePacketSec = 0;
    this.velBuffers[0].fill(0);
    this.velBuffers[1].fill(0);
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
    packetSeq = 0,
  ): void {
    outHands.length = 0;
    const safeDt = Math.min(Math.max(dt, 0.002), 0.08);
    const isFreshPacket = packetSeq !== this.lastPacketSeq;
    const aiDt = Math.min(0.12, Math.max(0.015, this.sincePacketSec > 0 ? this.sincePacketSec : 0.033));
    if (isFreshPacket) {
      this.lastPacketSeq = packetSeq;
      this.sincePacketSec = 0;
    } else {
      this.sincePacketSec = Math.min(0.12, this.sincePacketSec + safeDt);
    }

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

    // Gentle bounded 120Hz velocity lookahead between 30Hz webcam AI packets (zero overshoot)
    const predDt = Math.min(this.sincePacketSec, 0.028) * 0.45;
    const velDecay = this.sincePacketSec > 0.038 ? Math.exp(-12 * safeDt) : 1;

    for (let slot = 0; slot < 2; slot += 1) {
      const hand = assignedHand[slot];
      const gesture = assignedGesture[slot];
      const buf = this.buffers[slot];
      const tgt = this.targetBuffers[slot];
      const vel = this.velBuffers[slot];
      const flt = this.filters[slot];
      const pose = this.poses[slot];

      if (!hand) {
        // Hand Persistence: retain last smoothed pose up to 180ms with gentle confidence decay
        if (this.activeFlags[slot] && this.missedSec[slot] < HAND_PERSISTENCE_SEC) {
          this.missedSec[slot] += safeDt;
          const dropFrac = Math.min(1, this.missedSec[slot] / HAND_PERSISTENCE_SEC);
          pose.confidence = Math.max(0.35, 1 - dropFrac * 0.65);
          pose.palmSpeedMps *= Math.exp(-9.75 * safeDt);
          outHands.push(pose);
        } else {
          this.activeFlags[slot] = false;
          this.missedSec[slot] = 0;
          pose.confidence = 0;
          vel.fill(0);
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
        const ix = i * 2;
        const iy = ix + 1;

        if (!wasActive) {
          tgt[ix] = tx;
          tgt[iy] = ty;
          vel[ix] = 0;
          vel[iy] = 0;
          flt[ix].reset();
          flt[iy].reset();
          buf[ix] = flt[ix].filter(tx, safeDt, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA);
          buf[iy] = flt[iy].filter(ty, safeDt, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA);
        } else {
          if (isFreshPacket) {
            const instVx = Math.max(-2200, Math.min(2200, (tx - tgt[ix]) / aiDt));
            const instVy = Math.max(-2200, Math.min(2200, (ty - tgt[iy]) / aiDt));
            vel[ix] = vel[ix] * 0.35 + instVx * 0.65;
            vel[iy] = vel[iy] * 0.35 + instVy * 0.65;
            tgt[ix] = tx;
            tgt[iy] = ty;
          } else if (velDecay < 1) {
            vel[ix] *= velDecay;
            vel[iy] *= velDecay;
          }
          const predX = tgt[ix] + vel[ix] * predDt;
          const predY = tgt[iy] + vel[iy] * predDt;
          buf[ix] = flt[ix].filter(predX, safeDt, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA);
          buf[iy] = flt[iy].filter(predY, safeDt, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA);
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
      const speedBlend = 1 - Math.exp(-17.3 * safeDt);
      pose.palmSpeedMps += (instantMps - pose.palmSpeedMps) * speedBlend;

      pose.palmX = palmX;
      pose.palmY = palmY;
      pose.palmScale = Math.max(palmScale, 36);
      pose.gesture = gesture;
      outHands.push(pose);
    }
  }
}

/**
 * 120Hz Elastic Spring-Physics Smoother for the 3D Beveled Liquid Glass Lens vertices:
 * Delivers tactile glass inertia & elastic follow when stretching/rotating between 2 hands.
 */
class GlassLensSpring {
  private x = new Float32Array(4);
  private y = new Float32Array(4);
  private vx = new Float32Array(4);
  private vy = new Float32Array(4);
  private activeCount = 0;

  public reset(): void {
    this.activeCount = 0;
    this.vx.fill(0);
    this.vy.fill(0);
  }

  public update(targetPts: Point2D[], dt: number): Point2D[] {
    const n = Math.min(4, targetPts.length);
    if (n < 3) {
      this.activeCount = 0;
      return targetPts;
    }

    const safeDt = Math.min(Math.max(dt, 0.002), 0.033);
    if (this.activeCount !== n) {
      for (let i = 0; i < n; i += 1) {
        this.x[i] = targetPts[i].x;
        this.y[i] = targetPts[i].y;
        this.vx[i] = 0;
        this.vy[i] = 0;
      }
      this.activeCount = n;
    } else {
      // Critically-underdamped spring (stiffness=220, damping=22) — pure dt-normalized at 120Hz
      const stiffness = 220;
      const damping = 22;
      for (let i = 0; i < n; i += 1) {
        const ax = (targetPts[i].x - this.x[i]) * stiffness - this.vx[i] * damping;
        const ay = (targetPts[i].y - this.y[i]) * stiffness - this.vy[i] * damping;
        this.vx[i] += ax * safeDt;
        this.vy[i] += ay * safeDt;
        this.x[i] += this.vx[i] * safeDt;
        this.y[i] += this.vy[i] * safeDt;
      }
    }

    const out: Point2D[] = [];
    for (let i = 0; i < n; i += 1) {
      out.push({ x: this.x[i], y: this.y[i] });
    }
    return out;
  }
}

export class OpticalPrismEngine {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private vbo: WebGLBuffer;
  private videoTex: WebGLTexture;

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
  private rippleSim: RippleWaveSim;
  private bodyController = new BodyController();
  private glassSpring = new GlassLensSpring();
  private lensVertScratch = new Float32Array(8);
  private rippleStrength = 0;

  private trailBuffer: TrailFrame[] = [];
  private airStrokes: AirStroke[] = [];
  private activeStroke: AirStroke | null = null;
  private noteBeams: NoteSparkBeam[] = [];
  private swipeSwoosh: SwipeSwoosh | null = null;
  private lastPolyCache: { pts: Point2D[]; color: string }[] = [];
  private polyDissolves: PolyDissolveGhost[] = [];

  private transitionMgr = new TransitionManager();
  private elapsed = 0;

  private locs: {
    uVideo: WebGLUniformLocation | null;
    uRippleTex: WebGLUniformLocation | null;
    uCoverScale: WebGLUniformLocation | null;
    uVideoTexel: WebGLUniformLocation | null;
    uRippleTexel: WebGLUniformLocation | null;
    uMirrored: WebGLUniformLocation | null;
    uRippleStrength: WebGLUniformLocation | null;
    uRippleBias: WebGLUniformLocation | null;
    uRippleInvScale: WebGLUniformLocation | null;
    uShaderType: WebGLUniformLocation | null;
    uTint: WebGLUniformLocation | null;
    uLensVerts: WebGLUniformLocation | null;
    uLensCount: WebGLUniformLocation | null;
    uLensCenter: WebGLUniformLocation | null;
    uViewAspect: WebGLUniformLocation | null;
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
      uRippleTex: gl.getUniformLocation(this.program, 'uRippleTex'),
      uCoverScale: gl.getUniformLocation(this.program, 'uCoverScale'),
      uVideoTexel: gl.getUniformLocation(this.program, 'uVideoTexel'),
      uRippleTexel: gl.getUniformLocation(this.program, 'uRippleTexel'),
      uMirrored: gl.getUniformLocation(this.program, 'uMirrored'),
      uRippleStrength: gl.getUniformLocation(this.program, 'uRippleStrength'),
      uRippleBias: gl.getUniformLocation(this.program, 'uRippleBias'),
      uRippleInvScale: gl.getUniformLocation(this.program, 'uRippleInvScale'),
      uShaderType: gl.getUniformLocation(this.program, 'uShaderType'),
      uTint: gl.getUniformLocation(this.program, 'uTint'),
      uLensVerts: gl.getUniformLocation(this.program, 'uLensVerts[0]'),
      uLensCount: gl.getUniformLocation(this.program, 'uLensCount'),
      uLensCenter: gl.getUniformLocation(this.program, 'uLensCenter'),
      uViewAspect: gl.getUniformLocation(this.program, 'uViewAspect'),
    };

    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    if (!vao || !vbo) throw new Error('Failed to initialize WebGL2 buffers.');

    this.vao = vao;
    this.vbo = vbo;
    this.videoTex = this.createLinearTex();
    this.rippleSim = new RippleWaveSim(gl);

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
  }

  private createLinearTex(): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture();
    if (!tex) throw new Error('Failed to create WebGL2 texture.');
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  resetSmoothing(): void {
    this.interpolator.reset();
    this.bodyController.reset();
    this.cosmosFx?.reset();
    this.cosmosFx?.resetLightning();
    this.cosmosFx?.resetFluid();
    this.mysticFx.reset();
    this.rippleSim.clear();
    this.glassSpring.reset();
    this.trailBuffer = [];
  }

  setQuality(quality: GraphicsQuality): void {
    this.rippleSim.setQuality(quality);
    this.cosmosFx?.setQuality(quality);
  }

  getFluidParticleCount(): number {
    return this.cosmosFx?.getFluidParticleCount() ?? 12000;
  }

  startRunner(skipCountdown = false): void {
    this.cosmosFx?.startRunner(() => this.bodyController.beginCalibrationSample(), skipCountdown);
  }

  recalibrateRunner(): void {
    this.bodyController.recalibrateNow();
  }

  resetRunner(): void {
    this.cosmosFx?.resetRunner();
  }

  getRunnerSnapshot(): RunnerHudSnapshot | null {
    return this.cosmosFx?.getRunnerSnapshot() ?? null;
  }

  async loadCustomGlbFromFile(file: File): Promise<void> {
    if (!this.cosmosFx) return;
    const buf = await file.arrayBuffer();
    await this.cosmosFx.loadCustomGlbFromBuffer(buf, file.name.replace(/\.glb$/i, ''));
  }

  clearCustomGlb(): void {
    this.cosmosFx?.clearCustomGlb();
  }

  getRippleMeta(): RippleSimMeta {
    return this.rippleSim.getMeta();
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

    if (poly.shaderType === 0) {
      const count = Math.min(4, pts.length);
      let cx = 0;
      let cy = 0;
      for (let i = 0; i < count; i += 1) {
        const u = pts[i].x / viewW;
        const v = pts[i].y / viewH;
        this.lensVertScratch[i * 2] = u;
        this.lensVertScratch[i * 2 + 1] = v;
        cx += u;
        cy += v;
      }
      gl.uniform2fv(this.locs.uLensVerts, this.lensVertScratch);
      gl.uniform1i(this.locs.uLensCount, count);
      gl.uniform2f(this.locs.uLensCenter, cx / Math.max(count, 1), cy / Math.max(count, 1));
      gl.uniform1f(this.locs.uViewAspect, viewW / Math.max(viewH, 1));
    }

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
    handPacketSeq = 0,
    rawPose: Landmark3D[] | null = null,
    pipCtx: CanvasRenderingContext2D | null = null,
    videoEl: HTMLVideoElement | null = null,
  ): FrameOutput {
    this.elapsed += dt;
    this.cosmosFx?.setLightningSfxEnabled(settings.sfxEnabled);
    this.setQuality(settings.graphicsQuality);

    const gl = this.gl;
    if (overlayCtx) {
      overlayCtx.clearRect(0, 0, viewW, viewH);
    }

    // Exclusive Full-Body 3D Neon Subway Runner Mode (zero wasted hand-VFX work)
    if (settings.mode === 'RUNNER' && !settings.airDrawEnabled) {
      this.transitionMgr.setMode('RUNNER');
      this.transitionMgr.update(dt);
      const snapBefore = this.cosmosFx?.getRunnerSnapshot();
      const allowAutoIdle = !snapBefore || snapBefore.phase === 'READY';
      const bodyState = this.bodyController.update(rawPose, dt, settings.mirrored, allowAutoIdle);

      this.cosmosFx?.renderStage({
        cosmosActive: false,
        cosmosFade: 0,
        lightningActive: false,
        lightningFade: 0,
        lightningEnter: 0,
        fluidActive: false,
        fluidFade: 0,
        runnerActive: true,
        bodyState,
        onBeginRunnerCalibration: () => this.bodyController.beginCalibrationSample(),
        onFinishRunnerCalibration: () => this.bodyController.finishCalibrationSample(),
        hands: [],
        dt,
        viewW,
        viewH,
        overlayCtx,
      });

      if (pipCtx) {
        drawRunnerPipGuide(
          pipCtx,
          videoEl,
          bodyState,
          settings.mirrored,
          pipCtx.canvas.width,
          pipCtx.canvas.height,
        );
      }

      return {
        activeMode: 'RUNNER',
        indexTipPx: null,
        palmScreenNorm: null,
        isLeadFist: false,
      };
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
      handPacketSeq,
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
      if (targetMode === 'FLUID') this.cosmosFx?.resetFluid();
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
    let hasGlassLens = false;
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
        if (inPolys[i].shaderType === 0) {
          // 120Hz Elastic Spring Physics for Beveled Liquid Glass Lens vertices
          inPolys[i].pts = this.glassSpring.update(inPolys[i].pts, dt);
          hasGlassLens = true;
        }
        inPolys[i].tint[3] *= trans.activeAlpha;
        livePolygons.push(inPolys[i]);
      }
    }
    if (!hasGlassLens) {
      this.glassSpring.reset();
    }

    // Time-based motion trail buffer (identical visual duration at 60Hz and 120Hz)
    if (!settings.trailEnabled) {
      this.trailBuffer.length = 0;
    } else {
      for (let i = this.trailBuffer.length - 1; i >= 0; i -= 1) {
        this.trailBuffer[i].age += dt;
        if (this.trailBuffer[i].age >= TRAIL_MAX_AGE_SEC) {
          this.trailBuffer.splice(i, 1);
        }
      }
      if (livePolygons.length > 0) {
        this.trailBuffer.push({
          outlines: livePolygons.map((p) => ({
            pts: p.pts.map((v) => ({ x: v.x, y: v.y })),
            color: p.glow,
          })),
          age: 0,
        });
      }
    }

    // Smooth Reality Ripple wave simulation + 250ms flatten decay when leaving RIPPLE
    const isRippleTarget = !settings.airDrawEnabled && activeMode === 'RIPPLE';
    if (isRippleTarget) {
      this.rippleStrength = Math.min(1, this.rippleStrength + dt * 6.5);
    } else if (this.rippleStrength > 0) {
      this.rippleStrength = Math.max(0, this.rippleStrength - dt / 0.25);
      if (this.rippleStrength === 0) {
        this.rippleSim.clear();
      }
    }

    if (this.rippleStrength > 0.002) {
      this.rippleSim.step(isRippleTarget ? hands : [], dt, viewW, viewH);
    }

    // --- Single-Pass WebGL2 Webcam Pass-Through (+ Liquid Glass Ripple) + Optical Polygons ---
    gl.useProgram(this.program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.uniform1i(this.locs.uVideo, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.rippleSim.getHeightTexture());
    gl.uniform1i(this.locs.uRippleTex, 1);

    const [rBias, rInvScale] = this.rippleSim.getBiasAndInvScale();
    const [rTexX, rTexY] = this.rippleSim.getTexelSize();

    gl.uniform2f(this.locs.uCoverScale, coverX, coverY);
    gl.uniform2f(this.locs.uVideoTexel, 1.0 / Math.max(videoW, 1), 1.0 / Math.max(videoH, 1));
    gl.uniform2f(this.locs.uRippleTexel, rTexX, rTexY);
    gl.uniform1f(this.locs.uMirrored, settings.mirrored ? 1.0 : 0.0);
    gl.uniform1f(this.locs.uRippleStrength, this.rippleStrength);
    gl.uniform1f(this.locs.uRippleBias, rBias);
    gl.uniform1f(this.locs.uRippleInvScale, rInvScale);

    this.drawFullQuad(4);

    for (let i = 0; i < livePolygons.length; i += 1) {
      this.drawPolygonWebGL(livePolygons[i], viewW, viewH);
    }

    // Unbind both texture units cleanly
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // --- Canvas 2D & Shared Three.js WebGL2 + UnrealBloomPass Overlay ---
    if (overlayCtx) {
      if (settings.trailEnabled && this.trailBuffer.length > 1) {
        overlayCtx.save();
        overlayCtx.globalCompositeOperation = 'lighter';
        const len = this.trailBuffer.length;
        for (let i = 0; i < len - 1; i += 1) {
          const frame = this.trailBuffer[i];
          const lifeRem = Math.max(0, 1 - frame.age / TRAIL_MAX_AGE_SEC);
          const alpha = lifeRem * lifeRem * 0.52;
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
        // Shared Three.js Stage renders Cosmos, 3D Volumetric Lightning, AND 8,000-Particle Fluid in ONE UnrealBloomPass
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

        const fluidActive =
          activeMode === 'FLUID' ||
          (trans.outgoingMode === 'FLUID' && trans.outgoingAlpha > 0.02);
        const fluidFade =
          activeMode === 'FLUID' ? trans.activeAlpha : trans.outgoingAlpha;

        if (cosmosActive || lightningActive || fluidActive) {
          this.cosmosFx?.renderStage({
            cosmosActive,
            cosmosFade,
            lightningActive,
            lightningFade,
            lightningEnter,
            fluidActive,
            fluidFade,
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
    this.rippleSim.dispose();
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.vbo);
    gl.deleteTexture(this.videoTex);
  }
}
