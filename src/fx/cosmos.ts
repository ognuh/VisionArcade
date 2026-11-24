import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { FluidParticleSandbox } from './fluidParticles';
import { Lightning3DEngine } from './lightning3d';
import { elasticScale02To10 } from './transitionManager';
import type { HandPose2D } from './types';

interface PlanetSpec {
  id: string;
  tag: string;
  radiusMul: number;
  baseSpin: number;
  tiltZ: number;
  atmoColor: number;
  atmoIntensity: number;
}

const PLANETS: PlanetSpec[] = [
  {
    id: 'EARTH',
    tag: '01 // EARTH - 12,742 KM',
    radiusMul: 1.0,
    baseSpin: 0.42,
    tiltZ: 0.41,
    atmoColor: 0x38bdf8,
    atmoIntensity: 1.25,
  },
  {
    id: 'MARS',
    tag: '02 // MARS - 6,779 KM',
    radiusMul: 0.88,
    baseSpin: 0.36,
    tiltZ: 0.44,
    atmoColor: 0xfb923c,
    atmoIntensity: 0.82,
  },
  {
    id: 'SATURN',
    tag: '03 // SATURN - 116,460 KM',
    radiusMul: 0.94,
    baseSpin: 0.56,
    tiltZ: 0.46,
    atmoColor: 0xfde68a,
    atmoIntensity: 0.58,
  },
  {
    id: 'JUPITER',
    tag: '04 // JUPITER - 139,820 KM',
    radiusMul: 1.12,
    baseSpin: 0.68,
    tiltZ: 0.06,
    atmoColor: 0xfdba74,
    atmoIntensity: 0.62,
  },
  {
    id: 'NEPTUNE',
    tag: '05 // NEPTUNE - 49,244 KM',
    radiusMul: 0.96,
    baseSpin: 0.52,
    tiltZ: 0.49,
    atmoColor: 0x22d3ee,
    atmoIntensity: 1.15,
  },
  {
    id: 'SUN',
    tag: '06 // THE SUN - 1,392,700 KM',
    radiusMul: 1.16,
    baseSpin: 0.24,
    tiltZ: 0.12,
    atmoColor: 0xf97316,
    atmoIntensity: 1.5,
  },
];

const SWITCH_COOLDOWN_SEC = 0.85;
const SWITCH_DURATION_SEC = 0.65; // 650ms GLSL Simplex Dissolve -> Coalesce planet transition
const PLANET_DISSOLVE_SEC = 0.65; // 650ms Ethereal Dissolve when retracting hands
const HUD_DISPLAY_SEC = 2.6;
const BURST_PARTICLE_COUNT = 450; // 450 3D Curl-Noise Stardust Particles
const BH_STAR_COUNT = 1500;
const CAMERA_FOV_DEG = 42;
const CAMERA_Z = 10;

// Anti-accidental Black Hole gating thresholds
const BH_MIN_OPEN_FINGERS = 4;
const BH_MIN_NORM_DIST = 0.28;
const BH_CHARGE_SEC = 0.45;
const BH_EXIT_HYSTERESIS_SEC = 0.4;

// ============================================================================
// Seamless 3D 6-Octave Fractal Brownian Motion (FBM) + Domain Warping
// ============================================================================
const PERM = new Uint8Array(512);
(() => {
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i += 1) p[i] = i;
  let s = 133791;
  for (let i = 255; i > 0; i -= 1) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const j = s % (i + 1);
    const tmp = p[i];
    p[i] = p[j];
    p[j] = tmp;
  }
  for (let i = 0; i < 512; i += 1) PERM[i] = p[i & 255];
})();

function hash3(ix: number, iy: number, iz: number): number {
  return PERM[(PERM[(PERM[ix & 255] + iy) & 255] + iz) & 255] * (1 / 255);
}

function noise3(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const uz = fz * fz * (3 - 2 * fz);

  const n000 = hash3(ix, iy, iz);
  const n100 = hash3(ix + 1, iy, iz);
  const n010 = hash3(ix, iy + 1, iz);
  const n110 = hash3(ix + 1, iy + 1, iz);
  const n001 = hash3(ix, iy, iz + 1);
  const n101 = hash3(ix + 1, iy, iz + 1);
  const n011 = hash3(ix, iy + 1, iz + 1);
  const n111 = hash3(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;
  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return nxy0 + (nxy1 - nxy0) * uz;
}

function fbm3(x: number, y: number, z: number, octaves: number): number {
  let sum = 0;
  let amp = 0.5;
  let freq = 1;
  let norm = 0;
  for (let i = 0; i < octaves; i += 1) {
    sum += noise3(x * freq, y * freq, z * freq) * amp;
    norm += amp;
    freq *= 2.03;
    amp *= 0.5;
  }
  return sum / norm;
}

/** 6-Octave Domain-Warped FBM for realistic tectonic coastlines, mountains, and Jovian vortices. */
function warpedFbm6(x: number, y: number, z: number, warpStrength: number): number {
  const qx = fbm3(x + 1.7, y + 9.2, z + 3.1, 3) - 0.5;
  const qy = fbm3(x + 8.3, y + 2.8, z + 5.4, 3) - 0.5;
  const qz = fbm3(x + 4.1, y + 6.7, z + 1.9, 3) - 0.5;
  return fbm3(
    x + qx * warpStrength,
    y + qy * warpStrength,
    z + qz * warpStrength,
    6,
  );
}

function clampByte(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}

function makeCanvasTexture(
  canvas: HTMLCanvasElement,
  isColor: boolean,
): THREE.CanvasTexture {
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.colorSpace = isColor ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  return tex;
}

// ============================================================================
// Alpha-Preserving Post-Processing Passes (UnrealBloomPass over Live Webcam)
// ============================================================================
const SCENE_FEED_SHADER = {
  uniforms: {
    tScene: { value: null as THREE.Texture | null },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tScene;
    varying vec2 vUv;
    void main() {
      vec4 s = texture2D(tScene, vUv);
      // Premultiply RGB by alpha before feeding into UnrealBloomPass so transparent bg is pure 0
      gl_FragColor = vec4(s.rgb * s.a, s.a);
    }
  `,
};

const ALPHA_OUTPUT_SHADER = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    tSceneAlpha: { value: null as THREE.Texture | null },
    uExposure: { value: 1.08 },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform sampler2D tSceneAlpha;
    uniform float uExposure;
    varying vec2 vUv;

    vec3 acesToneMap(vec3 x) {
      const float a = 2.51;
      const float b = 0.03;
      const float c = 2.43;
      const float d = 0.59;
      const float e = 0.14;
      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }

    void main() {
      vec4 sceneRaw = texture2D(tSceneAlpha, vUv);
      vec3 bloomed = texture2D(tDiffuse, vUv).rgb;

      // Prevent inward bloom washout over pitch-black Event Horizon & deep night shadows
      float sceneLuma = dot(sceneRaw.rgb, vec3(0.299, 0.587, 0.114));
      float darkCoreMask = smoothstep(0.06, 0.0, sceneLuma) * smoothstep(0.88, 0.99, sceneRaw.a);
      vec3 hdr = mix(bloomed, bloomed * 0.14, darkCoreMask) * uExposure;

      vec3 mapped = acesToneMap(hdr);
      vec3 srgb = pow(mapped, vec3(1.0 / 2.2));

      float bloomAlpha = clamp(max(srgb.r, max(srgb.g, srgb.b)) * 1.18, 0.0, 1.0);
      float finalAlpha = clamp(max(sceneRaw.a, bloomAlpha), 0.0, 1.0);

      if (finalAlpha < 0.004) discard;
      gl_FragColor = vec4(srgb, finalAlpha);
    }
  `,
};

// ============================================================================
// GLSL Shaders: Atmospheric Scattering, Earth PBR + Cloud Shadow, Saturn Rings, Sun, Gargantua
// ============================================================================
const FRESNEL_VERT = `
varying vec3 vNormalView;
varying vec3 vViewDir;
varying vec3 vNormalWorld;

void main() {
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  vNormalView = normalize(normalMatrix * normal);
  vViewDir = normalize(-mvPos.xyz);
  vNormalWorld = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * mvPos;
}
`;

const FRESNEL_FRAG = `
uniform vec3 uColor;
uniform vec3 uLightDir;
uniform float uIntensity;
uniform float uPower;
uniform float uOpacity;
varying vec3 vNormalView;
varying vec3 vViewDir;
varying vec3 vNormalWorld;

void main() {
  float ndotv = clamp(dot(normalize(vNormalView), normalize(vViewDir)), 0.0, 1.0);
  float sunWrap = smoothstep(-0.22, 0.55, dot(normalize(vNormalWorld), normalize(uLightDir)));
  float outerSoft = smoothstep(0.0, 0.28, ndotv);
  float rim = pow(1.0 - ndotv, uPower) * outerSoft * uIntensity * (0.12 + 0.96 * sunWrap);
  vec3 col = uColor * rim;
  gl_FragColor = vec4(col, clamp(rim * 0.82 * uOpacity, 0.0, 1.0));
}
`;

// Shared GLSL 3D Simplex Noise snippet for Ethereal Dissolve shaders
const GLSL_DISSOLVE_NOISE = `
float vaHash31(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}
float vaNoise3(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(vaHash31(i + vec3(0,0,0)), vaHash31(i + vec3(1,0,0)), f.x),
        mix(vaHash31(i + vec3(0,1,0)), vaHash31(i + vec3(1,1,0)), f.x), f.y),
    mix(mix(vaHash31(i + vec3(0,0,1)), vaHash31(i + vec3(1,0,1)), f.x),
        mix(vaHash31(i + vec3(0,1,1)), vaHash31(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float vaDissolveFbm(vec3 p) {
  float v = 0.0;
  float a = 0.55;
  for (int i = 0; i < 3; i++) {
    v += a * vaNoise3(p);
    p = p * 2.12 + vec3(1.7, 3.1, 2.3);
    a *= 0.5;
  }
  return clamp(v / 0.96, 0.0, 1.0);
}
`;

function injectDissolveOnStandardMat(
  mat: THREE.MeshStandardMaterial,
  dissolveUniform: { value: number },
): void {
  mat.customProgramCacheKey = () => 'va-dissolve-std-v1';
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uDissolve = dissolveUniform;
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        '#include <common>\nvarying vec3 vDissolvePos;',
      )
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvDissolvePos = position;',
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>\nuniform float uDissolve;\nvarying vec3 vDissolvePos;\n${GLSL_DISSOLVE_NOISE}`,
      )
      .replace(
        '#include <dithering_fragment>',
        `
        if (uDissolve > 0.002) {
          float dn = vaDissolveFbm(vDissolvePos * 3.8);
          if (dn < uDissolve) discard;
          float edge = pow(smoothstep(uDissolve + 0.085, uDissolve, dn), 1.35);
          vec3 burnCol = mix(vec3(0.18, 1.18, 1.88), vec3(1.68, 1.15, 0.34), edge);
          gl_FragColor.rgb += burnCol * edge * 0.95;
        }
        #include <dithering_fragment>
        `,
      );
  };
}

const EARTH_VERT = `
varying vec2 vUv;
varying vec3 vNormalWorld;
varying vec3 vPosWorld;
varying vec3 vLocalPos;

void main() {
  vUv = uv;
  vLocalPos = position;
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vPosWorld = worldPos.xyz;
  vNormalWorld = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const EARTH_FRAG = `
uniform sampler2D uAlbedo;
uniform sampler2D uSpecular;
uniform sampler2D uNight;
uniform sampler2D uBump;
uniform sampler2D uClouds;
uniform float uCloudOffsetU;
uniform vec3 uLightDir;
uniform float uOpacity;
uniform float uDissolve;

varying vec2 vUv;
varying vec3 vNormalWorld;
varying vec3 vPosWorld;
varying vec3 vLocalPos;
${GLSL_DISSOLVE_NOISE}

void main() {
  float dissolveEdge = 0.0;
  if (uDissolve > 0.002) {
    float dn = vaDissolveFbm(vLocalPos * 4.0);
    if (dn < uDissolve) discard;
    dissolveEdge = pow(smoothstep(uDissolve + 0.085, uDissolve, dn), 1.35);
  }

  vec3 N = normalize(vNormalWorld);
  vec3 L = normalize(uLightDir);
  vec3 V = normalize(cameraPosition - vPosWorld);

  // Tangent-space-like micro relief from 6-octave bump height map
  float hC = texture2D(uBump, vUv).r;
  float hR = texture2D(uBump, vUv + vec2(0.002, 0.0)).r;
  float hU = texture2D(uBump, vUv + vec2(0.0, 0.0035)).r;
  vec3 Nb = normalize(N + vec3(hC - hR, hC - hU, 0.0) * 1.9);

  vec3 albedo = texture2D(uAlbedo, vUv).rgb;
  float oceanMask = texture2D(uSpecular, vUv).r;
  vec3 nightLights = texture2D(uNight, vUv).rgb;

  // Soft projected shadow from floating 3D cloud deck onto Earth's surface
  vec2 cloudShadowUv = vec2(fract(vUv.x + uCloudOffsetU - 0.005), clamp(vUv.y + 0.004, 0.0, 1.0));
  float cloudCover = texture2D(uClouds, cloudShadowUv).a;
  float cloudShadow = 1.0 - cloudCover * 0.34;

  float ndotl = dot(Nb, L);
  float dayFactor = smoothstep(-0.08, 0.20, ndotl);
  float diffuse = max(ndotl, 0.0) * cloudShadow;

  // Sparkling sunlight glint on oceans
  vec3 H = normalize(L + V);
  float spec = pow(max(dot(Nb, H), 0.0), 54.0) * oceanMask * 1.35 * dayFactor * cloudShadow;

  // Warm sunset scattering right across the sharp day/night terminator
  float terminator = exp(-pow((ndotl + 0.01) * 10.5, 2.0)) * 0.24;
  vec3 sunsetCol = vec3(1.0, 0.38, 0.12) * terminator;

  // Deep shadow night side (ultra-low ambient 0.03) with glowing amber city lights
  float nightMask = smoothstep(0.08, -0.18, ndotl);
  vec3 cityGlow = nightLights * nightMask * 1.85;

  // Rayleigh blue atmospheric scattering toward limb on sunlit side
  float fres = pow(1.0 - max(dot(N, V), 0.0), 3.2);
  vec3 atmoScatter = vec3(0.22, 0.66, 1.0) * fres * (0.14 + 0.96 * dayFactor);

  vec3 dayLit = albedo * (0.025 + diffuse * 1.32) + vec3(1.0, 0.95, 0.86) * spec;
  vec3 finalCol = dayLit + sunsetCol + cityGlow + atmoScatter;

  if (dissolveEdge > 0.001) {
    vec3 burn = mix(vec3(0.18, 1.22, 1.92), vec3(1.68, 1.15, 0.34), dissolveEdge);
    finalCol += burn * dissolveEdge * 0.95;
  }

  gl_FragColor = vec4(finalCol, uOpacity);
}
`;

const SATURN_RING_VERT = `
varying vec2 vUv;
varying vec3 vLocalPos;

void main() {
  vUv = uv;
  vLocalPos = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const SATURN_RING_FRAG = `
uniform sampler2D uRingTex;
uniform vec3 uLocalLightDir;
uniform float uOpacity;
uniform float uDissolve;
varying vec2 vUv;
varying vec3 vLocalPos;
${GLSL_DISSOLVE_NOISE}

void main() {
  vec4 tex = texture2D(uRingTex, vec2(vUv.x, 0.5));
  if (tex.a < 0.01) discard;

  float dissolveEdge = 0.0;
  if (uDissolve > 0.002) {
    float dn = vaDissolveFbm(vLocalPos * 2.5);
    if (dn < uDissolve) discard;
    dissolveEdge = pow(smoothstep(uDissolve + 0.085, uDissolve, dn), 1.35);
  }

  // Crisp cylindrical shadow cast by Saturn's sphere across the far side of the rings
  vec3 p = vLocalPos;
  vec3 L = normalize(uLocalLightDir);
  float proj = dot(p, L);
  float shadow = 1.0;
  if (proj < 0.0) {
    vec3 perp = p - L * proj;
    float distToAxis = length(perp);
    shadow = smoothstep(0.95, 1.03, distToAxis);
    shadow = mix(0.08, 1.0, shadow);
  }

  vec3 col = tex.rgb * shadow * 1.15;
  if (dissolveEdge > 0.001) {
    col += mix(vec3(0.18, 1.18, 1.88), vec3(1.68, 1.15, 0.34), dissolveEdge) * dissolveEdge * 0.95;
  }
  gl_FragColor = vec4(col, tex.a * uOpacity);
}
`;

const SUN_VERT = `
varying vec3 vPos;
varying vec3 vNormalView;
varying vec3 vViewDir;

void main() {
  vPos = normalize(position);
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  vNormalView = normalize(normalMatrix * normal);
  vViewDir = normalize(-mvPos.xyz);
  gl_Position = projectionMatrix * mvPos;
}
`;

const SUN_FRAG = `
uniform float uTime;
uniform float uOpacity;
uniform float uDissolve;
varying vec3 vPos;
varying vec3 vNormalView;
varying vec3 vViewDir;

vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod(i, 289.0);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 1.0/7.0;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

void main() {
  vec3 p = vPos * 3.4;
  float n1 = snoise(p + vec3(uTime * 0.16, -uTime * 0.11, uTime * 0.07));
  float n2 = snoise(p * 2.4 + vec3(-uTime * 0.30, uTime * 0.24, n1 * 0.65));
  float n3 = snoise(p * 5.6 + vec3(uTime * 0.44, n2 * 0.55, -uTime * 0.28));
  float plasma = clamp(0.52 + n1 * 0.32 + n2 * 0.22 + n3 * 0.14, 0.0, 1.0);

  float spots = smoothstep(0.18, 0.38, plasma);
  vec3 colDark = vec3(0.28, 0.02, 0.005);
  vec3 colMid  = vec3(0.92, 0.22, 0.02);
  vec3 colHot  = vec3(1.24, 0.68, 0.14);
  vec3 colCore = vec3(1.52, 1.24, 0.68); // Balanced HDR keeps rich granulation + fiery Bloom

  vec3 col = mix(colDark, colMid, spots);
  col = mix(col, colHot, smoothstep(0.38, 0.72, plasma));
  col = mix(col, colCore, smoothstep(0.76, 0.96, plasma));

  float ndotv = clamp(dot(normalize(vNormalView), normalize(vViewDir)), 0.0, 1.0);
  float limb = mix(0.58, 1.12, pow(ndotv, 0.52));
  float rim = pow(1.0 - ndotv, 2.8) * 0.85;
  col = col * limb + vec3(1.35, 0.42, 0.06) * rim;

  if (uDissolve > 0.002) {
    float dn = clamp(0.5 + n1 * 0.35 + n2 * 0.25, 0.0, 1.0);
    if (dn < uDissolve) discard;
    float edge = pow(smoothstep(uDissolve + 0.085, uDissolve, dn), 1.35);
    col += vec3(0.22, 1.25, 1.92) * edge * 0.95;
  }

  gl_FragColor = vec4(col, uOpacity);
}
`;

const SUN_CORONA_FRAG = `
uniform float uTime;
uniform float uOpacity;
varying vec3 vNormalView;
varying vec3 vViewDir;

void main() {
  vec3 N = normalize(vNormalView);
  vec3 V = normalize(vViewDir);
  float ndotv = clamp(dot(N, V), 0.0, 1.0);
  // Smooth outer edge feathering eliminates any hard outer circle outline
  float outerFeather = smoothstep(0.0, 0.48, ndotv);
  float rim = pow(1.0 - ndotv, 1.8) * outerFeather * 2.1;
  float pulse = 0.88 + 0.12 * sin(uTime * 3.4 + N.x * 9.0 + N.y * 7.0);
  vec3 col = mix(vec3(1.15, 0.26, 0.02), vec3(1.55, 0.88, 0.26), rim) * rim * 0.95 * pulse;
  gl_FragColor = vec4(col, clamp(rim * 0.72 * uOpacity, 0.0, 1.0));
}
`;

/**
 * Next-Gen Interstellar Gargantua Black Hole GLSL Shader:
 * - Volumetric Accretion Disk with animated polar FBM plasma filaments
 * - Relativistic Doppler Beaming (approaching side HDR white-blue feeding UnrealBloomPass;
 *   receding side molten deep crimson-orange)
 * - Razor-sharp HDR white Photon Ring around pitch-black Event Horizon void
 * - Upper & lower Gravitational Lensing arches curving naturally around Event Horizon
 */
const GARGANTUA_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const GARGANTUA_FRAG = `
uniform float uTime;
uniform float uSpinSpeed;
uniform float uOpacity;
varying vec2 vUv;

float hash21(vec2 p) {
  p = fract(p * vec2(127.1, 311.7));
  p += dot(p, p + 34.5);
  return fract(p.x * p.y);
}

float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
    mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x),
    f.y
  );
}

float fbmPolar(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * vnoise(p);
    p = p * 2.08 + vec2(1.9, 7.3);
    a *= 0.5;
  }
  return v;
}

void main() {
  vec2 p = (vUv - 0.5) * 2.0;
  float r = length(p);
  if (r > 0.99) discard;

  float ehR = 0.255;
  float t = uTime * uSpinSpeed;

  // Relativistic Doppler Beaming: approaching left side (-x) blueshifts & intensifies;
  // receding right side (+x) redshifts into deep molten crimson-orange.
  float doppler = clamp(1.16 - p.x * 0.84, 0.28, 1.78);
  float approachFactor = smoothstep(0.08, -0.62, p.x);

  // 1. Volumetric Equatorial Accretion Disk with fine Keplerian plasma filaments
  vec2 dp = vec2(p.x, p.y * 6.2);
  float dr = length(dp);
  float dAngle = atan(dp.y, dp.x);
  float shear = dAngle + 5.8 / (dr + 0.18) - t * 2.4;
  float filaments1 = fbmPolar(vec2(dr * 11.5 - t * 0.72, shear * 2.5));
  float filaments2 = fbmPolar(vec2(dr * 24.0 + t * 0.45, shear * 4.2 + filaments1 * 2.6));
  float rings = 0.55 + 0.45 * sin(dr * 48.0 - t * 3.8 + filaments1 * 4.2);

  float diskRadial = smoothstep(ehR * 1.02, ehR * 1.32, dr) * (1.0 - smoothstep(0.50, 0.96, dr));
  float diskProfile = exp(-abs(p.y) * (20.0 + dr * 11.0));
  float diskDensity = diskRadial * diskProfile * (0.34 + 0.65 * filaments1 + 0.35 * filaments2) * rings;

  // Smooth horizontal relativistic plasma stream crossing IN FRONT of the pitch-black Event Horizon
  float streamNoise = fbmPolar(vec2(p.x * 5.2 - t * 2.1, p.y * 34.0 + sin(p.x * 4.0) * 0.6));
  float streamFine = fbmPolar(vec2(p.x * 11.0 - t * 3.4, p.y * 58.0));
  float fgDiskBar =
    exp(-pow((p.y + 0.008) / 0.026, 2.0)) *
    (1.0 - smoothstep(0.42, 0.92, abs(p.x))) *
    (0.42 + 0.58 * streamNoise) *
    (0.70 + 0.45 * streamFine) *
    1.15;

  // 2. Upper & Lower Gravitational Lensing Arches (bent far-side disk image)
  float archR = length(vec2(p.x * 0.98, p.y * 1.04));
  float archCore = exp(-pow((archR - (ehR + 0.058)) / 0.040, 2.0));
  float archOuter = exp(-pow((archR - (ehR + 0.120)) / 0.092, 2.0)) * 0.30;
  float archSwirl = fbmPolar(vec2(archR * 15.0, atan(p.y, p.x) * 3.6 - t * 2.1));
  float archGate = smoothstep(0.02, 0.24, abs(p.y));
  float lowerWeight = (p.y < 0.0) ? 0.72 : 1.0;
  float lensing = (archCore * (0.55 + 0.65 * archSwirl) + archOuter * archSwirl) * archGate * lowerWeight;

  // 3. Razor-sharp thin Photon Ring right along Event Horizon edge
  float photonRing = exp(-pow((r - (ehR + 0.005)) / 0.0062, 2.0)) * 1.15;
  float innerRim = exp(-max(r - ehR, 0.0) * 14.0) * 0.30;

  // Pitch-black Event Horizon void (strictly obsidian black inside ehR except foreground equatorial stream)
  float voidMask = 1.0 - smoothstep(ehR - 0.008, ehR + 0.006, r);

  float bgPlasma = (lensing * 0.78 + diskDensity + innerRim) * (1.0 - voidMask);
  float fgPlasma = fgDiskBar;
  float energy = (bgPlasma + fgPlasma) * doppler + photonRing * (0.75 + 0.25 * doppler);

  // Calibrated Interstellar blackbody palette (clean HDR peak ~1.50 so UnrealBloomPass glows without whiteout)
  vec3 cDeepRed = vec3(0.44, 0.05, 0.008);
  vec3 cMolten  = vec3(0.98, 0.30, 0.03);
  vec3 cGold    = vec3(1.30, 0.76, 0.24);
  vec3 cWhite   = vec3(1.56, 1.48, 1.36);

  vec3 col = mix(cDeepRed, cMolten, smoothstep(0.04, 0.38, energy));
  col = mix(col, cGold, smoothstep(0.38, 0.85, energy));
  col = mix(col, cWhite, smoothstep(0.85, 1.38, energy));

  // Relativistic blue-white Doppler shift on the fast-approaching left wing
  col += vec3(0.28, 0.52, 0.92) * smoothstep(0.55, 1.45, energy) * approachFactor;

  float edgeFade = 1.0 - smoothstep(0.74, 0.98, r);
  float plasmaAlpha = clamp(energy * 0.95, 0.0, 1.0) * edgeFade;

  float finalAlpha = clamp(max(voidMask, plasmaAlpha), 0.0, 1.0) * uOpacity;
  float fgOcclusion = clamp(fgDiskBar * 1.45, 0.0, 1.0);
  vec3 finalCol = col * (1.0 - voidMask * (1.0 - fgOcclusion));
  finalCol += vec3(1.45, 1.36, 1.22) * photonRing * 0.42;

  gl_FragColor = vec4(finalCol, finalAlpha);
}
`;

const STAR_SWIRL_VERT = `
attribute vec4 aOrbit;
attribute float aColorMix;
uniform float uTime;
uniform float uSpinSpeed;
uniform float uOpacity;
varying float vAlpha;
varying float vHot;
varying float vColorMix;

void main() {
  float cycle = fract(aOrbit.y + uTime * aOrbit.z * 0.065 * uSpinSpeed);
  float radius = mix(4.1, 1.35, cycle);
  float invR = 1.0 / max(radius, 0.6);
  float angle = aOrbit.x + uTime * aOrbit.z * uSpinSpeed * (0.45 + invR * 1.15);

  vec3 pos = vec3(
    cos(angle) * radius,
    sin(angle) * radius * 0.24 + sin(angle * 2.0 + aOrbit.x) * (0.10 + 0.14 * (1.0 - cycle)),
    sin(angle) * radius * 0.82
  );

  vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mvPos;

  // Mask out any star whose 2D XY projection crosses inside the pitch-black Event Horizon silhouette
  float xyDist = length(pos.xy);
  float voidExclude = smoothstep(0.78, 1.15, xyDist);

  float edgeIn = smoothstep(0.0, 0.14, cycle);
  float edgeOut = 1.0 - smoothstep(0.84, 1.0, cycle);
  vAlpha = edgeIn * edgeOut * voidExclude * uOpacity * 0.72;
  vHot = smoothstep(0.45, 0.95, cycle);
  vColorMix = aColorMix;

  gl_PointSize = clamp(aOrbit.w * (0.85 + vHot * 0.55) * (8.2 / max(-mvPos.z, 1.0)), 1.4, 7.5);
}
`;

const STAR_SWIRL_FRAG = `
varying float vAlpha;
varying float vHot;
varying float vColorMix;

void main() {
  float dist = length(gl_PointCoord - 0.5);
  if (dist > 0.5) discard;
  float glow = exp(-dist * dist * 16.0);
  vec3 cool = mix(vec3(0.45, 0.72, 1.05), vec3(1.05, 0.58, 0.22), vColorMix);
  vec3 col = mix(cool, vec3(1.22, 1.12, 0.96), vHot * 0.75);
  gl_FragColor = vec4(col * glow, glow * vAlpha);
}
`;

// 450 3D Stardust Particles with Curl-Noise Swirling Drift & Bidirectional Coalesce/Dissolve
const BURST_VERT = `
attribute vec3 aVel;
attribute float aSize;
uniform float uProgress;
uniform float uOpacity;
uniform float uConverge; // 0.0 = outward Curl-Noise dissolve, 1.0 = inward coalesce
varying float vAlpha;
varying float vMix;

vec3 curlFlow(vec3 p, float t) {
  return vec3(
    sin(p.y * 2.4 + t * 3.1) - cos(p.z * 2.1 - t * 2.3),
    cos(p.z * 2.3 + t * 2.7) + 0.85,
    sin(p.x * 2.5 - t * 2.9) - cos(p.y * 2.2 + t * 2.1)
  );
}

void main() {
  // Reverse time smoothly when converging inward to materialize a planet
  float pTime = (uConverge > 0.5) ? (1.0 - uProgress) : uProgress;
  float ease = 1.0 - pow(1.0 - pTime, 2.1);

  vec3 basePos = position * (0.95 + ease * 0.35);
  vec3 drift = aVel * ease * 1.35;
  vec3 curl = curlFlow(position * 1.8 + aVel, pTime * 2.4) * (pTime * 0.72);
  vec3 pos = basePos + drift + curl + vec3(0.0, pTime * pTime * 1.15, 0.0);

  vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mvPos;

  float env = sin(uProgress * 3.14159);
  vAlpha = env * (1.0 - uProgress * 0.25) * uOpacity;
  vMix = fract(aSize * 0.37);
  gl_PointSize = clamp(aSize * (1.1 - pTime * 0.45) * (9.8 / max(-mvPos.z, 1.0)), 1.5, 11.5);
}
`;

const BURST_FRAG = `
varying float vAlpha;
varying float vMix;
void main() {
  if (vAlpha <= 0.01) discard;
  float dist = length(gl_PointCoord - 0.5);
  if (dist > 0.5) discard;
  float glow = exp(-dist * dist * 13.0);
  vec3 col = mix(vec3(0.48, 1.05, 1.85), vec3(1.75, 1.35, 0.55), vMix);
  gl_FragColor = vec4(col * glow, glow * vAlpha);
}
`;

// ============================================================================
// Procedural 1024x512 Multi-Octave FBM + Domain Warping Planet Texture Maps
// ============================================================================
function createEarthTextures(): {
  albedo: THREE.CanvasTexture;
  specular: THREE.CanvasTexture;
  night: THREE.CanvasTexture;
  bump: THREE.CanvasTexture;
  clouds: THREE.CanvasTexture;
} {
  const w = 1024;
  const h = 512;

  const albedoC = document.createElement('canvas'); albedoC.width = w; albedoC.height = h;
  const specC = document.createElement('canvas'); specC.width = w; specC.height = h;
  const nightC = document.createElement('canvas'); nightC.width = w; nightC.height = h;
  const bumpC = document.createElement('canvas'); bumpC.width = w; bumpC.height = h;
  const cloudC = document.createElement('canvas'); cloudC.width = 512; cloudC.height = 256;

  const aCtx = albedoC.getContext('2d')!;
  const sCtx = specC.getContext('2d')!;
  const nCtx = nightC.getContext('2d')!;
  const bCtx = bumpC.getContext('2d')!;

  const aImg = aCtx.createImageData(w, h);
  const sImg = sCtx.createImageData(w, h);
  const nImg = nCtx.createImageData(w, h);
  const bImg = bCtx.createImageData(w, h);

  const aD = aImg.data;
  const sD = sImg.data;
  const nD = nImg.data;
  const bD = bImg.data;

  for (let y = 0; y < h; y += 1) {
    const v = y / (h - 1);
    const phi = v * Math.PI;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    const absLat = Math.abs(cosPhi);

    for (let x = 0; x < w; x += 1) {
      const theta = (x / w) * Math.PI * 2;
      const sx = sinPhi * Math.cos(theta);
      const sy = cosPhi;
      const sz = sinPhi * Math.sin(theta);

      // 6-octave Domain-Warped FBM for realistic fractal continents & mountain chains
      const elev = warpedFbm6(sx * 1.95 + 2.4, sy * 1.95 - 1.2, sz * 1.95 + 1.8, 0.95);
      const ridge = 1 - Math.abs(fbm3(sx * 6.2 + 5.1, sy * 6.2, sz * 6.2 - 3.2, 4) - 0.5) * 2;
      const moisture = fbm3(sx * 2.8 - 4.4, sy * 2.8 + 1.7, sz * 2.8, 4);
      const idx = (y * w + x) * 4;

      const ice = absLat > 0.87 - elev * 0.08;
      if (ice) {
        aD[idx] = 238; aD[idx + 1] = 245; aD[idx + 2] = 252; aD[idx + 3] = 255;
        sD[idx] = 40; sD[idx + 1] = 40; sD[idx + 2] = 40; sD[idx + 3] = 255;
        bD[idx] = 165; bD[idx + 1] = 165; bD[idx + 2] = 165; bD[idx + 3] = 255;
        nD[idx + 3] = 255;
      } else if (elev < 0.512) {
        const depth = Math.max(0, (0.512 - elev) / 0.28);
        const shelf = Math.pow(Math.max(0, 1 - depth * 4.2), 2);
        aD[idx] = clampByte(7 + shelf * 28);
        aD[idx + 1] = clampByte(34 + (1 - depth) * 44 + shelf * 64);
        aD[idx + 2] = clampByte(88 + (1 - depth) * 66 + shelf * 46);
        aD[idx + 3] = 255;

        sD[idx] = 250; sD[idx + 1] = 250; sD[idx + 2] = 250; sD[idx + 3] = 255;
        bD[idx] = 28; bD[idx + 1] = 28; bD[idx + 2] = 28; bD[idx + 3] = 255;
        nD[idx + 3] = 255;
      } else {
        const landH = Math.min(1, (elev - 0.512) / 0.26);
        const desertMask =
          Math.max(0, 1 - Math.abs(absLat - 0.34) * 3.8) * Math.max(0, 0.58 - moisture) * 2.6;
        const mtnMask = Math.max(0, (landH * 0.65 + ridge * 0.45) - 0.52) * 2.4;

        let r = 28 + landH * 52 + desertMask * 155 + mtnMask * 95;
        let g = 92 + (1 - landH) * 38 + desertMask * 82 + mtnMask * 68;
        let b = 34 + landH * 24 + desertMask * 36 + mtnMask * 72;
        if (landH + ridge * 0.22 > 0.88) {
          r = 228; g = 234; b = 242; // high snow-capped ridges
        }

        aD[idx] = clampByte(r);
        aD[idx + 1] = clampByte(g);
        aD[idx + 2] = clampByte(b);
        aD[idx + 3] = 255;

        sD[idx] = 12; sD[idx + 1] = 12; sD[idx + 2] = 12; sD[idx + 3] = 255;

        const bv = clampByte(108 + landH * 98 + ridge * 48);
        bD[idx] = bv; bD[idx + 1] = bv; bD[idx + 2] = bv; bD[idx + 3] = 255;

        // Amber-gold night city lights along temperate coasts and metro clusters
        if (absLat < 0.72 && landH < 0.40) {
          const coastBias = 1 - landH / 0.40;
          const metro = noise3(sx * 16.5 + 7.1, sy * 16.5, sz * 16.5 - 3.3);
          const grid = noise3(sx * 48.0, sy * 48.0, sz * 48.0);
          const spark = Math.max(0, metro * coastBias - 0.44) * 5.2 * (grid > 0.54 ? 1 : 0.22);
          if (spark > 0.04) {
            nD[idx] = clampByte(255 * Math.min(1, spark * 1.5));
            nD[idx + 1] = clampByte(184 * Math.min(1, spark * 1.25));
            nD[idx + 2] = clampByte(78 * Math.min(1, spark));
          }
        }
        nD[idx + 3] = 255;
      }
    }
  }

  aCtx.putImageData(aImg, 0, 0);
  sCtx.putImageData(sImg, 0, 0);
  nCtx.putImageData(nImg, 0, 0);
  bCtx.putImageData(bImg, 0, 0);

  // Domain-warped swirling 3D cloud deck (512x256)
  const cw = cloudC.width;
  const ch = cloudC.height;
  const cCtx = cloudC.getContext('2d')!;
  const cImg = cCtx.createImageData(cw, ch);
  const cD = cImg.data;
  for (let y = 0; y < ch; y += 1) {
    const phi = (y / (ch - 1)) * Math.PI;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    for (let x = 0; x < cw; x += 1) {
      const theta = (x / cw) * Math.PI * 2;
      const sx = sinPhi * Math.cos(theta);
      const sy = cosPhi;
      const sz = sinPhi * Math.sin(theta);
      const cVal = warpedFbm6(sx * 3.0, sy * 3.8, sz * 3.0, 0.85);
      const alpha = Math.max(0, (cVal - 0.495) / 0.31);
      const idx = (y * cw + x) * 4;
      cD[idx] = 248;
      cD[idx + 1] = 252;
      cD[idx + 2] = 255;
      cD[idx + 3] = clampByte(Math.pow(Math.min(1, alpha), 1.2) * 238);
    }
  }
  cCtx.putImageData(cImg, 0, 0);

  return {
    albedo: makeCanvasTexture(albedoC, true),
    specular: makeCanvasTexture(specC, false),
    night: makeCanvasTexture(nightC, true),
    bump: makeCanvasTexture(bumpC, false),
    clouds: makeCanvasTexture(cloudC, true),
  };
}

function createMoonTextures(): { albedo: THREE.CanvasTexture; bump: THREE.CanvasTexture } {
  const w = 256;
  const h = 128;
  const aC = document.createElement('canvas'); aC.width = w; aC.height = h;
  const bC = document.createElement('canvas'); bC.width = w; bC.height = h;
  const aCtx = aC.getContext('2d')!;
  const bCtx = bC.getContext('2d')!;
  const aImg = aCtx.createImageData(w, h);
  const bImg = bCtx.createImageData(w, h);

  for (let y = 0; y < h; y += 1) {
    const phi = (y / (h - 1)) * Math.PI;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    for (let x = 0; x < w; x += 1) {
      const theta = (x / w) * Math.PI * 2;
      const sx = sinPhi * Math.cos(theta);
      const sy = cosPhi;
      const sz = sinPhi * Math.sin(theta);
      const maria = fbm3(sx * 2.5 + 1.1, sy * 2.5, sz * 2.5, 4);
      const craters = Math.abs(noise3(sx * 12.5, sy * 12.5, sz * 12.5) - 0.5) * 2;
      const shade = clampByte(110 + maria * 100 - (1 - craters) * 38);
      const idx = (y * w + x) * 4;
      aImg.data[idx] = shade;
      aImg.data[idx + 1] = clampByte(shade * 0.98);
      aImg.data[idx + 2] = clampByte(shade * 0.95);
      aImg.data[idx + 3] = 255;
      const bv = clampByte(craters * 215 + maria * 40);
      bImg.data[idx] = bv; bImg.data[idx + 1] = bv; bImg.data[idx + 2] = bv; bImg.data[idx + 3] = 255;
    }
  }
  aCtx.putImageData(aImg, 0, 0);
  bCtx.putImageData(bImg, 0, 0);
  return { albedo: makeCanvasTexture(aC, true), bump: makeCanvasTexture(bC, false) };
}

function createMarsTextures(): {
  albedo: THREE.CanvasTexture;
  bump: THREE.CanvasTexture;
  roughness: THREE.CanvasTexture;
} {
  const w = 1024;
  const h = 512;
  const aC = document.createElement('canvas'); aC.width = w; aC.height = h;
  const bC = document.createElement('canvas'); bC.width = w; bC.height = h;
  const rC = document.createElement('canvas'); rC.width = w; rC.height = h;
  const aCtx = aC.getContext('2d')!;
  const bCtx = bC.getContext('2d')!;
  const rCtx = rC.getContext('2d')!;
  const aImg = aCtx.createImageData(w, h);
  const bImg = bCtx.createImageData(w, h);
  const rImg = rCtx.createImageData(w, h);

  for (let y = 0; y < h; y += 1) {
    const phi = (y / (h - 1)) * Math.PI;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    const absLat = Math.abs(cosPhi);
    for (let x = 0; x < w; x += 1) {
      const theta = (x / w) * Math.PI * 2;
      const sx = sinPhi * Math.cos(theta);
      const sy = cosPhi;
      const sz = sinPhi * Math.sin(theta);

      const terrain = warpedFbm6(sx * 2.4 + 8.3, sy * 2.4 - 2.1, sz * 2.4 + 4.4, 0.75);
      const ridge = 1 - Math.abs(fbm3(sx * 7.5, sy * 7.5, sz * 7.5, 4) - 0.5) * 2;
      const canyon =
        Math.exp(-Math.pow((sy + 0.08 + Math.sin(theta * 2) * 0.04) * 15, 2)) *
        Math.max(0, Math.sin(theta - 0.5));
      const idx = (y * w + x) * 4;

      if (absLat > 0.91 - terrain * 0.05) {
        aImg.data[idx] = 240; aImg.data[idx + 1] = 232; aImg.data[idx + 2] = 224; aImg.data[idx + 3] = 255;
        bImg.data[idx] = 210; bImg.data[idx + 1] = 210; bImg.data[idx + 2] = 210; bImg.data[idx + 3] = 255;
        rImg.data[idx] = 155; rImg.data[idx + 1] = 155; rImg.data[idx + 2] = 155; rImg.data[idx + 3] = 255;
      } else {
        const basalt = Math.max(0, 0.52 - terrain) * 1.95;
        aImg.data[idx] = clampByte(208 - basalt * 98 - canyon * 76 + ridge * 26);
        aImg.data[idx + 1] = clampByte(94 - basalt * 52 - canyon * 46 + terrain * 36);
        aImg.data[idx + 2] = clampByte(48 - basalt * 28 - canyon * 24 + terrain * 18);
        aImg.data[idx + 3] = 255;

        const bv = clampByte(terrain * 168 + ridge * 76 - canyon * 98);
        bImg.data[idx] = bv; bImg.data[idx + 1] = bv; bImg.data[idx + 2] = bv; bImg.data[idx + 3] = 255;

        const rv = clampByte(218 + basalt * 24 - ridge * 18);
        rImg.data[idx] = rv; rImg.data[idx + 1] = rv; rImg.data[idx + 2] = rv; rImg.data[idx + 3] = 255;
      }
    }
  }
  aCtx.putImageData(aImg, 0, 0);
  bCtx.putImageData(bImg, 0, 0);
  rCtx.putImageData(rImg, 0, 0);
  return {
    albedo: makeCanvasTexture(aC, true),
    bump: makeCanvasTexture(bC, false),
    roughness: makeCanvasTexture(rC, false),
  };
}

function createSaturnTextures(): { albedo: THREE.CanvasTexture; ring: THREE.CanvasTexture } {
  const w = 1024;
  const h = 512;
  const aC = document.createElement('canvas'); aC.width = w; aC.height = h;
  const aCtx = aC.getContext('2d')!;
  const aImg = aCtx.createImageData(w, h);

  for (let y = 0; y < h; y += 1) {
    const v = y / (h - 1);
    const phi = v * Math.PI;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    const bandWave =
      Math.sin(v * Math.PI * 16) * 0.34 +
      Math.sin(v * Math.PI * 34 + 1.2) * 0.20 +
      Math.cos(v * Math.PI * 6) * 0.24;
    const polarTint = Math.pow(Math.abs(cosPhi), 3.0);

    for (let x = 0; x < w; x += 1) {
      const theta = (x / w) * Math.PI * 2;
      const sx = sinPhi * Math.cos(theta);
      const sy = cosPhi;
      const sz = sinPhi * Math.sin(theta);
      const turb = warpedFbm6(sx * 2.0, sy * 11.5, sz * 2.0, 0.45) - 0.5;
      const b = bandWave + turb * 0.36;
      const idx = (y * w + x) * 4;

      aImg.data[idx] = clampByte(226 + b * 24 - polarTint * 64);
      aImg.data[idx + 1] = clampByte(196 + b * 22 - polarTint * 52);
      aImg.data[idx + 2] = clampByte(136 + b * 26 - polarTint * 24);
      aImg.data[idx + 3] = 255;
    }
  }
  aCtx.putImageData(aImg, 0, 0);

  const rC = document.createElement('canvas');
  rC.width = 512; rC.height = 8;
  const rCtx = rC.getContext('2d')!;
  const rImg = rCtx.createImageData(512, 8);
  for (let x = 0; x < 512; x += 1) {
    const u = x / 511;
    let alpha = 0;
    let bright = 0.75;
    if (u < 0.14) {
      alpha = 0.16 + u * 1.35;
      bright = 0.56;
    } else if (u < 0.57) {
      alpha = 0.88 + Math.sin(u * 185) * 0.08 + Math.sin(u * 460) * 0.04;
      bright = 0.94 + Math.sin(u * 130) * 0.08;
    } else if (u < 0.65) {
      // Dark Cassini Division
      alpha = 0.05;
      bright = 0.25;
    } else if (u < 0.86) {
      alpha = 0.72 + Math.sin(u * 230) * 0.09;
      bright = 0.82 + Math.sin(u * 95) * 0.06;
    } else if (u < 0.89) {
      // Encke Gap
      alpha = 0.08;
      bright = 0.3;
    } else {
      alpha = Math.max(0, (1 - u) / 0.11) * 0.62;
      bright = 0.78;
    }

    const r = clampByte(238 * bright);
    const g = clampByte(214 * bright);
    const b = clampByte(166 * bright);
    const a = clampByte(Math.min(1, Math.max(0, alpha)) * 244);
    for (let y = 0; y < 8; y += 1) {
      const idx = (y * 512 + x) * 4;
      rImg.data[idx] = r; rImg.data[idx + 1] = g; rImg.data[idx + 2] = b; rImg.data[idx + 3] = a;
    }
  }
  rCtx.putImageData(rImg, 0, 0);

  return {
    albedo: makeCanvasTexture(aC, true),
    ring: makeCanvasTexture(rC, true),
  };
}

function createJupiterTextures(): { albedo: THREE.CanvasTexture; bump: THREE.CanvasTexture } {
  const w = 1024;
  const h = 512;
  const aC = document.createElement('canvas'); aC.width = w; aC.height = h;
  const bC = document.createElement('canvas'); bC.width = w; bC.height = h;
  const aCtx = aC.getContext('2d')!;
  const bCtx = bC.getContext('2d')!;
  const aImg = aCtx.createImageData(w, h);
  const bImg = bCtx.createImageData(w, h);

  for (let y = 0; y < h; y += 1) {
    const v = y / (h - 1);
    const phi = v * Math.PI;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);

    for (let x = 0; x < w; x += 1) {
      const u = x / w;
      const theta = u * Math.PI * 2;
      const sx = sinPhi * Math.cos(theta);
      const sy = cosPhi;
      const sz = sinPhi * Math.sin(theta);

      // Domain-Warped 6-octave FBM for turbulent atmospheric shear eddies
      const swirl = warpedFbm6(sx * 3.4 + 2.2, sy * 9.5, sz * 3.4 - 1.7, 1.15) - 0.5;
      const lat = v + swirl * 0.068;
      const band1 = Math.sin(lat * Math.PI * 16);
      const band2 = Math.cos(lat * Math.PI * 34 + swirl * 6.0);
      const zone = 0.5 + 0.34 * band1 + 0.16 * band2;

      let r = 176 + zone * 74;
      let g = 108 + zone * 118;
      let b = 74 + zone * 112;
      let spotElev = 0;

      // Sculpted Great Red Spot anticyclone
      const dx = (u - 0.36) / 0.082;
      const dy = (v - 0.62) / 0.054;
      const dSpot = dx * dx + dy * dy;
      if (dSpot < 1.38) {
        const spotM = Math.exp(-dSpot * 1.85);
        const spotSwirl = Math.sin(Math.atan2(dy, dx) * 2 + Math.sqrt(dSpot) * 10) * 16;
        r = r * (1 - spotM) + (222 + spotSwirl) * spotM;
        g = g * (1 - spotM) + (74 + spotSwirl * 0.5) * spotM;
        b = b * (1 - spotM) + 42 * spotM;
        spotElev = spotM * 85;
      }

      const idx = (y * w + x) * 4;
      aImg.data[idx] = clampByte(r);
      aImg.data[idx + 1] = clampByte(g);
      aImg.data[idx + 2] = clampByte(b);
      aImg.data[idx + 3] = 255;

      const bv = clampByte(115 + zone * 78 + swirl * 72 + spotElev);
      bImg.data[idx] = bv; bImg.data[idx + 1] = bv; bImg.data[idx + 2] = bv; bImg.data[idx + 3] = 255;
    }
  }
  aCtx.putImageData(aImg, 0, 0);
  bCtx.putImageData(bImg, 0, 0);
  return { albedo: makeCanvasTexture(aC, true), bump: makeCanvasTexture(bC, false) };
}

function createNeptuneTexture(): THREE.CanvasTexture {
  const w = 1024;
  const h = 512;
  const aC = document.createElement('canvas'); aC.width = w; aC.height = h;
  const aCtx = aC.getContext('2d')!;
  const aImg = aCtx.createImageData(w, h);

  for (let y = 0; y < h; y += 1) {
    const v = y / (h - 1);
    const phi = v * Math.PI;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);

    for (let x = 0; x < w; x += 1) {
      const u = x / w;
      const theta = u * Math.PI * 2;
      const sx = sinPhi * Math.cos(theta);
      const sy = cosPhi;
      const sz = sinPhi * Math.sin(theta);

      const flow = warpedFbm6(sx * 2.4 + 4.1, sy * 11.0, sz * 2.4, 0.72);
      const band = Math.sin((v + flow * 0.045) * Math.PI * 14) * 0.5 + 0.5;
      const streak = Math.max(0, fbm3(sx * 5.0, sy * 24.0, sz * 5.0, 3) - 0.64) * 3.2;

      let r = 20 + band * 34 + streak * 185;
      let g = 70 + band * 88 + streak * 172;
      let b = 188 + band * 64 + streak * 65;

      const dx = (u - 0.58) / 0.068;
      const dy = (v - 0.58) / 0.045;
      const ds = dx * dx + dy * dy;
      if (ds < 1.2) {
        const dm = Math.exp(-ds * 2.2) * 0.54;
        r *= 1 - dm; g *= 1 - dm; b *= 1 - dm * 0.7;
      }

      const idx = (y * w + x) * 4;
      aImg.data[idx] = clampByte(r);
      aImg.data[idx + 1] = clampByte(g);
      aImg.data[idx + 2] = clampByte(b);
      aImg.data[idx + 3] = 255;
    }
  }
  aCtx.putImageData(aImg, 0, 0);
  return makeCanvasTexture(aC, true);
}

// ============================================================================
// Hand Gesture & Extended Finger Count Helpers
// ============================================================================
function countExtendedFingers(h: HandPose2D): number {
  const p = h.pts;
  const wx = p[0];
  const wy = p[1];
  const s = Math.max(h.palmScale, 40);
  let count = 0;

  // Index (8 vs 6), Middle (12 vs 10), Ring (16 vs 14), Pinky (20 vs 18)
  const pairs = [
    [16, 12],
    [24, 20],
    [32, 28],
    [40, 36],
  ];
  for (let i = 0; i < pairs.length; i += 1) {
    const ti = pairs[i][0];
    const pi = pairs[i][1];
    if (Math.hypot(p[ti] - wx, p[ti + 1] - wy) > Math.hypot(p[pi] - wx, p[pi + 1] - wy) + s * 0.14) {
      count += 1;
    }
  }

  // Thumb (4 vs pinky MCP 17 compared against thumb MCP 2 vs 17)
  const dThumbTip = Math.hypot(p[8] - p[34], p[9] - p[35]);
  const dThumbMcp = Math.hypot(p[4] - p[34], p[5] - p[35]);
  if (dThumbTip > dThumbMcp + s * 0.18) {
    count += 1;
  }

  return count;
}

function isTwoFingerPose(h: HandPose2D): boolean {
  const p = h.pts;
  const wx = p[0];
  const wy = p[1];
  const s = Math.max(h.palmScale, 40);

  const indexExt = Math.hypot(p[16] - wx, p[17] - wy) > Math.hypot(p[12] - wx, p[13] - wy) + s * 0.16;
  const middleExt = Math.hypot(p[24] - wx, p[25] - wy) > Math.hypot(p[20] - wx, p[21] - wy) + s * 0.16;
  const ringCurled = Math.hypot(p[32] - wx, p[33] - wy) < Math.hypot(p[28] - wx, p[29] - wy) + s * 0.08;
  const pinkyCurled = Math.hypot(p[40] - wx, p[41] - wy) < Math.hypot(p[36] - wx, p[37] - wy) + s * 0.08;
  return indexExt && middleExt && ringCurled && pinkyCurled;
}

function isIndexHoverPose(h: HandPose2D): boolean {
  const p = h.pts;
  const wx = p[0];
  const wy = p[1];
  const s = Math.max(h.palmScale, 40);
  return Math.hypot(p[16] - wx, p[17] - wy) > Math.hypot(p[12] - wx, p[13] - wy) + s * 0.12;
}

// ============================================================================
// Three.js WebGL2 3D Cinema Cosmos Engine (EffectComposer + UnrealBloomPass)
// ============================================================================
interface BuiltPlanet {
  root: THREE.Group;
  spinGroup: THREE.Group;
  dissolveUniform: { value: number };
  cloudMesh?: THREE.Mesh;
  moonPivot?: THREE.Group;
  atmoMat?: THREE.ShaderMaterial;
  sunMat?: THREE.ShaderMaterial;
  sunCoronaMat?: THREE.ShaderMaterial;
  earthMat?: THREE.ShaderMaterial;
  ringMat?: THREE.ShaderMaterial;
  stdMat?: THREE.MeshStandardMaterial;
}

export class Cosmos3DEngine {
  private canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private lightDir = new THREE.Vector3(7.5, 4.2, 5.2).normalize();

  // Alpha-preserving EffectComposer + UnrealBloomPass pipeline
  private sceneRT: THREE.WebGLRenderTarget;
  private composer: EffectComposer;
  private feedPass: ShaderPass;
  private bloomPass: UnrealBloomPass;
  private outputPass: ShaderPass;

  private planetContainer = new THREE.Group();
  private builtPlanets: (BuiltPlanet | null)[] = new Array(PLANETS.length).fill(null);
  private sphereGeo: THREE.SphereGeometry | null = null;
  private atmoGeo: THREE.SphereGeometry | null = null;

  // Shared Three.js Volumetric Plasma Lightning & Fluid 8,000-Particle Sandbox (zero extra WebGL context / Bloom pass)
  private lightning3D = new Lightning3DEngine();
  private fluidSandbox = new FluidParticleSandbox();

  private burstPoints: THREE.Points | null = null;
  private burstMat: THREE.ShaderMaterial | null = null;
  private burstProgress = 1;
  private burstConverge = 0; // 0 = dissolve outward, 1 = coalesce inward

  private bhRoot = new THREE.Group();
  private bhMesh: THREE.Mesh | null = null;
  private bhMat: THREE.ShaderMaterial | null = null;
  private bhStars: THREE.Points | null = null;
  private bhStarsMat: THREE.ShaderMaterial | null = null;

  private initialized = false;
  private hasRenderedFrame = false;
  private disposed = false;
  private prewarmTimerId: number | null = null;
  private viewW = 1280;
  private viewH = 720;

  // Interactive Physics & Anti-Accidental Black Hole State
  private planetIdx = 0;
  private pendingPlanetIdx = 0;
  private switchCooldown = 0;
  private switchTimer = -1;
  private switchedMidpoint = false;
  private hudTimer = 0;
  private wasTwoFinger = false;

  private posX = 0;
  private posY = 0;
  private velX = 0;
  private velY = 0;
  private lastPlanetRadiusPx = 82;
  private initializedPos = false;
  private wasShowingPlanet = false;
  private planetDissolveT = 1; // 0 -> 1 ethereal GLSL dissolve when retracting hand
  private planetMaterializeT = 1; // 0 -> 1 reverse coalesce when raising hand back
  private spinY = 0;
  private spinVel = 0.42;
  private prevAnchorX = 0;
  private planetAlpha = 0;

  // Lightweight 2D rising astral wisps (Zero-GC typed arrays) for ethereal stardust trails
  private wispX = new Float32Array(48);
  private wispY = new Float32Array(48);
  private wispVx = new Float32Array(48);
  private wispVy = new Float32Array(48);
  private wispLife = new Float32Array(48);
  private wispMax = new Float32Array(48);
  private wispGold = new Uint8Array(48);
  private wispCursor = 0;

  // Black Hole Gating (450ms charge + 400ms exit hysteresis)
  private bhActive = false;
  private bhChargeSec = 0;
  private bhExitSec = 0;
  private bhBlend = 0;
  private bhX = 0;
  private bhY = 0;
  private bhScale = 1.0;
  private bhRoll = 0;
  private bhSpinSpeed = 1.0;
  private time = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: false,
      powerPreference: 'high-performance',
      premultipliedAlpha: true,
    });
    this.renderer.setPixelRatio(1);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(CAMERA_FOV_DEG, 16 / 9, 0.1, 100);
    this.camera.position.set(0, 0, CAMERA_Z);

    // Cinema Lighting: high-intensity Sun DirectionalLight (3.2) + ultra-low AmbientLight (0.06)
    // for crisp Day/Night Terminator shadow and real 3D sculptural depth
    const sunLight = new THREE.DirectionalLight(0xfff4e2, 3.2);
    sunLight.position.copy(this.lightDir).multiplyScalar(14);
    this.scene.add(sunLight);

    const ambient = new THREE.AmbientLight(0x162033, 0.06);
    this.scene.add(ambient);

    this.scene.add(this.planetContainer);
    this.scene.add(this.bhRoot);
    this.scene.add(this.lightning3D.root);
    this.scene.add(this.fluidSandbox.root);
    this.bhRoot.visible = false;

    const initW = Math.max(1, canvas.width || window.innerWidth);
    const initH = Math.max(1, canvas.height || window.innerHeight);

    const rtOpts: THREE.RenderTargetOptions = {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    };
    this.sceneRT = new THREE.WebGLRenderTarget(initW, initH, rtOpts);

    const composerRT = new THREE.WebGLRenderTarget(initW, initH, rtOpts);
    this.composer = new EffectComposer(this.renderer, composerRT);

    this.feedPass = new ShaderPass(SCENE_FEED_SHADER);
    this.feedPass.uniforms.tScene.value = this.sceneRT.texture;
    this.composer.addPass(this.feedPass);

    // Half-resolution UnrealBloomPass for 4x lower fillrate on RTX 3050 45W at 120 FPS
    const bloomW = Math.max(1, Math.floor(initW / 2));
    const bloomH = Math.max(1, Math.floor(initH / 2));
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(bloomW, bloomH), 0.85, 0.65, 0.55);
    this.composer.addPass(this.bloomPass);

    // Final alpha-preserving ACES tone-mapping output pass
    this.outputPass = new ShaderPass(ALPHA_OUTPUT_SHADER);
    this.outputPass.uniforms.tSceneAlpha.value = this.sceneRT.texture;
    this.composer.addPass(this.outputPass);

    this.resize(initW, initH);
    this.wispLife.fill(99);
    this.wispMax.fill(1);
    this.scheduleBackgroundPrewarm();
  }

  private spawnAstralWisps(cx: number, cy: number, radiusPx: number, count: number, goldBias = 0.4): void {
    for (let k = 0; k < count; k += 1) {
      const i = this.wispCursor;
      this.wispCursor = (this.wispCursor + 1) % 48;
      const a = Math.random() * Math.PI * 2;
      const r = radiusPx * (0.25 + Math.random() * 0.75);
      this.wispX[i] = cx + Math.cos(a) * r;
      this.wispY[i] = cy + Math.sin(a) * r;
      const spd = 28 + Math.random() * 95;
      this.wispVx[i] = Math.cos(a) * spd;
      this.wispVy[i] = Math.sin(a) * spd - 48 - Math.random() * 65; // ethereal upward float
      this.wispLife[i] = 0;
      this.wispMax[i] = 0.38 + Math.random() * 0.42;
      this.wispGold[i] = Math.random() < goldBias ? 1 : 0;
    }
  }

  /** Pre-warms all 6 planet meshes, textures, and WebGL2 shaders so transitions take < 5ms. */
  private scheduleBackgroundPrewarm(): void {
    let idx = 0;
    const step = () => {
      if (this.disposed) {
        this.prewarmTimerId = null;
        return;
      }
      if (idx < PLANETS.length) {
        if (!this.builtPlanets[idx]) {
          this.getOrBuildPlanet(idx);
        }
        idx += 1;
        this.prewarmTimerId = window.setTimeout(step, 40);
      } else {
        this.compileAllGpuResources();
        this.prewarmTimerId = null;
      }
    };
    this.prewarmTimerId = window.setTimeout(step, 60);
  }

  /** Compiles all planet/BlackHole/Bloom shaders and uploads all textures to GPU VRAM up front. */
  private compileAllGpuResources(): void {
    if (this.disposed) return;
    this.ensureCoreResources();

    const prevVis: boolean[] = [];
    for (let i = 0; i < this.builtPlanets.length; i += 1) {
      const bp = this.builtPlanets[i];
      if (!bp) continue;
      prevVis[i] = bp.root.visible;
      bp.root.visible = true;
      // Upload every texture to GPU VRAM immediately
      bp.root.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          const mat = obj.material as THREE.Material & {
            map?: THREE.Texture | null;
            bumpMap?: THREE.Texture | null;
            roughnessMap?: THREE.Texture | null;
            uniforms?: Record<string, { value: unknown }>;
          };
          if (mat.map) this.renderer.initTexture(mat.map);
          if (mat.bumpMap) this.renderer.initTexture(mat.bumpMap);
          if (mat.roughnessMap) this.renderer.initTexture(mat.roughnessMap);
          if (mat.uniforms) {
            for (const key in mat.uniforms) {
              const val = mat.uniforms[key]?.value;
              if (val instanceof THREE.Texture) this.renderer.initTexture(val);
            }
          }
        }
      });
    }

    const prevBhVis = this.bhRoot.visible;
    const prevBurstVis = this.burstPoints?.visible ?? false;
    const prevLightVis = this.lightning3D.root.visible;
    const prevFluidVis = this.fluidSandbox.root.visible;
    this.bhRoot.visible = true;
    this.lightning3D.root.visible = true;
    this.fluidSandbox.root.visible = true;
    if (this.burstPoints) this.burstPoints.visible = true;

    // Compile all shaders (Cosmos + Dissolve + 3D Lightning + Fluid + UnrealBloomPass) offscreen
    this.renderer.compile(this.scene, this.camera);
    this.renderer.setRenderTarget(this.sceneRT);
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(null);
    this.composer.render();
    this.renderer.clear();

    // Restore clean hidden state
    for (let i = 0; i < this.builtPlanets.length; i += 1) {
      const bp = this.builtPlanets[i];
      if (bp) bp.root.visible = prevVis[i] ?? false;
    }
    this.bhRoot.visible = prevBhVis;
    this.lightning3D.root.visible = prevLightVis;
    this.fluidSandbox.root.visible = prevFluidVis;
    if (this.burstPoints) this.burstPoints.visible = prevBurstVis;
    this.hasRenderedFrame = false;
  }

  /** Ensures all 6 planets + Gargantua shaders/textures are compiled and resident in GPU VRAM. */
  public prewarmAllNow(): void {
    for (let i = 0; i < PLANETS.length; i += 1) {
      if (!this.builtPlanets[i]) this.getOrBuildPlanet(i);
    }
    this.compileAllGpuResources();
  }

  public setLightningSfxEnabled(enabled: boolean): void {
    this.lightning3D.setSfxEnabled(enabled);
  }

  public resetLightning(): void {
    this.lightning3D.reset();
  }

  public resetFluid(): void {
    this.fluidSandbox.reset();
  }

  /** Reports whether Gargantua Black Hole is charging or active (blocks Air Swipe). */
  public isBlackHoleBusy(): boolean {
    return this.bhActive || this.bhChargeSec > 0.05 || this.bhBlend > 0.08;
  }

  resize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (w === this.viewW && h === this.viewH && this.canvas.width === w && this.canvas.height === h) {
      return;
    }
    this.viewW = w;
    this.viewH = h;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
    this.sceneRT.setSize(w, h);
    this.composer.setSize(w, h);
    this.bloomPass.setSize(Math.max(1, Math.floor(w / 2)), Math.max(1, Math.floor(h / 2)));
  }

  clear(): void {
    if (!this.hasRenderedFrame) return;
    this.renderer.setRenderTarget(null);
    this.renderer.clear();
    this.hasRenderedFrame = false;
  }

  reset(): void {
    this.initializedPos = false;
    this.wasShowingPlanet = false;
    this.planetDissolveT = 1;
    this.velX = 0;
    this.velY = 0;
    this.planetAlpha = 0;
    this.bhActive = false;
    this.bhChargeSec = 0;
    this.bhExitSec = 0;
    this.bhBlend = 0;
    this.wasTwoFinger = false;
    this.switchTimer = -1;
    this.burstProgress = 1;
    this.hudTimer = HUD_DISPLAY_SEC * 0.85;
  }

  private ensureCoreResources(): void {
    if (this.initialized) return;
    this.initialized = true;

    this.sphereGeo = new THREE.SphereGeometry(1, 64, 48);
    this.atmoGeo = new THREE.SphereGeometry(1.11, 48, 36);

    const burstGeo = new THREE.BufferGeometry();
    const bPos = new Float32Array(BURST_PARTICLE_COUNT * 3);
    const bVel = new Float32Array(BURST_PARTICLE_COUNT * 3);
    const bSize = new Float32Array(BURST_PARTICLE_COUNT);
    for (let i = 0; i < BURST_PARTICLE_COUNT; i += 1) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const sx = Math.sin(phi) * Math.cos(theta);
      const sy = Math.cos(phi);
      const sz = Math.sin(phi) * Math.sin(theta);
      // Spawn particles directly across the planet surface for true stardust dispersion
      const rSurf = 0.72 + Math.random() * 0.36;
      bPos[i * 3] = sx * rSurf;
      bPos[i * 3 + 1] = sy * rSurf;
      bPos[i * 3 + 2] = sz * rSurf;
      const speed = 1.1 + Math.random() * 2.2;
      bVel[i * 3] = sx * speed;
      bVel[i * 3 + 1] = sy * speed;
      bVel[i * 3 + 2] = sz * speed;
      bSize[i] = 3.6 + Math.random() * 6.2;
    }
    burstGeo.setAttribute('position', new THREE.BufferAttribute(bPos, 3));
    burstGeo.setAttribute('aVel', new THREE.BufferAttribute(bVel, 3));
    burstGeo.setAttribute('aSize', new THREE.BufferAttribute(bSize, 1));
    this.burstMat = new THREE.ShaderMaterial({
      vertexShader: BURST_VERT,
      fragmentShader: BURST_FRAG,
      uniforms: {
        uProgress: { value: 1 },
        uOpacity: { value: 0 },
        uConverge: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.burstPoints = new THREE.Points(burstGeo, this.burstMat);
    this.burstPoints.visible = false;
    this.planetContainer.add(this.burstPoints);

    // Next-Gen Interstellar Gargantua Black Hole Shader Quad (clean framing between both hands)
    const bhGeo = new THREE.PlaneGeometry(5.8, 5.8);
    this.bhMat = new THREE.ShaderMaterial({
      vertexShader: GARGANTUA_VERT,
      fragmentShader: GARGANTUA_FRAG,
      uniforms: {
        uTime: { value: 0 },
        uSpinSpeed: { value: 1.0 },
        uOpacity: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
    });
    this.bhMesh = new THREE.Mesh(bhGeo, this.bhMat);
    this.bhRoot.add(this.bhMesh);

    const starGeo = new THREE.BufferGeometry();
    const sPos = new Float32Array(BH_STAR_COUNT * 3);
    const sOrbit = new Float32Array(BH_STAR_COUNT * 4);
    const sCol = new Float32Array(BH_STAR_COUNT);
    for (let i = 0; i < BH_STAR_COUNT; i += 1) {
      sOrbit[i * 4] = Math.random() * Math.PI * 2;
      sOrbit[i * 4 + 1] = Math.random();
      sOrbit[i * 4 + 2] = 0.55 + Math.random() * 1.15;
      sOrbit[i * 4 + 3] = 2.4 + Math.random() * 5.2;
      sCol[i] = Math.random();
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(sPos, 3));
    starGeo.setAttribute('aOrbit', new THREE.BufferAttribute(sOrbit, 4));
    starGeo.setAttribute('aColorMix', new THREE.BufferAttribute(sCol, 1));
    this.bhStarsMat = new THREE.ShaderMaterial({
      vertexShader: STAR_SWIRL_VERT,
      fragmentShader: STAR_SWIRL_FRAG,
      uniforms: {
        uTime: { value: 0 },
        uSpinSpeed: { value: 1.0 },
        uOpacity: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.bhStars = new THREE.Points(starGeo, this.bhStarsMat);
    this.bhStars.frustumCulled = false;
    this.bhRoot.add(this.bhStars);
  }

  private getOrBuildPlanet(idx: number): BuiltPlanet {
    const existing = this.builtPlanets[idx];
    if (existing) return existing;

    this.ensureCoreResources();
    const spec = PLANETS[idx];
    const root = new THREE.Group();
    root.rotation.z = spec.tiltZ;
    const spinGroup = new THREE.Group();
    root.add(spinGroup);

    const dissolveUniform = { value: 0 };
    const built: BuiltPlanet = { root, spinGroup, dissolveUniform };

    if (spec.id === 'EARTH') {
      const tex = createEarthTextures();
      const earthMat = new THREE.ShaderMaterial({
        vertexShader: EARTH_VERT,
        fragmentShader: EARTH_FRAG,
        uniforms: {
          uAlbedo: { value: tex.albedo },
          uSpecular: { value: tex.specular },
          uNight: { value: tex.night },
          uBump: { value: tex.bump },
          uClouds: { value: tex.clouds },
          uCloudOffsetU: { value: 0 },
          uLightDir: { value: this.lightDir },
          uOpacity: { value: 1 },
          uDissolve: dissolveUniform,
        },
        transparent: true,
      });
      spinGroup.add(new THREE.Mesh(this.sphereGeo!, earthMat));
      built.earthMat = earthMat;

      // Separate rotating 3D Cloud Sphere with matching Dissolve
      const cloudGeo = new THREE.SphereGeometry(1.024, 48, 36);
      const cloudMat = new THREE.MeshStandardMaterial({
        map: tex.clouds,
        transparent: true,
        opacity: 0.9,
        roughness: 0.92,
        metalness: 0.0,
        depthWrite: false,
      });
      injectDissolveOnStandardMat(cloudMat, dissolveUniform);
      const cloudMesh = new THREE.Mesh(cloudGeo, cloudMat);
      root.add(cloudMesh);
      built.cloudMesh = cloudMesh;

      // Orbiting 3D Moon
      const moonTex = createMoonTextures();
      const moonPivot = new THREE.Group();
      moonPivot.rotation.z = -0.22;
      const moonGeo = new THREE.SphereGeometry(0.24, 32, 24);
      const moonMat = new THREE.MeshStandardMaterial({
        map: moonTex.albedo,
        bumpMap: moonTex.bump,
        bumpScale: 0.035,
        roughness: 0.95,
        metalness: 0.0,
      });
      injectDissolveOnStandardMat(moonMat, dissolveUniform);
      const moonMesh = new THREE.Mesh(moonGeo, moonMat);
      moonMesh.position.set(1.88, 0.12, 0);
      moonPivot.add(moonMesh);
      root.add(moonPivot);
      built.moonPivot = moonPivot;
    } else if (spec.id === 'MARS') {
      const tex = createMarsTextures();
      const mat = new THREE.MeshStandardMaterial({
        map: tex.albedo,
        bumpMap: tex.bump,
        bumpScale: 0.06,
        roughnessMap: tex.roughness,
        roughness: 0.92,
        metalness: 0.04,
        transparent: true,
      });
      injectDissolveOnStandardMat(mat, dissolveUniform);
      spinGroup.add(new THREE.Mesh(this.sphereGeo!, mat));
      built.stdMat = mat;
    } else if (spec.id === 'SATURN') {
      const tex = createSaturnTextures();
      const mat = new THREE.MeshStandardMaterial({
        map: tex.albedo,
        roughness: 0.85,
        metalness: 0.0,
        transparent: true,
      });
      injectDissolveOnStandardMat(mat, dissolveUniform);
      spinGroup.add(new THREE.Mesh(this.sphereGeo!, mat));
      built.stdMat = mat;

      const ringGeo = new THREE.RingGeometry(1.32, 2.38, 96, 1);
      const posAttr = ringGeo.attributes.position;
      const uvAttr = ringGeo.attributes.uv;
      for (let i = 0; i < posAttr.count; i += 1) {
        const dist = Math.hypot(posAttr.getX(i), posAttr.getY(i));
        uvAttr.setXY(i, (dist - 1.32) / (2.38 - 1.32), 0.5);
      }
      uvAttr.needsUpdate = true;

      const localLight = this.lightDir
        .clone()
        .applyAxisAngle(new THREE.Vector3(0, 0, 1), -spec.tiltZ);
      const ringMat = new THREE.ShaderMaterial({
        vertexShader: SATURN_RING_VERT,
        fragmentShader: SATURN_RING_FRAG,
        uniforms: {
          uRingTex: { value: tex.ring },
          uLocalLightDir: { value: localLight },
          uOpacity: { value: 1 },
          uDissolve: dissolveUniform,
        },
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.rotation.x = Math.PI * 0.44;
      root.add(ringMesh);
      built.ringMat = ringMat;
    } else if (spec.id === 'JUPITER') {
      const tex = createJupiterTextures();
      const mat = new THREE.MeshStandardMaterial({
        map: tex.albedo,
        bumpMap: tex.bump,
        bumpScale: 0.032,
        roughness: 0.82,
        metalness: 0.0,
        transparent: true,
      });
      injectDissolveOnStandardMat(mat, dissolveUniform);
      spinGroup.add(new THREE.Mesh(this.sphereGeo!, mat));
      built.stdMat = mat;
    } else if (spec.id === 'NEPTUNE') {
      const tex = createNeptuneTexture();
      const mat = new THREE.MeshStandardMaterial({
        map: tex,
        roughness: 0.72,
        metalness: 0.05,
        transparent: true,
      });
      injectDissolveOnStandardMat(mat, dissolveUniform);
      spinGroup.add(new THREE.Mesh(this.sphereGeo!, mat));
      built.stdMat = mat;
    } else {
      const sunMat = new THREE.ShaderMaterial({
        vertexShader: SUN_VERT,
        fragmentShader: SUN_FRAG,
        uniforms: {
          uTime: { value: 0 },
          uOpacity: { value: 1 },
          uDissolve: dissolveUniform,
        },
        transparent: true,
      });
      spinGroup.add(new THREE.Mesh(this.sphereGeo!, sunMat));
      built.sunMat = sunMat;

      const coronaGeo = new THREE.SphereGeometry(1.24, 48, 36);
      const coronaMat = new THREE.ShaderMaterial({
        vertexShader: FRESNEL_VERT,
        fragmentShader: SUN_CORONA_FRAG,
        uniforms: {
          uTime: { value: 0 },
          uOpacity: { value: 1 },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      root.add(new THREE.Mesh(coronaGeo, coronaMat));
      built.sunCoronaMat = coronaMat;
    }

    if (spec.id !== 'SUN') {
      const atmoMat = new THREE.ShaderMaterial({
        vertexShader: FRESNEL_VERT,
        fragmentShader: FRESNEL_FRAG,
        uniforms: {
          uColor: { value: new THREE.Color(spec.atmoColor) },
          uLightDir: { value: this.lightDir },
          uIntensity: { value: spec.atmoIntensity },
          uPower: { value: 3.0 },
          uOpacity: { value: 1 },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      root.add(new THREE.Mesh(this.atmoGeo!, atmoMat));
      built.atmoMat = atmoMat;
    }

    root.visible = false;
    this.planetContainer.add(root);
    this.builtPlanets[idx] = built;
    return built;
  }

  private screenToWorld(px: number, py: number): { x: number; y: number; pxToWorld: number } {
    const halfH = Math.tan(THREE.MathUtils.degToRad(CAMERA_FOV_DEG * 0.5)) * CAMERA_Z;
    const halfW = halfH * (this.viewW / Math.max(this.viewH, 1));
    return {
      x: ((px / Math.max(this.viewW, 1)) * 2 - 1) * halfW,
      y: (1 - (py / Math.max(this.viewH, 1)) * 2) * halfH,
      pxToWorld: (halfH * 2) / Math.max(this.viewH, 1),
    };
  }

  private setPlanetVisuals(planet: BuiltPlanet, op: number, dissolveAmount: number): void {
    const d = Math.max(0, Math.min(1, dissolveAmount));
    planet.dissolveUniform.value = d;
    const atmoOp = op * Math.max(0, 1 - d * 1.25);
    if (planet.earthMat) planet.earthMat.uniforms.uOpacity.value = op;
    if (planet.stdMat) planet.stdMat.opacity = op;
    if (planet.atmoMat) planet.atmoMat.uniforms.uOpacity.value = atmoOp;
    if (planet.ringMat) planet.ringMat.uniforms.uOpacity.value = op;
    if (planet.sunMat) planet.sunMat.uniforms.uOpacity.value = op;
    if (planet.sunCoronaMat) planet.sunCoronaMat.uniforms.uOpacity.value = atmoOp;
    if (planet.cloudMesh) {
      (planet.cloudMesh.material as THREE.MeshStandardMaterial).opacity = op * 0.9;
    }
  }

  /**
   * Unified 3D Cinema Stage Renderer sharing ONE WebGL2 EffectComposer + UnrealBloomPass:
   * Handles Cosmos (6 Planets + Black Hole + GLSL 3D Dissolve) AND 3D Volumetric Lightning simultaneously!
   */
  public renderStage(params: {
    cosmosActive: boolean;
    cosmosFade: number;
    lightningActive: boolean;
    lightningFade: number;
    lightningEnter: number;
    fluidActive: boolean;
    fluidFade: number;
    hands: HandPose2D[];
    dt: number;
    viewW: number;
    viewH: number;
    overlayCtx: CanvasRenderingContext2D | null;
  }): void {
    const {
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
    } = params;

    this.resize(viewW, viewH);
    this.ensureCoreResources();
    this.time += dt;

    let anyCosmos = false;
    if (cosmosActive && cosmosFade > 0.01) {
      anyCosmos = this.updateCosmosScene(hands, dt, viewW, viewH, cosmosFade, overlayCtx);
    } else {
      this.planetContainer.visible = false;
      this.bhRoot.visible = false;
    }

    let anyLightning = false;
    if (lightningActive && lightningFade > 0.01) {
      anyLightning = this.lightning3D.update(
        hands,
        dt,
        lightningFade,
        lightningEnter,
        viewW,
        viewH,
        (px, py) => this.screenToWorld(px, py),
        overlayCtx,
      );
    } else {
      this.lightning3D.clear();
    }

    let anyFluid = false;
    if (fluidActive && fluidFade > 0.01) {
      anyFluid = this.fluidSandbox.update(
        hands,
        dt,
        fluidFade,
        viewW,
        viewH,
        (px, py) => this.screenToWorld(px, py),
      );
    } else {
      this.fluidSandbox.clear();
    }

    if (anyCosmos || anyLightning || anyFluid) {
      this.renderer.setRenderTarget(this.sceneRT);
      this.renderer.clear();
      this.renderer.render(this.scene, this.camera);
      this.renderer.setRenderTarget(null);
      this.composer.render();
      this.hasRenderedFrame = true;
    } else {
      this.clear();
    }
  }

  private updateCosmosScene(
    hands: HandPose2D[],
    dt: number,
    viewW: number,
    viewH: number,
    fadeProgress: number,
    overlayCtx: CanvasRenderingContext2D | null,
  ): boolean {
    this.planetContainer.visible = true;
    if (this.switchCooldown > 0) this.switchCooldown = Math.max(0, this.switchCooldown - dt);
    if (this.hudTimer > 0) this.hudTimer = Math.max(0, this.hudTimer - dt);

    const modeAlpha = Math.min(1, Math.max(0, fadeProgress));

    // ---- 1. Strict Anti-Accidental Black Hole Gating (4+ fingers, dist >= 0.28, 450ms hold, 400ms exit hysteresis) ----
    let normPalmDist = 0;
    let bothWideOpen = false;
    let bothFists = false;
    if (hands.length >= 2) {
      const h0 = hands[0];
      const h1 = hands[1];
      normPalmDist = Math.hypot(
        (h1.palmX - h0.palmX) / Math.max(viewW, 1),
        (h1.palmY - h0.palmY) / Math.max(viewH, 1),
      );
      const f0 = countExtendedFingers(h0);
      const f1 = countExtendedFingers(h1);
      bothWideOpen =
        f0 >= BH_MIN_OPEN_FINGERS &&
        f1 >= BH_MIN_OPEN_FINGERS &&
        normPalmDist >= BH_MIN_NORM_DIST;
      bothFists = h0.gesture === 'FIST' && h1.gesture === 'FIST';
    }

    if (!this.bhActive) {
      if (bothWideOpen) {
        this.bhChargeSec = Math.min(BH_CHARGE_SEC, this.bhChargeSec + dt);
        if (this.bhChargeSec >= BH_CHARGE_SEC) {
          this.bhActive = true;
          this.bhExitSec = 0;
        }
      } else {
        this.bhChargeSec = Math.max(0, this.bhChargeSec - dt * 2.5);
      }
    } else {
      // Exit hysteresis (400ms): exit only if 1 hand is lowered (<2 hands) or both hands clench FIST
      const wantsExit = hands.length < 2 || bothFists;
      if (wantsExit) {
        this.bhExitSec += dt;
        if (this.bhExitSec >= BH_EXIT_HYSTERESIS_SEC) {
          this.bhActive = false;
          this.bhChargeSec = 0;
          this.bhExitSec = 0;
          // Burst Hawking stardust wisps as Gargantua implodes away
          this.spawnAstralWisps(this.bhX || viewW * 0.5, this.bhY || viewH * 0.5, 110 * this.bhScale, 36, 0.72);
        }
      } else {
        this.bhExitSec = Math.max(0, this.bhExitSec - dt * 2.0);
      }
    }

    // Draw thin gravitational charge-up ring converging between both palms during the 450ms hold
    if (!this.bhActive && this.bhChargeSec > 0.02 && hands.length >= 2 && overlayCtx) {
      const chargeT = Math.min(1, this.bhChargeSec / BH_CHARGE_SEC);
      const mx = (hands[0].palmX + hands[1].palmX) * 0.5;
      const my = (hands[0].palmY + hands[1].palmY) * 0.5;
      this.drawGravityChargeRing(overlayCtx, mx, my, chargeT, modeAlpha);
    }

    const targetBh = this.bhActive ? 1 : 0;
    this.bhBlend += (targetBh - this.bhBlend) * (1 - Math.exp(-5.8 * dt));

    // ---- 2. Single-Hand Celestial Body Physics + 2-Finger Switch + Ethereal Dissolve ----
    let showPlanet = false;

    if (hands.length >= 1) {
      const h = hands[0];
      const twoFinger = !this.bhActive && isTwoFingerPose(h);

      if (twoFinger && !this.wasTwoFinger && this.switchCooldown <= 0 && this.switchTimer < 0) {
        this.pendingPlanetIdx = (this.planetIdx + 1) % PLANETS.length;
        this.switchTimer = 0;
        this.switchedMidpoint = false;
        this.switchCooldown = SWITCH_COOLDOWN_SEC;
        this.burstProgress = 0;
        this.burstConverge = 0; // Outward Curl-Noise stardust dissolve
        this.hudTimer = HUD_DISPLAY_SEC;
        this.spawnAstralWisps(this.posX || viewW * 0.5, this.posY || viewH * 0.45, this.lastPlanetRadiusPx, 28, 0.45);
      }
      this.wasTwoFinger = twoFinger;

      if (!this.bhActive && (isIndexHoverPose(h) || twoFinger)) {
        showPlanet = true;
        // Reverse materialize: if hand just re-entered, coalesce stardust inward into 3D planet!
        if (!this.wasShowingPlanet) {
          this.planetMaterializeT = 0;
          this.burstProgress = 0;
          this.burstConverge = 1;
        }
        this.planetDissolveT = 1;
        const targetX = twoFinger ? (h.pts[16] + h.pts[24]) * 0.5 : h.pts[16];
        const targetY =
          (twoFinger ? (h.pts[17] + h.pts[25]) * 0.5 : h.pts[17]) - h.palmScale * 1.12;

        if (!this.initializedPos) {
          this.posX = targetX;
          this.posY = targetY;
          this.prevAnchorX = targetX;
          this.initializedPos = true;
          this.hudTimer = Math.max(this.hudTimer, HUD_DISPLAY_SEC * 0.8);
        }

        const stiffness = 64;
        const damping = 11.2;
        this.velX += ((targetX - this.posX) * stiffness - this.velX * damping) * dt;
        this.velY += ((targetY - this.posY) * stiffness - this.velY * damping) * dt;
        this.posX += this.velX * dt;
        this.posY += this.velY * dt;

        const moveDx = (targetX - this.prevAnchorX) / Math.max(dt, 0.008);
        this.prevAnchorX = targetX;
        const spec = PLANETS[this.planetIdx];
        this.spinVel += Math.max(-7, Math.min(7, moveDx * 0.0075)) * dt * 8.5;
        this.spinVel += (spec.baseSpin - this.spinVel) * (1 - Math.exp(-2.4 * dt));
        this.spinY += this.spinVel * dt;

        this.lastPlanetRadiusPx = Math.max(58, Math.min(132, h.palmScale * 0.86 * spec.radiusMul));
      }
    } else {
      this.wasTwoFinger = false;
    }

    // Trigger 650ms GLSL Simplex Dissolve + 450 Curl-Noise Stardust at planet's LAST position
    if (this.wasShowingPlanet && !showPlanet && !this.bhActive && this.planetAlpha > 0.12) {
      this.planetDissolveT = 0;
      this.burstProgress = 0;
      this.burstConverge = 0;
      this.spawnAstralWisps(this.posX, this.posY, this.lastPlanetRadiusPx, 36, 0.48);
    }
    this.wasShowingPlanet = showPlanet;

    let dissolveUniformVal = 0;
    let dissolveScale = 1;
    let dissolveAlpha = 1;
    if (!showPlanet) {
      if (this.planetDissolveT < 1) {
        this.planetDissolveT = Math.min(1, this.planetDissolveT + dt / PLANET_DISSOLVE_SEC);
        const k = this.planetDissolveT;
        // Burn surface away via GLSL 3D noise while levitating slightly — never shrinks into a solid lump!
        this.posY -= dt * (46 + k * 42);
        this.spinY += dt * (2.6 + k * 3.8);
        dissolveUniformVal = Math.pow(k, 0.88);
        dissolveScale = 1 + k * 0.08;
        dissolveAlpha = Math.max(0.02, 1 - Math.pow(k, 2.2));
      } else {
        this.initializedPos = false;
        dissolveAlpha = 0;
        dissolveUniformVal = 1;
      }
    } else if (this.planetMaterializeT < 1) {
      this.planetMaterializeT = Math.min(1, this.planetMaterializeT + dt / 0.46);
      dissolveUniformVal = Math.pow(1 - this.planetMaterializeT, 1.4);
    }

    const planetScreenX = this.initializedPos ? this.posX : viewW * 0.5;
    const planetScreenY = this.initializedPos ? this.posY : viewH * 0.45;
    const planetRadiusPx = this.lastPlanetRadiusPx;

    const handConf = hands[0]?.confidence ?? 1;
    if (showPlanet && !this.bhActive) {
      this.planetAlpha += (handConf - this.planetAlpha) * (1 - Math.exp(-9.5 * dt));
    } else if (this.planetDissolveT < 1) {
      this.planetAlpha = dissolveAlpha;
    } else {
      this.planetAlpha = 0;
    }

    let switchScale = 1;
    let switchAlpha = 1;
    let switchDissolve = 0;
    let switchFlashT = -1;
    if (this.switchTimer >= 0) {
      this.switchTimer += dt;
      const t = this.switchTimer / SWITCH_DURATION_SEC;
      switchFlashT = t;
      if (t >= 1) {
        this.switchTimer = -1;
        switchScale = 1;
        switchAlpha = 1;
        switchDissolve = 0;
      } else if (t < 0.48) {
        // Outgoing planet burns into glowing cyan/gold patches via GLSL uDissolve (0 -> 1)
        const k = t / 0.48;
        switchDissolve = Math.pow(k, 0.85);
        switchScale = 1 + k * 0.06;
        switchAlpha = Math.max(0.05, 1 - k * 0.35);
        this.spinY += dt * 7.5;
      } else {
        if (!this.switchedMidpoint) {
          this.switchedMidpoint = true;
          this.planetIdx = this.pendingPlanetIdx;
          this.spinVel = Math.max(PLANETS[this.planetIdx].baseSpin + 3.2, this.spinVel);
          // Reverse stardust coalesce into the incoming planet!
          this.burstProgress = 0;
          this.burstConverge = 1;
        }
        const k = (t - 0.48) / 0.52;
        switchDissolve = Math.pow(1 - k, 1.35);
        switchScale = elasticScale02To10(Math.min(1, 0.35 + k * 0.65));
        switchAlpha = Math.min(1, Math.pow(k, 0.55));
      }
    }
    const finalDissolve = Math.max(dissolveUniformVal, switchDissolve);

    // Crisp luminous Energy Flash Ring during 420ms 3D planet transition
    if (overlayCtx && switchFlashT >= 0 && switchFlashT < 1 && this.bhBlend < 0.2) {
      const ringAlpha = Math.sin(switchFlashT * Math.PI) * modeAlpha;
      const ringR = planetRadiusPx * (0.45 + switchFlashT * 1.45);
      overlayCtx.save();
      overlayCtx.globalCompositeOperation = 'lighter';
      overlayCtx.strokeStyle = `rgba(103, 232, 249, ${ringAlpha * 0.78})`;
      overlayCtx.lineWidth = 3.2 * (1 - switchFlashT) + 0.8;
      overlayCtx.beginPath();
      overlayCtx.arc(planetScreenX, planetScreenY, ringR, 0, Math.PI * 2);
      overlayCtx.stroke();

      overlayCtx.strokeStyle = `rgba(255, 255, 255, ${ringAlpha * 0.9})`;
      overlayCtx.lineWidth = 1.2;
      overlayCtx.beginPath();
      overlayCtx.arc(planetScreenX, planetScreenY, ringR * 0.86, 0, Math.PI * 2);
      overlayCtx.stroke();
      overlayCtx.restore();
    }

    // Mode ENTER Cosmos light-warp expansion ring
    if (overlayCtx && modeAlpha > 0.05 && modeAlpha < 0.96 && hands.length > 0) {
      const warpAlpha = Math.sin(modeAlpha * Math.PI) * 0.72;
      const warpR = 30 + modeAlpha * 160;
      const cx = hands[0].palmX;
      const cy = hands[0].palmY;
      overlayCtx.save();
      overlayCtx.globalCompositeOperation = 'lighter';
      overlayCtx.strokeStyle = `rgba(125, 211, 252, ${warpAlpha})`;
      overlayCtx.lineWidth = 2.4 * (1 - modeAlpha) + 0.8;
      overlayCtx.beginPath();
      overlayCtx.arc(cx, cy, warpR, 0, Math.PI * 2);
      overlayCtx.stroke();
      overlayCtx.restore();
    }

    if (this.burstProgress < 1 && this.burstPoints && this.burstMat) {
      this.burstProgress = Math.min(1, this.burstProgress + dt / PLANET_DISSOLVE_SEC);
      this.burstPoints.visible = this.burstProgress < 1;
      this.burstMat.uniforms.uProgress.value = this.burstProgress;
      this.burstMat.uniforms.uOpacity.value = modeAlpha;
      this.burstMat.uniforms.uConverge.value = this.burstConverge;
    } else if (this.burstPoints) {
      this.burstPoints.visible = false;
    }

    // ---- 3. Active 3D Planet Mesh & Spaghettification Morph ----
    for (let i = 0; i < this.builtPlanets.length; i += 1) {
      const bp = this.builtPlanets[i];
      if (bp) bp.root.visible = false;
    }

    const activePlanet = this.getOrBuildPlanet(this.planetIdx);
    const spagVis = this.bhBlend > 0.01 && this.bhBlend < 0.68 ? 1 - this.bhBlend / 0.68 : 0;
    const effectivePlanetVis =
      Math.max(this.planetAlpha * switchAlpha * (1 - this.bhBlend), spagVis) * modeAlpha;

    if (effectivePlanetVis > 0.015 || this.burstProgress < 1) {
      const worldP = this.screenToWorld(planetScreenX, planetScreenY);
      const bobY = Math.sin(this.time * 2.1) * 0.06;
      const baseScale = planetRadiusPx * worldP.pxToWorld * switchScale * dissolveScale;

      if (this.bhBlend > 0.02) {
        const bhWorld = this.screenToWorld(this.bhX || viewW * 0.5, this.bhY || viewH * 0.5);
        const pullT = Math.min(1, this.bhBlend * 1.45);
        this.planetContainer.position.set(
          THREE.MathUtils.lerp(worldP.x, bhWorld.x, pullT),
          THREE.MathUtils.lerp(worldP.y + bobY, bhWorld.y, pullT),
          0,
        );
        this.planetContainer.rotation.z = this.bhRoll;
        const stretch = 1 + pullT * 2.4;
        const pinch = Math.max(0.05, 1 - pullT * 0.92);
        this.planetContainer.scale.set(baseScale * stretch, baseScale * pinch, baseScale * pinch);
      } else {
        this.planetContainer.position.set(worldP.x, worldP.y + bobY, 0);
        this.planetContainer.rotation.z = 0;
        this.planetContainer.scale.setScalar(Math.max(0.04, baseScale));
      }

      activePlanet.root.visible = effectivePlanetVis > 0.015;
      activePlanet.spinGroup.rotation.y = this.spinY;
      if (activePlanet.cloudMesh) {
        const cloudSpin = this.spinY * 1.18 + this.time * 0.06;
        activePlanet.cloudMesh.rotation.y = cloudSpin;
        if (activePlanet.earthMat) {
          activePlanet.earthMat.uniforms.uCloudOffsetU.value =
            ((cloudSpin - this.spinY) / (Math.PI * 2)) % 1;
        }
      }
      if (activePlanet.moonPivot) {
        activePlanet.moonPivot.rotation.y = this.time * 0.72;
      }
      if (activePlanet.sunMat) activePlanet.sunMat.uniforms.uTime.value = this.time;
      if (activePlanet.sunCoronaMat) activePlanet.sunCoronaMat.uniforms.uTime.value = this.time;
      this.setPlanetVisuals(activePlanet, effectivePlanetVis, finalDissolve);

      if (overlayCtx && this.hudTimer > 0 && this.bhBlend < 0.2 && effectivePlanetVis > 0.25) {
        const hudAlpha =
          Math.min(1, this.hudTimer / 0.45) *
          Math.min(1, (HUD_DISPLAY_SEC - this.hudTimer + 0.18) / 0.25) *
          effectivePlanetVis;
        if (hudAlpha > 0.02) {
          this.drawPlanetHudTag(
            overlayCtx,
            planetScreenX,
            planetScreenY,
            planetRadiusPx,
            PLANETS[this.planetIdx].tag,
            hudAlpha,
          );
        }
      }
    }

    // ---- 4. Update Interstellar Gargantua Black Hole + 1,500 Spiraling 3D Stars ----
    if (this.bhActive && hands.length >= 2) {
      const h0 = hands[0];
      const h1 = hands[1];
      const targetBhX = (h0.palmX + h1.palmX) * 0.5;
      const targetBhY = (h0.palmY + h1.palmY) * 0.5;
      const distPx = Math.hypot(h1.palmX - h0.palmX, h1.palmY - h0.palmY);
      const targetRoll = -Math.atan2(h1.palmY - h0.palmY, h1.palmX - h0.palmX);
      const targetScale = Math.max(0.65, Math.min(1.85, distPx / 280));
      const targetSpin = 0.85 + Math.min(1.8, distPx / 220);

      const sm = 1 - Math.exp(-10 * dt);
      if (this.bhX === 0 && this.bhY === 0) {
        this.bhX = targetBhX;
        this.bhY = targetBhY;
      } else {
        this.bhX += (targetBhX - this.bhX) * sm;
        this.bhY += (targetBhY - this.bhY) * sm;
      }
      this.bhScale += (targetScale - this.bhScale) * sm;
      this.bhSpinSpeed += (targetSpin - this.bhSpinSpeed) * sm;

      let dAngle = targetRoll - this.bhRoll;
      while (dAngle > Math.PI) dAngle -= Math.PI * 2;
      while (dAngle < -Math.PI) dAngle += Math.PI * 2;
      this.bhRoll += dAngle * sm;
    }

    const bhAlpha = Math.max(0, (this.bhBlend - 0.14) / 0.86) * modeAlpha;
    if (bhAlpha > 0.01 && this.bhMat && this.bhStarsMat) {
      const bhWorld = this.screenToWorld(this.bhX || viewW * 0.5, this.bhY || viewH * 0.5);
      // Implode cleanly toward singularity (0.12 -> 1.0) when retracting hands instead of fading flat
      const implodeCurve = Math.pow(bhAlpha, 0.72);
      const worldScale = bhWorld.pxToWorld * 76 * this.bhScale * (0.12 + 0.88 * implodeCurve);
      const exitSpinBoost = this.bhActive ? 1 : 2.6;

      this.bhRoot.visible = true;
      this.bhRoot.position.set(bhWorld.x, bhWorld.y, 0);
      this.bhRoot.rotation.z = this.bhRoll;
      this.bhRoot.scale.setScalar(worldScale);

      this.bhMat.uniforms.uTime.value = this.time;
      this.bhMat.uniforms.uSpinSpeed.value = this.bhSpinSpeed * exitSpinBoost;
      this.bhMat.uniforms.uOpacity.value = bhAlpha;

      this.bhStarsMat.uniforms.uTime.value = this.time;
      this.bhStarsMat.uniforms.uSpinSpeed.value = this.bhSpinSpeed * exitSpinBoost;
      this.bhStarsMat.uniforms.uOpacity.value = bhAlpha;

      if (overlayCtx && bhAlpha > 0.35) {
        this.drawPlanetHudTag(
          overlayCtx,
          this.bhX,
          this.bhY,
          Math.min(180, 124 * this.bhScale),
          'SINGULARITY // GARGANTUA - RELATIVISTIC DISK',
          bhAlpha * 0.85,
        );
      }
    } else {
      this.bhRoot.visible = false;
    }

    // ---- 5. Render Rising 2D Astral Stardust Wisps on Overlay ----
    if (overlayCtx) {
      let anyWisp = false;
      for (let i = 0; i < 48; i += 1) {
        if (this.wispLife[i] < this.wispMax[i]) {
          anyWisp = true;
          break;
        }
      }
      if (anyWisp) {
        overlayCtx.save();
        overlayCtx.globalCompositeOperation = 'lighter';
        for (let i = 0; i < 48; i += 1) {
          if (this.wispLife[i] >= this.wispMax[i]) continue;
          this.wispLife[i] += dt;
          if (this.wispLife[i] >= this.wispMax[i]) continue;
          const frac = this.wispLife[i] / this.wispMax[i];
          const a = Math.sin(frac * Math.PI) * (1 - frac * 0.35);
          // Swirling gentle drift upward
          this.wispVx[i] *= 1 - Math.min(0.3, dt * 1.8);
          this.wispVy[i] -= 28 * dt;
          this.wispX[i] += this.wispVx[i] * dt + Math.sin(this.time * 6 + i) * 18 * dt;
          this.wispY[i] += this.wispVy[i] * dt;
          const r = (1.6 + (1 - frac) * 2.6);
          overlayCtx.fillStyle = this.wispGold[i]
            ? `rgba(253, 224, 152, ${a * 0.9})`
            : `rgba(125, 211, 252, ${a * 0.9})`;
          overlayCtx.beginPath();
          overlayCtx.arc(this.wispX[i], this.wispY[i], r, 0, Math.PI * 2);
          overlayCtx.fill();
        }
        overlayCtx.restore();
      }
    }

    return effectivePlanetVis > 0.015 || bhAlpha > 0.01 || this.burstProgress < 1;
  }

  private drawGravityChargeRing(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    progress: number,
    alpha: number,
  ): void {
    const r = 96 - progress * 58;
    const sweep = progress * Math.PI * 2;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = alpha * (0.35 + progress * 0.65);

    ctx.strokeStyle = 'rgba(251, 191, 36, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(255, 245, 220, 0.92)';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI * 0.5, -Math.PI * 0.5 + sweep);
    ctx.stroke();

    // Converging micro-spokes pulling energy inward
    for (let i = 0; i < 6; i += 1) {
      const a = this.time * 3.2 + (i * Math.PI) / 3;
      const rOut = r + 24 * (1 - progress * 0.6);
      const rIn = Math.max(8, r - 10);
      ctx.strokeStyle = 'rgba(103, 232, 249, 0.55)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * rOut, cy + Math.sin(a) * rOut);
      ctx.lineTo(cx + Math.cos(a) * rIn, cy + Math.sin(a) * rIn);
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawPlanetHudTag(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radiusPx: number,
    label: string,
    alpha: number,
  ): void {
    ctx.save();
    ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
    const anchorX = cx + radiusPx * 0.78;
    const anchorY = cy - radiusPx * 0.56;
    const elbowX = anchorX + 20;
    const elbowY = anchorY - 14;

    ctx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    const textW = ctx.measureText(label).width;
    const lineEndX = elbowX + textW + 14;

    ctx.strokeStyle = 'rgba(103, 232, 249, 0.72)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(anchorX, anchorY);
    ctx.lineTo(elbowX, elbowY);
    ctx.lineTo(lineEndX, elbowY);
    ctx.stroke();

    ctx.fillStyle = 'rgba(224, 242, 254, 0.95)';
    ctx.beginPath();
    ctx.arc(anchorX, anchorY, 2.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = 'rgba(8, 12, 20, 0.65)';
    ctx.fillRect(elbowX + 2, elbowY - 18, textW + 12, 16);

    ctx.fillStyle = '#e0f2fe';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, elbowX + 8, elbowY - 10);
    ctx.restore();
  }

  dispose(): void {
    this.disposed = true;
    if (this.prewarmTimerId !== null) {
      window.clearTimeout(this.prewarmTimerId);
      this.prewarmTimerId = null;
    }
    this.scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Points) {
        obj.geometry.dispose();
        const mat = obj.material;
        if (Array.isArray(mat)) {
          mat.forEach((m) => m.dispose());
        } else {
          mat.dispose();
        }
      }
    });
    this.lightning3D.dispose();
    this.fluidSandbox.dispose();
    this.sceneRT.dispose();
    this.bloomPass.dispose();
    this.composer.dispose();
    this.renderer.dispose();
  }
}
