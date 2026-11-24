import * as THREE from 'three';
import { RunnerAudioEngine } from '../audio/runnerAudio';
import type { BodyControlState, RunnerLane } from '../vision/bodyController';

export type RunnerPhase = 'READY' | 'CALIBRATING' | 'PLAYING' | 'GAME_OVER';

export interface RunnerHudSnapshot {
  phase: RunnerPhase;
  score: number;
  coins: number;
  bestScore: number;
  speedMultiplier: number;
  countdownNum: number;
  lane: RunnerLane;
  actionLabel: string;
  hasBody: boolean;
  isCalibrated: boolean;
  handsUpProgress: number;
  distance: number;
}

type ObstacleKind = 'LOW_BARRIER' | 'HIGH_GATE' | 'TRAIN';

interface PooledObstacle {
  active: boolean;
  kind: ObstacleKind;
  lane: RunnerLane;
  z: number;
  halfLen: number;
  mesh: THREE.Group;
}

interface PooledCoin {
  active: boolean;
  lane: RunnerLane;
  x: number;
  y: number;
  z: number;
  mesh: THREE.Mesh;
}

const LANE_WIDTH = 2.65;
const BASE_SPEED = 24;
const MAX_SPEED = 52;
const JUMP_VY = 16.6;
const GRAVITY = 44.5;
const FAST_DROP_VY = -30;
const MIN_SLIDE_SEC = 0.65;

const ARCH_COUNT = 12;
const ARCH_SPACING = 11.5;
const OBSTACLE_POOL_PER_KIND = 9;
const COIN_POOL_SIZE = 54;
const SPARK_POOL = 100;
const BEST_STORAGE_KEY = 'visionarcade.runnerBest';

// ============================================================================
// Glossy Neon Road Shader with Real-Time Specular Neon Reflections
// ============================================================================
const ROAD_VERT = `
varying vec3 vWorldPos;
varying vec2 vUv;
void main() {
  vUv = uv;
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const ROAD_FRAG = `
uniform float uScrollZ;
uniform float uSpeedNorm;
uniform vec4 uReflLights[4]; // x, z, colorType (0=cyan, 1=magenta, 2=orange, 3=gold), intensity
varying vec3 vWorldPos;
varying vec2 vUv;

vec3 reflColor(float id) {
  if (id < 0.5) return vec3(0.14, 0.92, 1.45);
  if (id < 1.5) return vec3(1.38, 0.22, 1.18);
  if (id < 2.5) return vec3(1.55, 0.42, 0.14);
  return vec3(1.55, 1.24, 0.34);
}

void main() {
  float x = vWorldPos.x;
  float z = vWorldPos.z;
  float distFade = exp(-max(0.0, -z) * 0.0175);

  // Dark wet cyber-asphalt base with subtle perspective sheen
  vec3 base = mix(vec3(0.022, 0.024, 0.058), vec3(0.055, 0.026, 0.11), clamp((-z) / 115.0, 0.0, 1.0));

  // Perspective synthwave grid lines
  float gz = fract((z - uScrollZ) * 0.28);
  float gx = fract((x + 4.8) * 0.377);
  float lineZ = smoothstep(0.055, 0.0, min(gz, 1.0 - gz));
  float lineX = smoothstep(0.032, 0.0, min(gx, 1.0 - gx));
  float grid = max(lineZ * 0.85, lineX * 0.55) * distFade;
  vec3 gridCol = vec3(0.14, 0.52, 1.15) * grid * 0.42;

  // Glowing cyan lane dividers at x = +/- 1.325
  float divL = exp(-pow((abs(x) - 1.325) / 0.052, 2.0));
  vec3 divCol = vec3(0.16, 0.98, 1.48) * divL * (0.78 + uSpeedNorm * 0.35) * distFade;

  // Glowing magenta outer track curbs at x = +/- 4.2
  float curb = exp(-pow((abs(x) - 4.22) / 0.11, 2.0));
  vec3 curbCol = vec3(1.48, 0.22, 1.12) * curb * 1.15 * distFade;

  // Center horizon sun reflection streak shimmering down the wet center road
  float sunStreak = exp(-pow(x / 1.9, 2.0)) * smoothstep(-6.0, -95.0, z) * 0.22;
  vec3 sunRefl = vec3(1.15, 0.28, 0.62) * sunStreak;

  // Dynamic glossy neon reflections cast by Runner + nearby Trains / Gates / Barriers
  vec3 dynRefl = vec3(0.0);
  for (int i = 0; i < 4; i++) {
    vec4 rl = uReflLights[i];
    if (rl.w > 0.01) {
      float dx = x - rl.x;
      float dz = (z - rl.y) * 0.22; // stretched longitudinally along wet road
      float glow = exp(-(dx * dx * 0.85 + dz * dz * 0.45)) * rl.w;
      dynRefl += reflColor(rl.z) * glow * 0.46 * distFade;
    }
  }

  vec3 col = base + gridCol + divCol + curbCol + sunRefl + dynRefl;
  gl_FragColor = vec4(col, 1.0);
}
`;

// ============================================================================
// Synthwave Sliced Horizon Sun Shader
// ============================================================================
const SUN_DISC_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const SUN_DISC_FRAG = `
uniform float uTime;
varying vec2 vUv;
void main() {
  vec2 p = (vUv - 0.5) * 2.0;
  float r = length(p);
  if (r > 1.0) discard;

  // Classic Outrun horizontal scanline slices in the lower half of the sun
  float sliceMask = 1.0;
  if (p.y < 0.05) {
    float wave = sin((p.y - uTime * 0.03) * 44.0);
    float cut = smoothstep(-0.15, -0.78, p.y);
    sliceMask = step(cut * 0.68 - 0.25, wave);
  }

  vec3 cTop = vec3(1.18, 0.88, 0.24);
  vec3 cMid = vec3(1.12, 0.30, 0.22);
  vec3 cBot = vec3(0.96, 0.10, 0.56);
  vec3 col = mix(cBot, cMid, smoothstep(-0.85, 0.05, p.y));
  col = mix(col, cTop, smoothstep(0.05, 0.85, p.y));

  float edgeSoft = 1.0 - smoothstep(0.92, 1.0, r);
  float alpha = edgeSoft * sliceMask;
  if (alpha < 0.02) discard;
  gl_FragColor = vec4(col, alpha);
}
`;

const SPARK_VERT = `
attribute float aAlpha;
attribute float aSize;
attribute vec3 aColor;
varying float vAlpha;
varying vec3 vColor;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  vAlpha = aAlpha;
  vColor = aColor;
  gl_PointSize = clamp(aSize * (14.0 / max(-mv.z, 1.0)), 1.5, 14.0);
}
`;

const SPARK_FRAG = `
varying float vAlpha;
varying vec3 vColor;
void main() {
  if (vAlpha <= 0.01) discard;
  float d = length(gl_PointCoord - 0.5);
  if (d > 0.5) discard;
  float g = exp(-d * d * 14.0);
  gl_FragColor = vec4(vColor * g, g * vAlpha);
}
`;

function loadBestScore(): number {
  try {
    const raw = window.localStorage.getItem(BEST_STORAGE_KEY);
    const parsed = raw ? Number.parseInt(raw, 10) : 0;
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
  } catch {
    return 0;
  }
}

function saveBestScore(score: number): void {
  try {
    window.localStorage.setItem(BEST_STORAGE_KEY, String(Math.floor(score)));
  } catch {
    // ignore private browsing quota errors
  }
}

/**
 * Ultra 3D Neon Subway Runner (Three.js sharing Cosmos3DEngine's EffectComposer + Half-Res UnrealBloomPass):
 * - Synthwave Horizon Sun + Cyber Skyline + Glossy Neon Road Reflections
 * - Dynamic FOV (60° -> 73°) + Smooth Camera Banking into lane switches + Edge Speed Lines
 * - Articulated 3D Cyber Runner (lane strafe bank, jump flip, crouch slide + mid-air Fast Drop)
 * - Guaranteed-solvable Subway Surfers obstacle waves (Low Barrier, High Laser Gate, Neon Train) + Coins
 */
export class Runner3DGame {
  public readonly scene = new THREE.Scene();
  public readonly camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 185);

  private audio = new RunnerAudioEngine();

  private phase: RunnerPhase = 'READY';
  private score = 0;
  private coins = 0;
  private bestScore = loadBestScore();
  private distance = 0;
  private speed = BASE_SPEED;
  private elapsed = 0;

  // Calibration countdown (3 -> 2 -> 1 -> GO)
  private calibTimer = 0;
  private lastCountdownBeep = 4;

  // Player kinematics
  private lane: RunnerLane = 0;
  private playerX = 0;
  private playerY = 0;
  private playerVy = 0;
  private isGrounded = true;
  private slideTimer = 0;
  private isSliding = false;
  private jumpFlipAngle = 0;
  private runCycle = 0;
  private camBankZ = 0;
  private camShake = 0;
  private actionLabel = 'READY';

  // Character rig
  private runnerRoot = new THREE.Group();
  private torsoGroup = new THREE.Group();
  private leftArm = new THREE.Group();
  private rightArm = new THREE.Group();
  private leftLeg = new THREE.Group();
  private rightLeg = new THREE.Group();

  // Track & environment
  private roadMat: THREE.ShaderMaterial;
  private sunMat: THREE.ShaderMaterial;
  private reflLights: [THREE.Vector4, THREE.Vector4, THREE.Vector4, THREE.Vector4];
  private arches: THREE.Group[] = [];
  private scrollZ = 0;

  // Pooled obstacles & coins (Zero-GC during play)
  private obstacles: PooledObstacle[] = [];
  private coinPool: PooledCoin[] = [];
  private nextWaveDist = 26;
  private lastSafeLane: RunnerLane = 0;

  // Pooled 3D sparks
  private sparkGeo: THREE.BufferGeometry;
  private sparkMat: THREE.ShaderMaterial;
  private sPos = new Float32Array(SPARK_POOL * 3);
  private sCol = new Float32Array(SPARK_POOL * 3);
  private sAlpha = new Float32Array(SPARK_POOL);
  private sSize = new Float32Array(SPARK_POOL);
  private svx = new Float32Array(SPARK_POOL);
  private svy = new Float32Array(SPARK_POOL);
  private svz = new Float32Array(SPARK_POOL);
  private sLife = new Float32Array(SPARK_POOL);
  private sMax = new Float32Array(SPARK_POOL);
  private sparkCursor = 0;

  private latestBodyHas = false;
  private latestBodyCalibrated = false;
  private latestHandsUp = 0;

  constructor() {
    this.scene.fog = new THREE.FogExp2(0x050512, 0.0052);
    this.camera.position.set(0, 4.3, 8.6);
    this.camera.lookAt(0, 1.5, -14);

    // Studio cyberpunk lighting
    const hemi = new THREE.HemisphereLight(0x38bdf8, 0x3b0764, 0.95);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xe0f2fe, 1.55);
    key.position.set(6, 14, 8);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xf43f97, 1.15);
    rim.position.set(-8, 6, -18);
    this.scene.add(rim);

    // 1. Glossy Reflective Neon Road
    this.reflLights = [
      new THREE.Vector4(0, 0, 0, 1.0),
      new THREE.Vector4(0, -30, 0, 0),
      new THREE.Vector4(0, -30, 0, 0),
      new THREE.Vector4(0, -30, 0, 0),
    ];
    this.roadMat = new THREE.ShaderMaterial({
      uniforms: {
        uScrollZ: { value: 0 },
        uSpeedNorm: { value: 0 },
        uReflLights: { value: this.reflLights },
      },
      vertexShader: ROAD_VERT,
      fragmentShader: ROAD_FRAG,
    });
    const roadMesh = new THREE.Mesh(new THREE.PlaneGeometry(11.6, 175), this.roadMat);
    roadMesh.rotation.x = -Math.PI * 0.5;
    roadMesh.position.set(0, 0, -74);
    this.scene.add(roadMesh);

    // Subtle dark outer void floor planes on left & right of track
    const sideFloorMat = new THREE.MeshBasicMaterial({ color: 0x050510 });
    const leftSide = new THREE.Mesh(new THREE.PlaneGeometry(60, 175), sideFloorMat);
    leftSide.rotation.x = -Math.PI * 0.5;
    leftSide.position.set(-35.8, -0.02, -74);
    this.scene.add(leftSide);
    const rightSide = new THREE.Mesh(new THREE.PlaneGeometry(60, 175), sideFloorMat);
    rightSide.rotation.x = -Math.PI * 0.5;
    rightSide.position.set(35.8, -0.02, -74);
    this.scene.add(rightSide);

    // 2. Synthwave Sliced Horizon Sun + Halo + Cyber City Skyline
    this.sunMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 } },
      vertexShader: SUN_DISC_VERT,
      fragmentShader: SUN_DISC_FRAG,
      transparent: true,
      depthWrite: false,
    });
    const sunDisc = new THREE.Mesh(new THREE.PlaneGeometry(46, 46), this.sunMat);
    sunDisc.position.set(0, 18.5, -142);
    this.scene.add(sunDisc);

    // Distant cyber skyline silhouette flanking the sun
    const towerMat = new THREE.MeshStandardMaterial({
      color: 0x0b0b1d,
      emissive: 0x1e1b4b,
      emissiveIntensity: 0.45,
      roughness: 0.5,
      metalness: 0.6,
    });
    const winMatCyan = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.25, 1.15, 1.6) });
    const winMatPink = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.4, 0.25, 1.05) });
    const skylineGroup = new THREE.Group();
    skylineGroup.position.set(0, 0, -132);
    const towerSpecs = [
      [-42, 24, 7], [-34, 34, 6], [-26, 20, 6], [-19, 28, 5], [-13, 15, 4.5],
      [13, 16, 4.5], [19, 29, 5], [27, 21, 6], [35, 36, 6], [43, 23, 7],
    ];
    for (let i = 0; i < towerSpecs.length; i += 1) {
      const [tx, th, tw] = towerSpecs[i];
      const box = new THREE.Mesh(new THREE.BoxGeometry(tw, th, 4), towerMat);
      box.position.set(tx, th * 0.5, 0);
      skylineGroup.add(box);
      const strip = new THREE.Mesh(
        new THREE.BoxGeometry(0.28, th * 0.82, 0.2),
        i % 2 === 0 ? winMatCyan : winMatPink,
      );
      strip.position.set(tx + (tx < 0 ? tw * 0.45 : -tw * 0.45), th * 0.5, 2.08);
      skylineGroup.add(strip);
    }
    this.scene.add(skylineGroup);

    // 3. Repeating Glowing Neon Cyberpunk Track Arches / Side Pylons
    this.buildTrackArches();

    // 4. Articulated 3D Cyber Runner Character
    this.buildRunnerRig();

    // 5. Object Pools: Obstacles (Low Barrier, High Gate, Neon Train), Gold Coins, Sparks
    this.buildObstaclePools();
    this.buildCoinPool();

    this.sLife.fill(99);
    this.sMax.fill(1);
    this.sparkGeo = new THREE.BufferGeometry();
    this.sparkGeo.setAttribute('position', new THREE.BufferAttribute(this.sPos, 3));
    this.sparkGeo.setAttribute('aColor', new THREE.BufferAttribute(this.sCol, 3));
    this.sparkGeo.setAttribute('aAlpha', new THREE.BufferAttribute(this.sAlpha, 1));
    this.sparkGeo.setAttribute('aSize', new THREE.BufferAttribute(this.sSize, 1));
    this.sparkMat = new THREE.ShaderMaterial({
      vertexShader: SPARK_VERT,
      fragmentShader: SPARK_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const sparkPoints = new THREE.Points(this.sparkGeo, this.sparkMat);
    sparkPoints.frustumCulled = false;
    this.scene.add(sparkPoints);
  }

  private buildTrackArches(): void {
    const pillarMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.7,
      roughness: 0.3,
    });
    const neonCyan = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.14, 0.92, 1.25) });
    const neonMagenta = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.08, 0.16, 0.86) });
    const pillarGeo = new THREE.BoxGeometry(0.36, 6.2, 0.44);
    const stripGeo = new THREE.BoxGeometry(0.08, 5.9, 0.10);
    const beamGeo = new THREE.BoxGeometry(9.8, 0.14, 0.18);

    for (let i = 0; i < ARCH_COUNT; i += 1) {
      const grp = new THREE.Group();
      const leftP = new THREE.Mesh(pillarGeo, pillarMat);
      leftP.position.set(-4.85, 3.1, 0);
      grp.add(leftP);

      const rightP = new THREE.Mesh(pillarGeo, pillarMat);
      rightP.position.set(4.85, 3.1, 0);
      grp.add(rightP);

      const neonMat = i % 2 === 0 ? neonCyan : neonMagenta;
      const leftS = new THREE.Mesh(stripGeo, neonMat);
      leftS.position.set(-4.62, 3.1, 0.18);
      grp.add(leftS);

      const rightS = new THREE.Mesh(stripGeo, neonMat);
      rightS.position.set(4.62, 3.1, 0.18);
      grp.add(rightS);

      const topBeam = new THREE.Mesh(beamGeo, neonMat);
      topBeam.position.set(0, 6.25, 0);
      grp.add(topBeam);

      grp.position.set(0, 0, 8 - i * ARCH_SPACING);
      this.scene.add(grp);
      this.arches.push(grp);
    }
  }

  private buildRunnerRig(): void {
    const armorMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.65,
      roughness: 0.28,
    });
    const trimCyan = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.18, 1.05, 1.38) });
    const trimWhite = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.12, 1.24, 1.38) });

    // Torso
    const chest = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.86, 0.42), armorMat);
    chest.position.set(0, 1.24, 0);
    this.torsoGroup.add(chest);

    const spineCore = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.62, 0.08), trimCyan);
    spineCore.position.set(0, 1.24, 0.22);
    this.torsoGroup.add(spineCore);

    const shoulderPad = new THREE.Mesh(new THREE.BoxGeometry(0.92, 0.14, 0.46), trimCyan);
    shoulderPad.position.set(0, 1.62, 0);
    this.torsoGroup.add(shoulderPad);

    // Head + Glowing Visor
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.46, 0.48), armorMat);
    head.position.set(0, 1.95, 0);
    this.torsoGroup.add(head);

    const visor = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.13, 0.12), trimWhite);
    visor.position.set(0, 1.98, -0.22);
    this.torsoGroup.add(visor);

    // Arms
    const armGeo = new THREE.BoxGeometry(0.2, 0.66, 0.2);
    const cuffGeo = new THREE.BoxGeometry(0.22, 0.1, 0.22);

    this.leftArm.position.set(-0.5, 1.56, 0);
    const laMesh = new THREE.Mesh(armGeo, armorMat);
    laMesh.position.set(0, -0.3, 0);
    const laCuff = new THREE.Mesh(cuffGeo, trimCyan);
    laCuff.position.set(0, -0.56, 0);
    this.leftArm.add(laMesh, laCuff);

    this.rightArm.position.set(0.5, 1.56, 0);
    const raMesh = new THREE.Mesh(armGeo, armorMat);
    raMesh.position.set(0, -0.3, 0);
    const raCuff = new THREE.Mesh(cuffGeo, trimCyan);
    raCuff.position.set(0, -0.56, 0);
    this.rightArm.add(raMesh, raCuff);

    this.torsoGroup.add(this.leftArm, this.rightArm);
    this.runnerRoot.add(this.torsoGroup);

    // Legs
    const legGeo = new THREE.BoxGeometry(0.24, 0.78, 0.24);
    const bootGeo = new THREE.BoxGeometry(0.26, 0.16, 0.36);

    this.leftLeg.position.set(-0.2, 0.82, 0);
    const llMesh = new THREE.Mesh(legGeo, armorMat);
    llMesh.position.set(0, -0.38, 0);
    const llBoot = new THREE.Mesh(bootGeo, trimCyan);
    llBoot.position.set(0, -0.74, -0.04);
    this.leftLeg.add(llMesh, llBoot);

    this.rightLeg.position.set(0.2, 0.82, 0);
    const rlMesh = new THREE.Mesh(legGeo, armorMat);
    rlMesh.position.set(0, -0.38, 0);
    const rlBoot = new THREE.Mesh(bootGeo, trimCyan);
    rlBoot.position.set(0, -0.74, -0.04);
    this.rightLeg.add(rlMesh, rlBoot);

    this.runnerRoot.add(this.leftLeg, this.rightLeg);
    this.scene.add(this.runnerRoot);
  }

  private buildObstaclePools(): void {
    const darkMetal = new THREE.MeshStandardMaterial({
      color: 0x111827,
      metalness: 0.72,
      roughness: 0.28,
    });
    const orangeGlow = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.75, 0.45, 0.14) });
    const pinkGlow = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.55, 0.22, 1.45) });
    const trainBodyMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.62,
      roughness: 0.3,
    });
    const trainCyan = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.22, 1.38, 1.85) });
    const headlightMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.9, 1.78, 1.05) });

    // 1. LOW_BARRIER pool (jump over or dodge)
    for (let i = 0; i < OBSTACLE_POOL_PER_KIND; i += 1) {
      const g = new THREE.Group();
      const postL = new THREE.Mesh(new THREE.BoxGeometry(0.22, 1.05, 0.28), darkMetal);
      postL.position.set(-1.02, 0.52, 0);
      const postR = new THREE.Mesh(new THREE.BoxGeometry(0.22, 1.05, 0.28), darkMetal);
      postR.position.set(1.02, 0.52, 0);
      const barTop = new THREE.Mesh(new THREE.BoxGeometry(2.26, 0.28, 0.26), orangeGlow);
      barTop.position.set(0, 0.88, 0);
      const barMid = new THREE.Mesh(new THREE.BoxGeometry(2.05, 0.14, 0.16), orangeGlow);
      barMid.position.set(0, 0.44, 0);
      g.add(postL, postR, barTop, barMid);
      g.visible = false;
      this.scene.add(g);
      this.obstacles.push({ active: false, kind: 'LOW_BARRIER', lane: 0, z: -200, halfLen: 0.55, mesh: g });
    }

    // 2. HIGH_GATE pool (crouch/slide under or dodge)
    for (let i = 0; i < OBSTACLE_POOL_PER_KIND; i += 1) {
      const g = new THREE.Group();
      const postL = new THREE.Mesh(new THREE.BoxGeometry(0.2, 2.9, 0.26), darkMetal);
      postL.position.set(-1.12, 1.45, 0);
      const postR = new THREE.Mesh(new THREE.BoxGeometry(0.2, 2.9, 0.26), darkMetal);
      postR.position.set(1.12, 1.45, 0);
      const header = new THREE.Mesh(new THREE.BoxGeometry(2.42, 0.72, 0.4), darkMetal);
      header.position.set(0, 2.48, 0);
      const laserMain = new THREE.Mesh(new THREE.BoxGeometry(2.26, 0.46, 0.22), pinkGlow);
      laserMain.position.set(0, 1.68, 0);
      const laserEdge = new THREE.Mesh(new THREE.BoxGeometry(2.22, 0.14, 0.26), pinkGlow);
      laserEdge.position.set(0, 1.24, 0);
      g.add(postL, postR, header, laserMain, laserEdge);
      g.visible = false;
      this.scene.add(g);
      this.obstacles.push({ active: false, kind: 'HIGH_GATE', lane: 0, z: -200, halfLen: 0.6, mesh: g });
    }

    // 3. TRAIN pool (full cyber metro car — must switch lane)
    for (let i = 0; i < OBSTACLE_POOL_PER_KIND; i += 1) {
      const g = new THREE.Group();
      const body = new THREE.Mesh(new THREE.BoxGeometry(2.18, 2.75, 11.4), trainBodyMat);
      body.position.set(0, 1.48, 0);
      const roof = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.22, 11.1), trainCyan);
      roof.position.set(0, 2.9, 0);
      const stripeL = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.24, 11.2), trainCyan);
      stripeL.position.set(-1.1, 1.55, 0);
      const stripeR = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.24, 11.2), trainCyan);
      stripeR.position.set(1.1, 1.55, 0);
      const windshield = new THREE.Mesh(new THREE.BoxGeometry(1.78, 0.85, 0.12), trainCyan);
      windshield.position.set(0, 2.05, 5.7);
      const lightL = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.24, 0.14), headlightMat);
      lightL.position.set(-0.68, 0.82, 5.72);
      const lightR = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.24, 0.14), headlightMat);
      lightR.position.set(0.68, 0.82, 5.72);
      g.add(body, roof, stripeL, stripeR, windshield, lightL, lightR);
      g.visible = false;
      this.scene.add(g);
      this.obstacles.push({ active: false, kind: 'TRAIN', lane: 0, z: -200, halfLen: 5.75, mesh: g });
    }
  }

  private buildCoinPool(): void {
    const coinGeo = new THREE.OctahedronGeometry(0.36, 0);
    const coinMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(1.75, 1.38, 0.28),
    });
    for (let i = 0; i < COIN_POOL_SIZE; i += 1) {
      const mesh = new THREE.Mesh(coinGeo, coinMat);
      mesh.visible = false;
      this.scene.add(mesh);
      this.coinPool.push({ active: false, lane: 0, x: 0, y: 1.0, z: -200, mesh });
    }
  }

  public setSfxEnabled(enabled: boolean): void {
    this.audio.setEnabled(enabled);
  }

  public resetToReady(): void {
    this.phase = 'READY';
    this.clearTrackEntities();
    this.score = 0;
    this.coins = 0;
    this.distance = 0;
    this.speed = BASE_SPEED;
    this.lane = 0;
    this.playerX = 0;
    this.playerY = 0;
    this.playerVy = 0;
    this.isGrounded = true;
    this.slideTimer = 0;
    this.isSliding = false;
    this.camShake = 0;
    this.actionLabel = 'READY';
  }

  public startRun(
    onBeginCalibration?: () => void,
    skipCountdown = false,
  ): void {
    this.audio.prewarm();
    this.clearTrackEntities();
    this.score = 0;
    this.coins = 0;
    this.distance = 0;
    this.speed = BASE_SPEED;
    this.lane = 0;
    this.playerX = 0;
    this.playerY = 0;
    this.playerVy = 0;
    this.isGrounded = true;
    this.slideTimer = 0;
    this.isSliding = false;
    this.camShake = 0;
    this.nextWaveDist = 26;
    this.lastSafeLane = 0;

    if (skipCountdown) {
      this.phase = 'PLAYING';
      this.actionLabel = 'RUN';
      this.audio.playGo();
      return;
    }

    this.phase = 'CALIBRATING';
    this.calibTimer = 2.15;
    this.lastCountdownBeep = 4;
    this.actionLabel = 'CALIBRATING';
    onBeginCalibration?.();
  }

  private clearTrackEntities(): void {
    for (let i = 0; i < this.obstacles.length; i += 1) {
      this.obstacles[i].active = false;
      this.obstacles[i].mesh.visible = false;
    }
    for (let i = 0; i < this.coinPool.length; i += 1) {
      this.coinPool[i].active = false;
      this.coinPool[i].mesh.visible = false;
    }
  }

  private acquireObstacle(kind: ObstacleKind): PooledObstacle | null {
    for (let i = 0; i < this.obstacles.length; i += 1) {
      const o = this.obstacles[i];
      if (!o.active && o.kind === kind) return o;
    }
    return null;
  }

  private spawnCoin(lane: RunnerLane, y: number, z: number): void {
    for (let i = 0; i < this.coinPool.length; i += 1) {
      const c = this.coinPool[i];
      if (!c.active) {
        c.active = true;
        c.lane = lane;
        c.x = lane * LANE_WIDTH;
        c.y = y;
        c.z = z;
        c.mesh.position.set(c.x, c.y, c.z);
        c.mesh.visible = true;
        return;
      }
    }
  }

  private spawnSparks(x: number, y: number, z: number, r: number, g: number, b: number, count: number, spread = 4.2): void {
    for (let k = 0; k < count; k += 1) {
      const i = this.sparkCursor;
      this.sparkCursor = (this.sparkCursor + 1) % SPARK_POOL;
      const a = Math.random() * Math.PI * 2;
      const spd = (0.35 + Math.random() * 0.85) * spread;
      this.sPos[i * 3] = x + (Math.random() - 0.5) * 0.4;
      this.sPos[i * 3 + 1] = y + (Math.random() - 0.5) * 0.35;
      this.sPos[i * 3 + 2] = z + (Math.random() - 0.5) * 0.4;
      this.sCol[i * 3] = r;
      this.sCol[i * 3 + 1] = g;
      this.sCol[i * 3 + 2] = b;
      this.svx[i] = Math.cos(a) * spd;
      this.svy[i] = 1.6 + Math.random() * spread * 0.9;
      this.svz[i] = Math.sin(a) * spd;
      this.sLife[i] = 0;
      this.sMax[i] = 0.25 + Math.random() * 0.35;
      this.sSize[i] = 4.0 + Math.random() * 6.0;
    }
  }

  /** Guaranteed-solvable Subway Surfers obstacle & coin wave generator. */
  private spawnNextWave(): void {
    const spawnZ = -112;
    const lanes: RunnerLane[] = [-1, 0, 1];
    // Pick a guaranteed open/solvable primary lane within 1 step of previous safe lane
    const candidateSafe = lanes.filter((l) => Math.abs(l - this.lastSafeLane) <= 1);
    const safeLane = candidateSafe[Math.floor(Math.random() * candidateSafe.length)];
    this.lastSafeLane = safeLane;

    const pattern = Math.random();
    if (pattern < 0.34) {
      // Pattern A: 1 or 2 Trains + safe open lane (with coin trail) or jumpable barrier
      for (const l of lanes) {
        if (l === safeLane) {
          for (let c = 0; c < 5; c += 1) {
            this.spawnCoin(l, 1.05, spawnZ + 5 - c * 2.5);
          }
        } else if (Math.random() < 0.78) {
          const train = this.acquireObstacle('TRAIN');
          if (train) {
            train.active = true;
            train.lane = l;
            train.z = spawnZ;
            train.mesh.position.set(l * LANE_WIDTH, 0, spawnZ);
            train.mesh.visible = true;
          }
        }
      }
    } else if (pattern < 0.68) {
      // Pattern B: Low Barrier in safe lane (with jump arc of 5 gold coins!) + flank trains/gates
      const barrier = this.acquireObstacle('LOW_BARRIER');
      if (barrier) {
        barrier.active = true;
        barrier.lane = safeLane;
        barrier.z = spawnZ;
        barrier.mesh.position.set(safeLane * LANE_WIDTH, 0, spawnZ);
        barrier.mesh.visible = true;
        // Parabolic coin arc rewarding jump
        for (let c = -2; c <= 2; c += 1) {
          const arcY = 2.65 - (c * c) * 0.32;
          this.spawnCoin(safeLane, arcY, spawnZ + c * 2.1);
        }
      }
      for (const l of lanes) {
        if (l === safeLane) continue;
        const kind: ObstacleKind = Math.random() < 0.55 ? 'TRAIN' : 'HIGH_GATE';
        const ob = this.acquireObstacle(kind);
        if (ob) {
          ob.active = true;
          ob.lane = l;
          ob.z = spawnZ;
          ob.mesh.position.set(l * LANE_WIDTH, 0, spawnZ);
          ob.mesh.visible = true;
        }
      }
    } else {
      // Pattern C: High Overhead Laser Gate in safe lane (low coin line rewarding crouch!) + side obstacles
      const gate = this.acquireObstacle('HIGH_GATE');
      if (gate) {
        gate.active = true;
        gate.lane = safeLane;
        gate.z = spawnZ;
        gate.mesh.position.set(safeLane * LANE_WIDTH, 0, spawnZ);
        gate.mesh.visible = true;
        for (let c = -1; c <= 2; c += 1) {
          this.spawnCoin(safeLane, 0.55, spawnZ + c * 2.1);
        }
      }
      for (const l of lanes) {
        if (l === safeLane) continue;
        if (Math.random() < 0.8) {
          const kind: ObstacleKind = Math.random() < 0.5 ? 'LOW_BARRIER' : 'TRAIN';
          const ob = this.acquireObstacle(kind);
          if (ob) {
            ob.active = true;
            ob.lane = l;
            ob.z = spawnZ;
            ob.mesh.position.set(l * LANE_WIDTH, 0, spawnZ);
            ob.mesh.visible = true;
          }
        }
      }
    }

    const speedRatio = (this.speed - BASE_SPEED) / (MAX_SPEED - BASE_SPEED);
    const gap = 30 - speedRatio * 7.5 + Math.random() * 5.5;
    this.nextWaveDist = this.distance + Math.max(20.5, gap);
  }

  public getHudSnapshot(): RunnerHudSnapshot {
    const countdownNum =
      this.phase === 'CALIBRATING' ? Math.max(1, Math.min(3, Math.ceil(this.calibTimer / 0.7))) : 0;
    return {
      phase: this.phase,
      score: Math.floor(this.score),
      coins: this.coins,
      bestScore: Math.floor(this.bestScore),
      speedMultiplier: Math.round((this.speed / BASE_SPEED) * 10) / 10,
      countdownNum,
      lane: this.lane,
      actionLabel: this.actionLabel,
      hasBody: this.latestBodyHas,
      isCalibrated: this.latestBodyCalibrated,
      handsUpProgress: this.latestHandsUp,
      distance: Math.floor(this.distance),
    };
  }

  public update(
    body: BodyControlState,
    dt: number,
    viewW: number,
    viewH: number,
    overlayCtx: CanvasRenderingContext2D | null,
    onFinishCalibration?: () => void,
    onBeginCalibration?: () => void,
  ): void {
    const safeDt = Math.min(Math.max(dt, 0.002), 0.05);
    this.elapsed += safeDt;
    this.latestBodyHas = body.hasBody;
    this.latestBodyCalibrated = body.isCalibrated;
    this.latestHandsUp = body.handsUpProgress;

    // Update camera aspect
    const aspect = Math.max(0.5, viewW / Math.max(viewH, 1));
    if (Math.abs(this.camera.aspect - aspect) > 0.002) {
      this.camera.aspect = aspect;
    }

    // Touchless both-hands-up (1.2s) starts or restarts automatically
    if ((this.phase === 'READY' || this.phase === 'GAME_OVER') && body.handsUpTriggered) {
      this.startRun(onBeginCalibration, false);
    }

    // Calibration countdown phase
    if (this.phase === 'CALIBRATING') {
      this.calibTimer -= safeDt;
      const num = Math.max(1, Math.min(3, Math.ceil(this.calibTimer / 0.7)));
      if (num < this.lastCountdownBeep) {
        this.lastCountdownBeep = num;
        this.audio.playCountdownTick(num);
      }
      if (this.calibTimer <= 0) {
        onFinishCalibration?.();
        this.phase = 'PLAYING';
        this.actionLabel = 'RUN';
        this.audio.playGo();
      }
    }

    // Active scroll speed (gentle scenic cruise when READY/CALIBRATING)
    const activeSpeed =
      this.phase === 'PLAYING'
        ? this.speed
        : this.phase === 'GAME_OVER'
          ? 0
          : BASE_SPEED * 0.42;

    if (this.phase === 'PLAYING') {
      this.speed = Math.min(MAX_SPEED, BASE_SPEED + this.distance * 0.022);
      const dz = this.speed * safeDt;
      this.distance += dz;
      this.score += dz * 1.45;
      if (this.score > this.bestScore) {
        this.bestScore = this.score;
      }

      // 1. Lane switch
      if (body.hasBody && body.lane !== this.lane) {
        this.lane = body.lane;
        this.audio.playLaneSwitch();
      }

      // 2. Jump (allowed even while changing lanes!)
      if (body.jumpTriggered && this.isGrounded) {
        this.playerVy = JUMP_VY;
        this.isGrounded = false;
        this.isSliding = false;
        this.slideTimer = 0;
        this.jumpFlipAngle = 0;
        this.actionLabel = 'JUMP';
        this.audio.playJump();
        this.spawnSparks(this.playerX, 0.2, 0.2, 0.3, 1.5, 2.0, 10, 3.4);
      }

      // 3. Crouch / Slide OR mid-air Fast Drop
      if (body.crouchTriggered) {
        if (!this.isGrounded) {
          this.playerVy = FAST_DROP_VY;
          this.slideTimer = MIN_SLIDE_SEC;
          this.isSliding = true;
          this.actionLabel = 'FAST DROP';
          this.audio.playCrouch();
        } else {
          this.slideTimer = MIN_SLIDE_SEC;
          this.isSliding = true;
          this.actionLabel = 'SLIDE';
          this.audio.playCrouch();
          this.spawnSparks(this.playerX, 0.15, 0.4, 1.5, 0.3, 1.4, 8, 2.6);
        }
      }

      if (this.isSliding) {
        if (this.slideTimer > 0) {
          this.slideTimer = Math.max(0, this.slideTimer - safeDt);
        }
        if (this.slideTimer <= 0 && !body.isCrouching) {
          this.isSliding = false;
        } else if (this.isGrounded && Math.random() < safeDt * 24) {
          this.spawnSparks(this.playerX, 0.12, 0.45, 0.28, 1.35, 1.9, 1, 1.6);
        }
      }

      // Vertical physics
      if (!this.isGrounded) {
        this.playerVy -= GRAVITY * safeDt;
        this.playerY += this.playerVy * safeDt;
        this.jumpFlipAngle += safeDt * 7.8;
        if (this.playerY <= 0) {
          this.playerY = 0;
          this.playerVy = 0;
          this.isGrounded = true;
          this.jumpFlipAngle = 0;
        }
      }

      if (this.isGrounded && !this.isSliding) {
        this.actionLabel = this.lane === -1 ? 'LEFT' : this.lane === 1 ? 'RIGHT' : 'RUN';
      }

      // Spawn waves & advance obstacles
      if (this.distance >= this.nextWaveDist) {
        this.spawnNextWave();
      }

      const playerBottom = this.playerY;
      const playerTop = this.playerY + (this.isSliding ? 0.84 : 1.92);

      for (let i = 0; i < this.obstacles.length; i += 1) {
        const ob = this.obstacles[i];
        if (!ob.active) continue;
        ob.z += dz;
        ob.mesh.position.z = ob.z;
        if (ob.z > 16) {
          ob.active = false;
          ob.mesh.visible = false;
          continue;
        }

        // Fair AABB collision check around runner z = 0
        const obX = ob.lane * LANE_WIDTH;
        const hitX = Math.abs(this.playerX - obX) < 0.92;
        const hitZ = Math.abs(ob.z) < ob.halfLen + 0.34;
        if (hitX && hitZ) {
          let hitY = true;
          if (ob.kind === 'LOW_BARRIER') {
            hitY = playerBottom < 0.88;
          } else if (ob.kind === 'HIGH_GATE') {
            hitY = playerTop > 1.06;
          } else {
            hitY = playerBottom < 2.68;
          }
          if (hitY) {
            this.phase = 'GAME_OVER';
            this.actionLabel = 'CRASH';
            this.camShake = 0.55;
            saveBestScore(this.bestScore);
            this.audio.playCrash();
            this.spawnSparks(this.playerX, this.playerY + 1.1, 0, 1.85, 0.45, 0.2, 42, 7.2);
            this.spawnSparks(this.playerX, this.playerY + 1.1, 0, 0.25, 1.5, 2.0, 26, 6.4);
            break;
          }
        }
      }

      // Advance & collect spinning Gold Coins
      for (let i = 0; i < this.coinPool.length; i += 1) {
        const c = this.coinPool[i];
        if (!c.active) continue;
        c.z += dz;
        c.mesh.position.z = c.z;
        c.mesh.rotation.y = this.elapsed * 4.2 + i * 0.4;
        c.mesh.position.y = c.y + Math.sin(this.elapsed * 5.0 + i) * 0.1;

        if (c.z > 12) {
          c.active = false;
          c.mesh.visible = false;
          continue;
        }

        if (
          Math.abs(this.playerX - c.x) < 1.12 &&
          Math.abs(c.z) < 1.15 &&
          c.y >= playerBottom - 0.4 &&
          c.y <= playerTop + 0.55
        ) {
          c.active = false;
          c.mesh.visible = false;
          this.coins += 1;
          this.score += 35;
          if (this.score > this.bestScore) this.bestScore = this.score;
          this.audio.playCoin();
          this.spawnSparks(c.x, c.y, 0, 1.85, 1.48, 0.35, 10, 3.8);
        }
      }
    } else if (body.hasBody && this.phase === 'READY') {
      // Preview lane alignment while waiting to start
      this.lane = body.lane;
    }

    // Smooth lateral runner interpolation + banking roll
    const targetX = this.lane * LANE_WIDTH;
    const prevX = this.playerX;
    this.playerX += (targetX - this.playerX) * (1 - Math.exp(-16.5 * safeDt));
    const latVel = (this.playerX - prevX) / safeDt;

    this.runnerRoot.position.set(this.playerX, this.playerY, 0);
    const targetBank = Math.max(-0.34, Math.min(0.34, -latVel * 0.022));
    this.runnerRoot.rotation.z += (targetBank - this.runnerRoot.rotation.z) * (1 - Math.exp(-14 * safeDt));

    // Articulated limbs + crouch slide posture + jump tuck flip
    this.runCycle += safeDt * (activeSpeed * 0.48);
    if (this.isSliding && this.isGrounded) {
      this.runnerRoot.scale.set(1.06, 0.48, 1.14);
      this.torsoGroup.rotation.x = 0.42;
      this.leftLeg.rotation.x = -0.9;
      this.rightLeg.rotation.x = 0.65;
      this.leftArm.rotation.x = 0.7;
      this.rightArm.rotation.x = -0.5;
    } else {
      this.runnerRoot.scale.lerp(new THREE.Vector3(1, 1, 1), 1 - Math.exp(-18 * safeDt));
      if (!this.isGrounded) {
        this.torsoGroup.rotation.x = Math.min(Math.PI * 2, this.jumpFlipAngle);
        this.leftLeg.rotation.x = -0.65;
        this.rightLeg.rotation.x = 0.45;
        this.leftArm.rotation.x = -1.1;
        this.rightArm.rotation.x = -1.1;
      } else {
        this.torsoGroup.rotation.x = 0.14;
        const swing = Math.sin(this.runCycle) * 0.92;
        this.leftLeg.rotation.x = swing;
        this.rightLeg.rotation.x = -swing;
        this.leftArm.rotation.x = -swing * 0.85;
        this.rightArm.rotation.x = swing * 0.85;
      }
    }

    // Scroll neon track shader & repeating arches
    this.scrollZ += activeSpeed * safeDt;
    const speedNorm = Math.max(0, Math.min(1, (this.speed - BASE_SPEED) / (MAX_SPEED - BASE_SPEED)));
    this.roadMat.uniforms.uScrollZ.value = this.scrollZ;
    this.roadMat.uniforms.uSpeedNorm.value = this.phase === 'PLAYING' ? speedNorm : 0;
    this.sunMat.uniforms.uTime.value = this.elapsed;

    // Update dynamic glossy neon road reflections (slot 0 = runner, slots 1..3 = closest obstacles)
    this.reflLights[0].set(this.playerX, 0.2, 0, 1.0);
    let rSlot = 1;
    for (let i = 0; i < this.obstacles.length && rSlot < 4; i += 1) {
      const ob = this.obstacles[i];
      if (!ob.active || ob.z < -68 || ob.z > 8) continue;
      const colId = ob.kind === 'TRAIN' ? 0 : ob.kind === 'HIGH_GATE' ? 1 : 2;
      this.reflLights[rSlot].set(ob.lane * LANE_WIDTH, ob.z + (ob.kind === 'TRAIN' ? 4.5 : 0), colId, 1.15);
      rSlot += 1;
    }
    for (; rSlot < 4; rSlot += 1) {
      this.reflLights[rSlot].w = 0;
    }

    for (let i = 0; i < this.arches.length; i += 1) {
      const arch = this.arches[i];
      arch.position.z += activeSpeed * safeDt;
      if (arch.position.z > 14) {
        arch.position.z -= ARCH_COUNT * ARCH_SPACING;
      }
    }

    // Dynamic FOV (60° -> 73°) + Smooth Camera Banking & Impact Shake
    const targetFov = 60 + (this.phase === 'PLAYING' ? speedNorm * 13 : 0);
    this.camera.fov += (targetFov - this.camera.fov) * (1 - Math.exp(-4.5 * safeDt));
    this.camera.updateProjectionMatrix();

    const targetCamBank = Math.max(-0.085, Math.min(0.085, -latVel * 0.0055));
    this.camBankZ += (targetCamBank - this.camBankZ) * (1 - Math.exp(-8.5 * safeDt));

    let shakeX = 0;
    let shakeY = 0;
    if (this.camShake > 0.005) {
      shakeX = (Math.random() - 0.5) * this.camShake * 0.9;
      shakeY = (Math.random() - 0.5) * this.camShake * 0.7;
      this.camShake = Math.max(0, this.camShake - safeDt * 1.45);
    }

    this.camera.position.set(
      this.playerX * 0.42 + shakeX,
      4.25 + this.playerY * 0.24 + shakeY,
      8.55,
    );
    this.camera.lookAt(this.playerX * 0.25, 1.55 + this.playerY * 0.15, -18);
    this.camera.rotation.z += this.camBankZ;

    // Update 3D burst sparks
    let anySpark = false;
    for (let i = 0; i < SPARK_POOL; i += 1) {
      if (this.sLife[i] >= this.sMax[i]) {
        this.sAlpha[i] = 0;
        continue;
      }
      this.sLife[i] += safeDt;
      if (this.sLife[i] >= this.sMax[i]) {
        this.sAlpha[i] = 0;
        continue;
      }
      anySpark = true;
      this.svy[i] -= 12 * safeDt;
      this.sPos[i * 3] += this.svx[i] * safeDt;
      this.sPos[i * 3 + 1] = Math.max(0.05, this.sPos[i * 3 + 1] + this.svy[i] * safeDt);
      this.sPos[i * 3 + 2] += (this.svz[i] + activeSpeed * 0.35) * safeDt;
      this.sAlpha[i] = 1 - this.sLife[i] / this.sMax[i];
    }
    if (anySpark) {
      (this.sparkGeo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
      (this.sparkGeo.attributes.aColor as THREE.BufferAttribute).needsUpdate = true;
      (this.sparkGeo.attributes.aAlpha as THREE.BufferAttribute).needsUpdate = true;
      (this.sparkGeo.attributes.aSize as THREE.BufferAttribute).needsUpdate = true;
    }

    // Edge Speed Lines on 2D Overlay when running fast
    if (overlayCtx && this.phase === 'PLAYING' && speedNorm > 0.08) {
      this.drawSpeedLines(overlayCtx, viewW, viewH, speedNorm);
    }
  }

  private drawSpeedLines(
    ctx: CanvasRenderingContext2D,
    viewW: number,
    viewH: number,
    speedNorm: number,
  ): void {
    const intensity = Math.min(1, speedNorm * 1.15);
    const cx = viewW * 0.5;
    const cy = viewH * 0.44;
    const maxR = Math.hypot(cx, cy);
    const lineCount = Math.floor(14 + intensity * 18);

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';

    for (let i = 0; i < lineCount; i += 1) {
      const angle = (i / lineCount) * Math.PI * 2 + Math.sin(i * 19.3) * 0.14;
      const phase = (this.elapsed * (3.2 + intensity * 3.8) + i * 0.37) % 1;
      const r0 = maxR * (0.58 + phase * 0.24);
      const r1 = r0 + maxR * (0.14 + intensity * 0.18);
      const alpha = Math.sin(phase * Math.PI) * intensity * 0.34;
      if (alpha <= 0.01) continue;

      ctx.strokeStyle =
        i % 3 === 0
          ? `rgba(244, 114, 182, ${alpha})`
          : `rgba(56, 189, 248, ${alpha})`;
      ctx.lineWidth = 1.2 + intensity * 1.4;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(angle) * r0, cy + Math.sin(angle) * r0);
      ctx.lineTo(cx + Math.cos(angle) * r1, cy + Math.sin(angle) * r1);
      ctx.stroke();
    }
    ctx.restore();
  }

  public dispose(): void {
    this.audio.dispose();
    this.scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Points) {
        obj.geometry.dispose();
        const mat = obj.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat.dispose();
      }
    });
  }
}
