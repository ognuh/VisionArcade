import type { Landmark3D } from '../types/vision';
import { OneEuroAxis } from './smoothing';

export type RunnerLane = -1 | 0 | 1;

export interface BodyControlState {
  hasBody: boolean;
  isCalibrated: boolean;
  lane: RunnerLane;
  laneChanged: boolean;
  jumpTriggered: boolean;
  crouchTriggered: boolean;
  isJumpZone: boolean;
  isCrouching: boolean;
  /** Touchless start/restart hold progress (0..1 over 1.2s with both hands overhead). */
  handsUpProgress: number;
  handsUpTriggered: boolean;
  /** Filtered normalized screen-space shoulder midpoint (already mirrored if mirrored=true). */
  midX: number;
  midY: number;
  shoulderWidth: number;
  /** Calibrated standing reference posture in normalized screen space. */
  baseX: number;
  baseY: number;
  baseShoulderWidth: number;
  /** Relative horizontal offset in shoulder-width units. */
  dxNorm: number;
  /** Positive = jumping up, negative = crouching down (in shoulder-width units). */
  dyUpNorm: number;
  /** Latest mirrored upper-body landmarks for PiP skeleton rendering. */
  pose: Landmark3D[] | null;
}

const LANE_ENTER_THRESHOLD = 0.42;
const LANE_RETURN_THRESHOLD = 0.26;
const JUMP_HEIGHT_RATIO = 0.3;
const JUMP_VEL_RATIO = 2.35;
const JUMP_COOLDOWN_SEC = 0.35;
const CROUCH_ENTER_RATIO = 0.32;
const CROUCH_EXIT_RATIO = 0.24;
const HANDS_UP_HOLD_SEC = 1.2;

const BODY_MIN_CUTOFF = 2.4;
const BODY_BETA = 0.09;

function visOf(lm: Landmark3D | undefined): number {
  if (!lm) return 0;
  return lm.visibility ?? 1;
}

/**
 * Full-Body Gesture Decoder for VisionArcade Neon Subway Runner:
 * - Tracks shoulder midpoint (landmarks 11 & 12) normalized by shoulder width,
 *   so it works reliably in small rooms with only chest/upper-torso visible.
 * - One-Euro filtered (< 40ms latency) with hysteresis 3-lane switching,
 *   Jump (height + upward velocity spike, 350ms cooldown), Crouch/Fast-Drop,
 *   and 1.2s touchless both-hands-overhead start/restart trigger.
 */
export class BodyController {
  private filterX = new OneEuroAxis();
  private filterY = new OneEuroAxis();
  private filterW = new OneEuroAxis();

  private hasBody = false;
  private isCalibrated = false;
  private lane: RunnerLane = 0;

  private midX = 0.5;
  private midY = 0.54;
  private prevMidY = 0.54;
  private shoulderW = 0.22;

  private baseX = 0.5;
  private baseY = 0.54;
  private baseShoulderW = 0.22;

  private jumpCooldown = 0;
  private jumpArmed = true;
  private isCrouching = false;

  private handsUpHoldSec = 0;
  private handsUpConsumed = false;

  // Calibration averaging accumulator
  private collectingCalib = false;
  private calibSumX = 0;
  private calibSumY = 0;
  private calibSumW = 0;
  private calibSamples = 0;

  private missedBodySec = 0;
  private latestPose: Landmark3D[] | null = null;

  public reset(): void {
    this.filterX.reset();
    this.filterY.reset();
    this.filterW.reset();
    this.hasBody = false;
    this.lane = 0;
    this.jumpCooldown = 0;
    this.jumpArmed = true;
    this.isCrouching = false;
    this.handsUpHoldSec = 0;
    this.handsUpConsumed = false;
    this.missedBodySec = 0;
    this.latestPose = null;
    this.collectingCalib = false;
    this.calibSamples = 0;
  }

  public beginCalibrationSample(): void {
    this.collectingCalib = true;
    this.calibSumX = 0;
    this.calibSumY = 0;
    this.calibSumW = 0;
    this.calibSamples = 0;
    this.lane = 0;
    this.isCrouching = false;
    this.jumpCooldown = 0;
    this.jumpArmed = true;
  }

  public finishCalibrationSample(): void {
    if (this.calibSamples >= 3) {
      this.baseX = this.calibSumX / this.calibSamples;
      this.baseY = this.calibSumY / this.calibSamples;
      this.baseShoulderW = Math.max(0.11, Math.min(0.45, this.calibSumW / this.calibSamples));
      this.isCalibrated = true;
    } else if (this.hasBody) {
      this.baseX = this.midX;
      this.baseY = this.midY;
      this.baseShoulderW = Math.max(0.11, Math.min(0.45, this.shoulderW));
      this.isCalibrated = true;
    }
    this.collectingCalib = false;
    this.lane = 0;
    this.isCrouching = false;
  }

  public recalibrateNow(): void {
    if (this.hasBody) {
      this.baseX = this.midX;
      this.baseY = this.midY;
      this.baseShoulderW = Math.max(0.11, Math.min(0.45, this.shoulderW));
      this.isCalibrated = true;
    } else {
      this.baseX = 0.5;
      this.baseY = 0.54;
      this.baseShoulderW = 0.22;
      this.isCalibrated = false;
    }
    this.lane = 0;
    this.isCrouching = false;
    this.jumpCooldown = 0;
    this.jumpArmed = true;
  }

  public update(
    rawPose: Landmark3D[] | null,
    dt: number,
    mirrored: boolean,
    allowAutoIdleBaseline: boolean,
  ): BodyControlState {
    const safeDt = Math.min(Math.max(dt, 0.004), 0.08);
    if (this.jumpCooldown > 0) {
      this.jumpCooldown = Math.max(0, this.jumpCooldown - safeDt);
    }

    const ls = rawPose && rawPose.length > 12 ? rawPose[11] : undefined;
    const rs = rawPose && rawPose.length > 12 ? rawPose[12] : undefined;
    const validShoulders = !!ls && !!rs && visOf(ls) >= 0.35 && visOf(rs) >= 0.35;

    let laneChanged = false;
    let jumpTriggered = false;
    let crouchTriggered = false;
    let isJumpZone = false;
    let handsUpTriggered = false;

    if (validShoulders && rawPose) {
      this.missedBodySec = 0;
      this.latestPose = rawPose;

      const lsx = mirrored ? 1 - ls.x : ls.x;
      const rsx = mirrored ? 1 - rs.x : rs.x;
      const rawMidX = (lsx + rsx) * 0.5;
      const rawMidY = (ls.y + rs.y) * 0.5;
      const rawShoulderW = Math.max(0.09, Math.hypot(rsx - lsx, rs.y - ls.y));

      const wasTracking = this.hasBody;
      this.prevMidY = wasTracking ? this.midY : rawMidY;

      this.midX = this.filterX.filter(rawMidX, safeDt, BODY_MIN_CUTOFF, BODY_BETA);
      this.midY = this.filterY.filter(rawMidY, safeDt, BODY_MIN_CUTOFF, BODY_BETA);
      this.shoulderW = this.filterW.filter(rawShoulderW, safeDt, 1.4, 0.02);
      this.hasBody = true;

      if (!wasTracking && !this.isCalibrated) {
        this.baseX = this.midX;
        this.baseY = this.midY;
        this.baseShoulderW = Math.max(0.11, Math.min(0.45, this.shoulderW));
      } else if (allowAutoIdleBaseline && !this.isCalibrated) {
        const blend = 1 - Math.exp(-2.2 * safeDt);
        this.baseX += (this.midX - this.baseX) * blend;
        this.baseY += (this.midY - this.baseY) * blend;
        this.baseShoulderW += (Math.max(0.11, Math.min(0.45, this.shoulderW)) - this.baseShoulderW) * blend;
      }

      if (this.collectingCalib) {
        this.calibSumX += this.midX;
        this.calibSumY += this.midY;
        this.calibSumW += this.shoulderW;
        this.calibSamples += 1;
        this.baseX = this.calibSumX / this.calibSamples;
        this.baseY = this.calibSumY / this.calibSamples;
        this.baseShoulderW = Math.max(0.11, Math.min(0.45, this.calibSumW / this.calibSamples));
      }

      const sw = Math.max(0.1, this.baseShoulderW);
      const dxNorm = (this.midX - this.baseX) / sw;
      const dyUpNorm = (this.baseY - this.midY) / sw; // positive = jumped upward
      const dyDownNorm = -dyUpNorm; // positive = crouched downward
      const upVelNorm = (this.prevMidY - this.midY) / (safeDt * sw);

      // 1. 3-Lane Hysteresis (Left: -1, Center: 0, Right: +1)
      const prevLane = this.lane;
      if (dxNorm < -LANE_ENTER_THRESHOLD) {
        this.lane = -1;
      } else if (dxNorm > LANE_ENTER_THRESHOLD) {
        this.lane = 1;
      } else if (Math.abs(dxNorm) <= LANE_RETURN_THRESHOLD) {
        this.lane = 0;
      }
      laneChanged = this.lane !== prevLane;

      // 2. Jump Detection (height threshold OR explosive upward velocity spike, 350ms cooldown)
      isJumpZone = dyUpNorm > JUMP_HEIGHT_RATIO;
      if (!this.jumpArmed && dyUpNorm < JUMP_HEIGHT_RATIO * 0.55) {
        this.jumpArmed = true;
      }
      const wantsJump =
        isJumpZone || (upVelNorm > JUMP_VEL_RATIO && dyUpNorm > JUMP_HEIGHT_RATIO * 0.45);
      if (wantsJump && this.jumpArmed && this.jumpCooldown <= 0) {
        jumpTriggered = true;
        this.jumpCooldown = JUMP_COOLDOWN_SEC;
        this.jumpArmed = false;
      }

      // 3. Crouch / Roll Detection (with release hysteresis)
      const wasCrouching = this.isCrouching;
      if (!this.isCrouching && dyDownNorm > CROUCH_ENTER_RATIO) {
        this.isCrouching = true;
        crouchTriggered = true;
      } else if (this.isCrouching && dyDownNorm < CROUCH_EXIT_RATIO) {
        this.isCrouching = false;
      }
      if (wasCrouching && !this.isCrouching) {
        // clean release
      }

      // 4. Touchless Overhead Both-Hands-Up Detector (hold 1.2s to Start / Play Again)
      const lw = rawPose[15];
      const rw = rawPose[16];
      const nose = rawPose[0];
      const wristsVisible = visOf(lw) >= 0.32 && visOf(rw) >= 0.32;
      const noseGateY = nose && visOf(nose) >= 0.3 ? nose.y + 0.08 * sw : this.midY - 0.28 * sw;
      const bothHandsUp =
        wristsVisible &&
        lw.y < this.midY - 0.32 * sw &&
        rw.y < this.midY - 0.32 * sw &&
        lw.y < noseGateY &&
        rw.y < noseGateY;

      if (bothHandsUp) {
        if (!this.handsUpConsumed) {
          this.handsUpHoldSec = Math.min(HANDS_UP_HOLD_SEC, this.handsUpHoldSec + safeDt);
          if (this.handsUpHoldSec >= HANDS_UP_HOLD_SEC) {
            handsUpTriggered = true;
            this.handsUpConsumed = true;
          }
        }
      } else {
        this.handsUpHoldSec = Math.max(0, this.handsUpHoldSec - safeDt * 2.4);
        if (this.handsUpHoldSec <= 0.05) {
          this.handsUpConsumed = false;
        }
      }

      return {
        hasBody: true,
        isCalibrated: this.isCalibrated,
        lane: this.lane,
        laneChanged,
        jumpTriggered,
        crouchTriggered,
        isJumpZone,
        isCrouching: this.isCrouching,
        handsUpProgress: this.handsUpConsumed ? 1 : Math.min(1, this.handsUpHoldSec / HANDS_UP_HOLD_SEC),
        handsUpTriggered,
        midX: this.midX,
        midY: this.midY,
        shoulderWidth: this.shoulderW,
        baseX: this.baseX,
        baseY: this.baseY,
        baseShoulderWidth: this.baseShoulderW,
        dxNorm,
        dyUpNorm,
        pose: this.latestPose,
      };
    }

    // Brief persistence when pose drops for < 0.25s
    this.missedBodySec += safeDt;
    if (this.missedBodySec > 0.25) {
      this.hasBody = false;
      this.latestPose = null;
      this.isCrouching = false;
      this.handsUpHoldSec = Math.max(0, this.handsUpHoldSec - safeDt * 2.5);
      if (this.handsUpHoldSec <= 0.05) this.handsUpConsumed = false;
    }

    const sw = Math.max(0.1, this.baseShoulderW);
    return {
      hasBody: this.hasBody,
      isCalibrated: this.isCalibrated,
      lane: this.lane,
      laneChanged: false,
      jumpTriggered: false,
      crouchTriggered: false,
      isJumpZone: false,
      isCrouching: this.isCrouching,
      handsUpProgress: Math.min(1, this.handsUpHoldSec / HANDS_UP_HOLD_SEC),
      handsUpTriggered: false,
      midX: this.midX,
      midY: this.midY,
      shoulderWidth: this.shoulderW,
      baseX: this.baseX,
      baseY: this.baseY,
      baseShoulderWidth: this.baseShoulderW,
      dxNorm: (this.midX - this.baseX) / sw,
      dyUpNorm: (this.baseY - this.midY) / sw,
      pose: this.latestPose,
    };
  }
}

const UPPER_BODY_EDGES: readonly [number, number][] = [
  [11, 12], // shoulders
  [11, 13],
  [13, 15], // left arm
  [12, 14],
  [14, 16], // right arm
  [11, 23],
  [12, 24],
  [23, 24], // torso
];

/**
 * Renders the bottom-right 220x124 Camera PiP Body Guide:
 * - Live mirrored webcam feed (or sleek cyber grid fallback in mock/no-cam mode)
 * - 2 vertical Lane dividers + active lane tint
 * - 2 horizontal Jump (cyan) & Crouch (magenta) threshold lines lighting up on trigger
 * - Neon upper-body skeleton + glowing shoulder midpoint beacon
 */
export function drawRunnerPipGuide(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement | null,
  state: BodyControlState,
  mirrored: boolean,
  w: number,
  h: number,
): void {
  ctx.save();
  ctx.clearRect(0, 0, w, h);

  // 1. Live mirrored webcam background (cover-fit) or dark cyber studio backdrop
  const hasLiveVideo =
    !!video && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
  if (hasLiveVideo && video) {
    const vidAspect = video.videoWidth / video.videoHeight;
    const pipAspect = w / Math.max(h, 1);
    let sx = 0;
    let sy = 0;
    let sw = video.videoWidth;
    let sh = video.videoHeight;
    if (vidAspect > pipAspect) {
      sw = video.videoHeight * pipAspect;
      sx = (video.videoWidth - sw) * 0.5;
    } else {
      sh = video.videoWidth / pipAspect;
      sy = (video.videoHeight - sh) * 0.5;
    }

    ctx.save();
    if (mirrored) {
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, w, h);
    ctx.restore();

    // Subtle dark vignette so neon guide lines pop crisply
    ctx.fillStyle = 'rgba(5, 8, 18, 0.44)';
    ctx.fillRect(0, 0, w, h);
  } else {
    ctx.fillStyle = '#070a14';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 22; x < w; x += 22) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 20; y < h; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  }

  // 2. Lane vertical threshold dividers (baseX +/- 0.42 * baseShoulderW)
  const sw = Math.max(0.11, state.baseShoulderWidth);
  const leftX = Math.max(12, Math.min(w - 12, (state.baseX - LANE_ENTER_THRESHOLD * sw) * w));
  const rightX = Math.max(12, Math.min(w - 12, (state.baseX + LANE_ENTER_THRESHOLD * sw) * w));
  const jumpY = Math.max(10, Math.min(h - 10, (state.baseY - JUMP_HEIGHT_RATIO * sw) * h));
  const crouchY = Math.max(10, Math.min(h - 10, (state.baseY + CROUCH_ENTER_RATIO * sw) * h));

  // Highlight active lane band
  ctx.fillStyle = 'rgba(56, 189, 248, 0.14)';
  if (state.lane === -1) {
    ctx.fillRect(0, 0, leftX, h);
  } else if (state.lane === 0) {
    ctx.fillRect(leftX, 0, Math.max(0, rightX - leftX), h);
  } else {
    ctx.fillRect(rightX, 0, Math.max(0, w - rightX), h);
  }

  // Vertical lane divider lines
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1.25;
  ctx.strokeStyle =
    state.lane === -1 ? 'rgba(56, 189, 248, 0.95)' : 'rgba(255, 255, 255, 0.28)';
  ctx.beginPath();
  ctx.moveTo(leftX, 0);
  ctx.lineTo(leftX, h);
  ctx.stroke();

  ctx.strokeStyle =
    state.lane === 1 ? 'rgba(56, 189, 248, 0.95)' : 'rgba(255, 255, 255, 0.28)';
  ctx.beginPath();
  ctx.moveTo(rightX, 0);
  ctx.lineTo(rightX, h);
  ctx.stroke();

  // 3. Horizontal Jump & Crouch threshold lines
  ctx.setLineDash([5, 3]);
  const jumpLit = state.isJumpZone || state.jumpTriggered;
  ctx.lineWidth = jumpLit ? 2.2 : 1.2;
  ctx.strokeStyle = jumpLit ? '#67e8f9' : 'rgba(34, 211, 238, 0.55)';
  ctx.beginPath();
  ctx.moveTo(0, jumpY);
  ctx.lineTo(w, jumpY);
  ctx.stroke();

  const crouchLit = state.isCrouching || state.crouchTriggered;
  ctx.lineWidth = crouchLit ? 2.2 : 1.2;
  ctx.strokeStyle = crouchLit ? '#f0abfc' : 'rgba(217, 70, 239, 0.55)';
  ctx.beginPath();
  ctx.moveTo(0, crouchY);
  ctx.lineTo(w, crouchY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Tiny line labels
  ctx.font = '600 8px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textBaseline = 'top';
  ctx.fillStyle = jumpLit ? '#ffffff' : 'rgba(103, 232, 249, 0.85)';
  ctx.fillText('JUMP', 6, Math.max(2, jumpY - 10));
  ctx.fillStyle = crouchLit ? '#ffffff' : 'rgba(240, 171, 252, 0.85)';
  ctx.fillText('DUCK', 6, Math.min(h - 10, crouchY + 2));

  // 4. Neon Upper-Body Skeleton + Shoulder Midpoint Beacon
  const pose = state.pose;
  if (pose && pose.length >= 17) {
    const px = (idx: number) => ((mirrored ? 1 - pose[idx].x : pose[idx].x) * w);
    const py = (idx: number) => pose[idx].y * h;

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Glow under-stroke
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.45)';
    ctx.lineWidth = 4;
    for (let i = 0; i < UPPER_BODY_EDGES.length; i += 1) {
      const [a, b] = UPPER_BODY_EDGES[i];
      if (!pose[a] || !pose[b] || visOf(pose[a]) < 0.3 || visOf(pose[b]) < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
      ctx.stroke();
    }

    // Crisp bright neon core
    ctx.strokeStyle = '#e0f2fe';
    ctx.lineWidth = 1.6;
    for (let i = 0; i < UPPER_BODY_EDGES.length; i += 1) {
      const [a, b] = UPPER_BODY_EDGES[i];
      if (!pose[a] || !pose[b] || visOf(pose[a]) < 0.3 || visOf(pose[b]) < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
      ctx.stroke();
    }

    // Head ring from nose (lm[0])
    if (pose[0] && visOf(pose[0]) >= 0.3) {
      ctx.strokeStyle = '#67e8f9';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px(0), py(0), Math.max(6, sw * w * 0.24), 0, Math.PI * 2);
      ctx.stroke();
    }

    // Joint dots (shoulders, elbows, wrists)
    const joints = [11, 12, 13, 14, 15, 16];
    for (let i = 0; i < joints.length; i += 1) {
      const idx = joints[i];
      if (!pose[idx] || visOf(pose[idx]) < 0.3) continue;
      ctx.fillStyle = idx === 15 || idx === 16 ? '#f472b6' : '#38bdf8';
      ctx.beginPath();
      ctx.arc(px(idx), py(idx), 2.8, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Shoulder Midpoint Tracking Beacon
  if (state.hasBody) {
    const mx = state.midX * w;
    const my = state.midY * h;
    const beaconCol = jumpLit ? '#67e8f9' : crouchLit ? '#f0abfc' : '#ffffff';
    ctx.fillStyle = 'rgba(34, 211, 238, 0.28)';
    ctx.beginPath();
    ctx.arc(mx, my, 8, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = beaconCol;
    ctx.beginPath();
    ctx.arc(mx, my, 3.6, 0, Math.PI * 2);
    ctx.fill();
  }

  // Top-right lane status tag
  const laneTag = state.lane === -1 ? 'L' : state.lane === 1 ? 'R' : 'C';
  const actionTag = jumpLit ? ' · JUMP' : crouchLit ? ' · DUCK' : '';
  const pill = `${laneTag}${actionTag}`;
  ctx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
  const tw = ctx.measureText(pill).width;
  ctx.fillStyle = 'rgba(10, 12, 22, 0.78)';
  ctx.fillRect(w - tw - 14, 5, tw + 10, 15);
  ctx.fillStyle = '#e2e8f0';
  ctx.textBaseline = 'middle';
  ctx.fillText(pill, w - tw - 9, 12.5);

  ctx.restore();
}
