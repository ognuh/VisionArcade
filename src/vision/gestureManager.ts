import type { CycleModeId, GestureType, Landmark3D, PrismSettings } from '../types/vision';

export type GestureOwner = 'UI_CURSOR' | 'AIR_DRAW' | 'MODE_GESTURE' | 'AIR_SWIPE' | 'NONE';

export interface SwipeIndicatorState {
  active: boolean;
  dir: -1 | 0 | 1;
  /** 0..1 progress toward commit threshold while hand is armed & swiping horizontally. */
  progress: number;
  /** Normalized screen coordinates [0..1] of the swiping hand palm. */
  normX: number;
  normY: number;
}

export interface GestureFrameContext {
  hands: Landmark3D[][];
  gestures: GestureType[];
  nowMs: number;
  settings: PrismSettings;
  activeMode: CycleModeId;
  isHoveringUi: boolean;
  cosmosBlackHoleBusy: boolean;
}

export interface GestureDecision {
  owner: GestureOwner;
  swipeDir: -1 | 0 | 1;
  swipeIndicator: SwipeIndicatorState;
  allowModeGestures: boolean;
  allowAirDraw: boolean;
}

interface SwipeArmState {
  armed: boolean;
  dir: -1 | 0 | 1;
  armStartMs: number;
  startX: number;
  startY: number;
  lastX: number;
  lastY: number;
  lastT: number;
  peakSpeed: number;
}

// Strict Air Swipe 2-Phase Arm -> Commit + Return-Stroke Lock constants
const SWIPE_ZONE_MIN_Y = 0.15;
const SWIPE_ZONE_MAX_Y = 0.72;
const UI_DOCK_ZONE_Y = 0.75;
const ARM_MIN_DURATION_MS = 100;
const COMMIT_MIN_WINDOW_MS = 120;
const COMMIT_MAX_WINDOW_MS = 380;
const COMMIT_MIN_DX = 0.22;
const MAX_ANGLE_RATIO = 0.55; // |Δy| < 0.55 * |Δx| (~29 deg max deviation)
const ARM_MIN_SPEED = 0.32; // normalized screen widths / sec
const COMMIT_MIN_PEAK_SPEED = 0.68;
const RETURN_STROKE_LOCK_MS = 950;
const SETTLE_SPEED_THRESHOLD = 0.24;

function dist2D(a: Landmark3D, b: Landmark3D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Counts clearly extended fingers (0..5) on a raw 21-landmark hand. */
export function countRawExtendedFingers(hand: Landmark3D[]): number {
  const wrist = hand[0];
  let count = 0;
  if (dist2D(hand[4], hand[17]) > dist2D(hand[3], hand[17]) * 1.05) count += 1;
  if (dist2D(hand[8], wrist) > dist2D(hand[6], wrist) * 1.07) count += 1;
  if (dist2D(hand[12], wrist) > dist2D(hand[10], wrist) * 1.07) count += 1;
  if (dist2D(hand[16], wrist) > dist2D(hand[14], wrist) * 1.07) count += 1;
  if (dist2D(hand[20], wrist) > dist2D(hand[18], wrist) * 1.07) count += 1;
  return count;
}

function isTwoFingerRawPose(hand: Landmark3D[]): boolean {
  const wrist = hand[0];
  const idx = dist2D(hand[8], wrist) > dist2D(hand[6], wrist) * 1.08;
  const mid = dist2D(hand[12], wrist) > dist2D(hand[10], wrist) * 1.08;
  const ring = dist2D(hand[16], wrist) < dist2D(hand[14], wrist) * 1.04;
  const pinky = dist2D(hand[20], wrist) < dist2D(hand[18], wrist) * 1.04;
  return idx && mid && ring && pinky;
}

/**
 * Centralized Gesture Orchestrator (Priority Finite State Machine):
 * Strict Priority: UI Air-Cursor (bottom 25%) > Air Draw > Mode-Specific Gestures > Air Swipe.
 * Includes 2-Phase Arm -> Commit Air Swipe + 950ms Return-Stroke Lock so pulling the hand
 * back to rest never triggers an unintended reverse swipe.
 */
export class GestureManager {
  private armStates: [SwipeArmState, SwipeArmState] = [
    this.createEmptyArm(),
    this.createEmptyArm(),
  ];
  private lockUntilMs = 0;
  private awaitingSettle = false;
  private lastSwipeDir: -1 | 0 | 1 = 0;

  private createEmptyArm(): SwipeArmState {
    return {
      armed: false,
      dir: 0,
      armStartMs: 0,
      startX: 0,
      startY: 0,
      lastX: 0,
      lastY: 0,
      lastT: 0,
      peakSpeed: 0,
    };
  }

  public reset(): void {
    this.armStates[0] = this.createEmptyArm();
    this.armStates[1] = this.createEmptyArm();
    this.lockUntilMs = 0;
    this.awaitingSettle = false;
    this.lastSwipeDir = 0;
  }

  private clearArms(): void {
    this.armStates[0].armed = false;
    this.armStates[0].dir = 0;
    this.armStates[0].lastT = 0;
    this.armStates[1].armed = false;
    this.armStates[1].dir = 0;
    this.armStates[1].lastT = 0;
  }

  public update(ctx: GestureFrameContext): GestureDecision {
    const emptyIndicator: SwipeIndicatorState = {
      active: false,
      dir: 0,
      progress: 0,
      normX: 0.5,
      normY: 0.5,
    };

    const { hands, gestures, nowMs, settings, activeMode, isHoveringUi, cosmosBlackHoleBusy } = ctx;
    // Full-body Runner mode owns the stage — disable hand Air Swipe while playing Runner
    if (activeMode === 'RUNNER') {
      this.clearArms();
      this.awaitingSettle = false;
      return {
        owner: 'MODE_GESTURE',
        swipeDir: 0,
        swipeIndicator: emptyIndicator,
        allowModeGestures: true,
        allowAirDraw: false,
      };
    }

    if (hands.length === 0) {
      this.clearArms();
      this.awaitingSettle = false;
      return {
        owner: 'NONE',
        swipeDir: 0,
        swipeIndicator: emptyIndicator,
        allowModeGestures: true,
        allowAirDraw: false,
      };
    }

    // ---- Priority 1: UI Air-Cursor (bottom 25% of viewport or actively hovering a UI button) ----
    const lead = hands[0];
    const leadTipY = lead && lead.length >= 21 ? lead[8].y : 0;
    if (isHoveringUi || leadTipY >= UI_DOCK_ZONE_Y) {
      this.clearArms();
      return {
        owner: 'UI_CURSOR',
        swipeDir: 0,
        swipeIndicator: emptyIndicator,
        allowModeGestures: false,
        allowAirDraw: false,
      };
    }

    // ---- Priority 2: Air Draw (blocks Air Swipe completely during drawing) ----
    if (settings.airDrawEnabled) {
      this.clearArms();
      return {
        owner: 'AIR_DRAW',
        swipeDir: 0,
        swipeIndicator: emptyIndicator,
        allowModeGestures: false,
        allowAirDraw: true,
      };
    }

    // ---- Priority 3: Mode-Specific Gestures (Cosmos 1-finger / 2-finger / 2-hand Black Hole) ----
    if (activeMode === 'COSMOS') {
      if (cosmosBlackHoleBusy) {
        this.clearArms();
        return {
          owner: 'MODE_GESTURE',
          swipeDir: 0,
          swipeIndicator: emptyIndicator,
          allowModeGestures: true,
          allowAirDraw: false,
        };
      }

      // If 2 wide-open hands are charging Black Hole, block Air Swipe immediately
      if (hands.length >= 2) {
        const f0 = countRawExtendedFingers(hands[0]);
        const f1 = countRawExtendedFingers(hands[1]);
        const p0x = (hands[0][0].x + hands[0][9].x) * 0.5;
        const p0y = (hands[0][0].y + hands[0][9].y) * 0.5;
        const p1x = (hands[1][0].x + hands[1][9].x) * 0.5;
        const p1y = (hands[1][0].y + hands[1][9].y) * 0.5;
        if (f0 >= 4 && f1 >= 4 && Math.hypot(p1x - p0x, p1y - p0y) >= 0.24) {
          this.clearArms();
          return {
            owner: 'MODE_GESTURE',
            swipeDir: 0,
            swipeIndicator: emptyIndicator,
            allowModeGestures: true,
            allowAirDraw: false,
          };
        }
      }

      // Single-hand 1-finger POINT or 2-finger switch owns Cosmos (cannot swipe with < 4 fingers anyway)
      if (hands.length === 1) {
        const g0 = gestures[0] ?? 'IDLE';
        const openCount = countRawExtendedFingers(hands[0]);
        if (openCount < 4 || g0 === 'POINT' || isTwoFingerRawPose(hands[0])) {
          this.clearArms();
          return {
            owner: 'MODE_GESTURE',
            swipeDir: 0,
            swipeIndicator: emptyIndicator,
            allowModeGestures: true,
            allowAirDraw: false,
          };
        }
      }
    }

    // ---- Priority 4: 2-Phase Arm -> Commit Air Swipe with 950ms Return-Stroke Lock ----
    const lockedByTime = nowMs < this.lockUntilMs;

    let bestIndicator: SwipeIndicatorState = emptyIndicator;
    let committedDir: -1 | 0 | 1 = 0;

    for (let h = 0; h < Math.min(hands.length, 2); h += 1) {
      const hand = hands[h];
      const arm = this.armStates[h];
      if (!hand || hand.length < 21) {
        arm.armed = false;
        arm.lastT = 0;
        continue;
      }

      const openFingers = countRawExtendedFingers(hand);
      const rawX = (hand[0].x + hand[9].x) * 0.5;
      const rawY = (hand[0].y + hand[9].y) * 0.5;
      const sx = settings.mirrored ? 1 - rawX : rawX;
      const sy = rawY;

      const dtSec = arm.lastT > 0 ? Math.max(0.008, (nowMs - arm.lastT) / 1000) : 0.033;
      const vx = arm.lastT > 0 ? (sx - arm.lastX) / dtSec : 0;

      // Return-stroke settle check: wait for hand to slow down after a commit before unlocking
      if (this.awaitingSettle) {
        if (!lockedByTime && Math.abs(vx) < SETTLE_SPEED_THRESHOLD) {
          this.awaitingSettle = false;
        } else if (
          lockedByTime &&
          this.lastSwipeDir !== 0 &&
          Math.sign(vx) === -this.lastSwipeDir
        ) {
          // Actively returning hand in opposite direction -> keep locked
          arm.lastX = sx;
          arm.lastY = sy;
          arm.lastT = nowMs;
          arm.armed = false;
          continue;
        }
      }

      if (lockedByTime || this.awaitingSettle) {
        arm.lastX = sx;
        arm.lastY = sy;
        arm.lastT = nowMs;
        arm.armed = false;
        continue;
      }

      // Gate: >= 4 open fingers AND safe Y band
      // In FLUID mode, restrict Air Swipe to top-edge band (0.08..0.26) with stricter dx/speed so particle play never misfires
      const isFluidMode = activeMode === 'FLUID';
      const minY = isFluidMode ? 0.08 : SWIPE_ZONE_MIN_Y;
      const maxY = isFluidMode ? 0.26 : SWIPE_ZONE_MAX_Y;
      const minCommitDx = isFluidMode ? 0.30 : COMMIT_MIN_DX;
      const minCommitSpeed = isFluidMode ? 1.15 : COMMIT_MIN_PEAK_SPEED;

      if (openFingers < 4 || sy < minY || sy > maxY) {
        arm.armed = false;
        arm.dir = 0;
        arm.lastX = sx;
        arm.lastY = sy;
        arm.lastT = nowMs;
        continue;
      }

      const instDir: -1 | 0 | 1 =
        vx > ARM_MIN_SPEED ? 1 : vx < -ARM_MIN_SPEED ? -1 : 0;

      if (!arm.armed) {
        if (instDir !== 0) {
          arm.armed = true;
          arm.dir = instDir;
          arm.armStartMs = nowMs;
          arm.startX = arm.lastT > 0 ? arm.lastX : sx;
          arm.startY = arm.lastT > 0 ? arm.lastY : sy;
          arm.peakSpeed = Math.abs(vx);
        }
      } else {
        const elapsedMs = nowMs - arm.armStartMs;
        const dx = sx - arm.startX;
        const dy = sy - arm.startY;
        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);
        arm.peakSpeed = Math.max(arm.peakSpeed, Math.abs(vx));

        // Require continuous same-direction motion during arming (blocks 1-frame teleport false positives)
        const wrongDir = instDir !== 0 && instDir !== arm.dir;
        const stalledEarly =
          vx * arm.dir < 0.12 && (elapsedMs < COMMIT_MIN_WINDOW_MS || absDx < COMMIT_MIN_DX);
        const tooVertical = absDx > 0.04 && absDy > absDx * MAX_ANGLE_RATIO * 1.35;
        if (wrongDir || stalledEarly || elapsedMs > COMMIT_MAX_WINDOW_MS || tooVertical) {
          if (instDir !== 0 && wrongDir) {
            arm.dir = instDir;
            arm.armStartMs = nowMs;
            arm.startX = sx;
            arm.startY = sy;
            arm.peakSpeed = Math.abs(vx);
          } else {
            arm.armed = false;
            arm.dir = 0;
          }
        } else {
          // Phase 1 confirmed (>= 100ms monotonic horizontal travel): show Directional Indicator
          if (elapsedMs >= ARM_MIN_DURATION_MS && absDx > 0.05) {
            const prog = Math.min(1, absDx / minCommitDx);
            if (!bestIndicator.active || prog > bestIndicator.progress) {
              bestIndicator = {
                active: true,
                dir: arm.dir,
                progress: prog,
                normX: sx,
                normY: sy,
              };
            }
          }

          // Phase 2 Commit: |Δx| >= minCommitDx within 120ms-380ms, angle < ~29 deg, sufficient peak velocity
          if (
            elapsedMs >= COMMIT_MIN_WINDOW_MS &&
            elapsedMs <= COMMIT_MAX_WINDOW_MS &&
            absDx >= minCommitDx &&
            absDy < absDx * MAX_ANGLE_RATIO &&
            arm.peakSpeed >= minCommitSpeed
          ) {
            committedDir = dx > 0 ? 1 : -1;
            this.lockUntilMs = nowMs + RETURN_STROKE_LOCK_MS;
            this.awaitingSettle = true;
            this.lastSwipeDir = committedDir;
            this.clearArms();
            break;
          }
        }
      }

      arm.lastX = sx;
      arm.lastY = sy;
      arm.lastT = nowMs;
    }

    if (committedDir !== 0) {
      return {
        owner: 'AIR_SWIPE',
        swipeDir: committedDir,
        swipeIndicator: emptyIndicator,
        allowModeGestures: false,
        allowAirDraw: false,
      };
    }

    return {
      owner: bestIndicator.active ? 'AIR_SWIPE' : 'MODE_GESTURE',
      swipeDir: 0,
      swipeIndicator: bestIndicator,
      allowModeGestures: !bestIndicator.active,
      allowAirDraw: false,
    };
  }
}
