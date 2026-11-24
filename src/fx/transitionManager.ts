import type { CycleModeId } from '../types/vision';

export const MODE_ENTER_SEC = 0.28; // 280ms smooth Cubic Ease-Out enter
export const MODE_EXIT_SEC = 0.22; // 220ms smooth shrink/fade exit

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function easeOutCubic(t: number): number {
  const k = clamp01(t);
  return 1 - Math.pow(1 - k, 3);
}

/** Smooth elastic spring overshoot curve: 0.2 -> 1.04 -> 1.0 for 3D model reveals. */
export function elasticScale02To10(t: number): number {
  const k = clamp01(t);
  if (k <= 0) return 0.2;
  if (k >= 1) return 1.0;
  // Critically underdamped overshoot peaking cleanly near 1.04 around k=0.72
  const base = 1 - Math.pow(1 - k, 3);
  const overshoot = Math.sin(k * Math.PI) * (1 - k) * 0.14;
  return 0.2 + 0.8 * base + overshoot;
}

export interface ModeTransitionSnapshot {
  activeMode: CycleModeId;
  enterProgress: number; // 0..1 eased ENTER progress
  activeAlpha: number; // 0..1 opacity for incoming/active mode
  outgoingMode: CycleModeId | null;
  outgoingProgress: number; // 1 -> 0 remaining life of outgoing mode
  outgoingAlpha: number; // 1 -> 0 opacity for outgoing mode
  isTransitioning: boolean;
}

/**
 * Unified 3-Phase Scene Transition Controller (ENTER 280ms -> ACTIVE -> EXIT 220ms):
 * Crossfades outgoing and incoming VFX modes in parallel so switching modes never pops or flickers.
 */
export class TransitionManager {
  private activeMode: CycleModeId = 'GLASS';
  private enterElapsed = MODE_ENTER_SEC;
  private outgoingMode: CycleModeId | null = null;
  private exitElapsed = MODE_EXIT_SEC;
  private initialized = false;

  public reset(initialMode: CycleModeId = 'GLASS'): void {
    this.activeMode = initialMode;
    this.enterElapsed = MODE_ENTER_SEC;
    this.outgoingMode = null;
    this.exitElapsed = MODE_EXIT_SEC;
    this.initialized = true;
  }

  /** Requests a mode transition; returns true when a real switch started. */
  public setMode(nextMode: CycleModeId): boolean {
    if (!this.initialized) {
      this.reset(nextMode);
      return false;
    }
    if (nextMode === this.activeMode) return false;

    this.outgoingMode = this.activeMode;
    this.exitElapsed = 0;
    this.activeMode = nextMode;
    this.enterElapsed = 0;
    return true;
  }

  public update(dt: number): ModeTransitionSnapshot {
    if (this.enterElapsed < MODE_ENTER_SEC) {
      this.enterElapsed = Math.min(MODE_ENTER_SEC, this.enterElapsed + dt);
    }

    if (this.outgoingMode !== null) {
      this.exitElapsed += dt;
      if (this.exitElapsed >= MODE_EXIT_SEC) {
        this.outgoingMode = null;
        this.exitElapsed = MODE_EXIT_SEC;
      }
    }

    const enterRaw = clamp01(this.enterElapsed / MODE_ENTER_SEC);
    const enterProgress = easeOutCubic(enterRaw);
    const outgoingRem =
      this.outgoingMode !== null ? clamp01(1 - this.exitElapsed / MODE_EXIT_SEC) : 0;
    const outgoingAlpha = outgoingRem * outgoingRem;

    return {
      activeMode: this.activeMode,
      enterProgress,
      activeAlpha: enterProgress,
      outgoingMode: this.outgoingMode,
      outgoingProgress: outgoingRem,
      outgoingAlpha,
      isTransitioning: enterRaw < 1 || this.outgoingMode !== null,
    };
  }
}
