import type { Landmark3D } from '../types/vision';

export interface SnapEvent {
  /** Normalized MediaPipe coordinates (before mirror/cover mapping). */
  normX: number;
  normY: number;
}

interface HandSnapState {
  prepReadyAt: number;
  prepMidX: number;
  prepMidY: number;
  prevMiddleToPalm: number;
  cooldownUntil: number;
}

const PREP_DIST_THRESHOLD = 0.075;
const SNAP_SEPARATION_THRESHOLD = 0.11;
const SNAP_WINDOW_MS = 220;
const MIDDLE_INWARD_SNAP_DELTA = 0.024;
const SNAP_COOLDOWN_MS = 500;

function createState(): HandSnapState {
  return {
    prepReadyAt: -1,
    prepMidX: 0.5,
    prepMidY: 0.5,
    prevMiddleToPalm: 0,
    cooldownUntil: 0,
  };
}

function dist2D(a: Landmark3D, b: Landmark3D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Real-world tuned Finger Snap detector (Thumb #4 & Middle #12):
 * Phase 1: dist(4, 12) < 0.075.
 * Phase 2: Within < 220ms, dist(4, 12) > 0.11 OR middle tip (#12) snaps rapidly
 * toward palm base (#0 or #5). 500ms cooldown per hand.
 */
export class FingerSnapDetector {
  private states: [HandSnapState, HandSnapState] = [createState(), createState()];

  reset(): void {
    this.states = [createState(), createState()];
  }

  update(hands: Landmark3D[][], nowMs: number): SnapEvent[] {
    const events: SnapEvent[] = [];

    for (let h = 0; h < 2; h += 1) {
      const hand = hands[h];
      const st = this.states[h];

      if (!hand || hand.length < 21) {
        st.prepReadyAt = -1;
        st.prevMiddleToPalm = 0;
        continue;
      }

      const thumb = hand[4];
      const middle = hand[12];
      const wrist = hand[0];
      const indexMcp = hand[5];

      const thumbMiddleDist = dist2D(thumb, middle);
      const middleToPalm = Math.min(dist2D(middle, wrist), dist2D(middle, indexMcp));

      // Phase 1: Thumb & Middle pinched close (< 0.075)
      if (thumbMiddleDist < PREP_DIST_THRESHOLD) {
        st.prepReadyAt = nowMs;
        st.prepMidX = (thumb.x + middle.x) * 0.5;
        st.prepMidY = (thumb.y + middle.y) * 0.5;
        st.prevMiddleToPalm = middleToPalm;
        continue;
      }

      // Phase 2: Within < 220ms, either wide separation (> 0.11) OR fast inward middle-finger snap
      if (st.prepReadyAt >= 0) {
        const elapsed = nowMs - st.prepReadyAt;
        if (elapsed <= SNAP_WINDOW_MS) {
          const inwardMove = st.prevMiddleToPalm - middleToPalm;
          const separated = thumbMiddleDist > SNAP_SEPARATION_THRESHOLD;
          const fastInward = inwardMove > MIDDLE_INWARD_SNAP_DELTA && thumbMiddleDist > 0.082;

          if ((separated || fastInward) && nowMs >= st.cooldownUntil) {
            events.push({ normX: st.prepMidX, normY: st.prepMidY });
            st.cooldownUntil = nowMs + SNAP_COOLDOWN_MS;
            st.prepReadyAt = -1;
          }
        } else {
          st.prepReadyAt = -1;
        }
      }

      st.prevMiddleToPalm = middleToPalm;
    }

    return events;
  }
}
