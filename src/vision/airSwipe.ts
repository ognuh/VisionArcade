import type { GestureType, Landmark3D } from '../types/vision';

interface SamplePoint {
  x: number;
  y: number;
  t: number;
}

const MIN_WINDOW_MS = 80;
const MAX_WINDOW_MS = 280;
const MIN_DISPLACEMENT_X = 0.24; // 24% of viewport width
const MAX_VERTICAL_RATIO = 0.6;  // |Δy| < |Δx| * 0.6
const SWIPE_COOLDOWN_MS = 650;

/**
 * Air Swipe 2.0 — Sliding Window Displacement Detector:
 * - Tracks (x, y, t) of palm center (landmark 9, mirror-corrected) across a 280ms sliding window.
 * - Works naturally with either hand as long as the hand is NOT clenched into a tight FIST.
 * - Triggers when |Δx| >= 0.24 within 80ms–280ms and |Δy| < |Δx| * 0.6 (650ms cooldown).
 */
export class AirSwipeDetector {
  private histories: [SamplePoint[], SamplePoint[]] = [[], []];
  private cooldownUntil = 0;

  reset(): void {
    this.histories[0].length = 0;
    this.histories[1].length = 0;
    this.cooldownUntil = 0;
  }

  update(
    hands: Landmark3D[][],
    gestures: GestureType[],
    nowMs: number,
    mirrored: boolean,
  ): -1 | 0 | 1 {
    if (nowMs < this.cooldownUntil) {
      this.histories[0].length = 0;
      this.histories[1].length = 0;
      return 0;
    }

    for (let h = 0; h < 2; h += 1) {
      const hand = hands[h];
      const hist = this.histories[h];
      const g = gestures[h] ?? 'IDLE';

      if (!hand || hand.length < 21 || g === 'FIST') {
        hist.length = 0;
        continue;
      }

      const screenX = mirrored ? 1 - hand[9].x : hand[9].x;
      const screenY = hand[9].y;
      hist.push({ x: screenX, y: screenY, t: nowMs });

      while (hist.length > 0 && nowMs - hist[0].t > MAX_WINDOW_MS) {
        hist.shift();
      }

      if (hist.length < 2) continue;

      for (let i = 0; i < hist.length - 1; i += 1) {
        const dtMs = nowMs - hist[i].t;
        if (dtMs < MIN_WINDOW_MS) break;

        const dx = screenX - hist[i].x;
        const absDx = Math.abs(dx);
        const absDy = Math.abs(screenY - hist[i].y);

        if (absDx >= MIN_DISPLACEMENT_X && absDy < absDx * MAX_VERTICAL_RATIO) {
          this.cooldownUntil = nowMs + SWIPE_COOLDOWN_MS;
          this.histories[0].length = 0;
          this.histories[1].length = 0;
          return dx > 0 ? 1 : -1;
        }
      }
    }

    return 0;
  }
}
