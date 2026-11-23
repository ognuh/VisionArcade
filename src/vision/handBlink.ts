import type { GestureType, Landmark3D } from '../types/vision';

export interface BlinkTapEvent {
  normX: number;
  normY: number;
}

interface HandBlinkState {
  pinchedAtMs: number;
  fistAtMs: number;
  cooldownUntil: number;
}

const PINCH_CLOSE_DIST = 0.068;
const PINCH_OPEN_DIST = 0.095;
const BLINK_WINDOW_MS = 420;
const TAP_COOLDOWN_MS = 140;

function createState(): HandBlinkState {
  return { pinchedAtMs: -1, fistAtMs: -1, cooldownUntil: 0 };
}

/**
 * Rapid Hand Blink / Pinch-Tap Detector for Cyber Notebook typing:
 * Fires when user rapidly pinches & releases thumb (#4) + index (#8)
 * OR rapidly closes & opens fist (FIST -> OPEN).
 */
export class HandBlinkDetector {
  private states: [HandBlinkState, HandBlinkState] = [createState(), createState()];

  reset(): void {
    this.states = [createState(), createState()];
  }

  update(hands: Landmark3D[][], gestures: GestureType[], nowMs: number): BlinkTapEvent[] {
    const events: BlinkTapEvent[] = [];

    for (let h = 0; h < 2; h += 1) {
      const hand = hands[h];
      const st = this.states[h];
      const g = gestures[h] ?? 'IDLE';

      if (!hand || hand.length < 21) {
        st.pinchedAtMs = -1;
        st.fistAtMs = -1;
        continue;
      }

      const thumb = hand[4];
      const index = hand[8];
      const d48 = Math.hypot(thumb.x - index.x, thumb.y - index.y);

      let fired = false;

      // Path A: Rapid Pinch-Tap (4 & 8 close -> open)
      if (d48 < PINCH_CLOSE_DIST) {
        st.pinchedAtMs = nowMs;
      } else if (d48 > PINCH_OPEN_DIST && st.pinchedAtMs >= 0) {
        if (nowMs - st.pinchedAtMs <= BLINK_WINDOW_MS && nowMs >= st.cooldownUntil) {
          fired = true;
        }
        st.pinchedAtMs = -1;
      } else if (st.pinchedAtMs >= 0 && nowMs - st.pinchedAtMs > BLINK_WINDOW_MS) {
        st.pinchedAtMs = -1;
      }

      // Path B: Rapid Fist -> Open hand blink
      if (g === 'FIST') {
        st.fistAtMs = nowMs;
      } else if (g === 'OPEN' && st.fistAtMs >= 0) {
        if (nowMs - st.fistAtMs <= BLINK_WINDOW_MS && nowMs >= st.cooldownUntil) {
          fired = true;
        }
        st.fistAtMs = -1;
      } else if (st.fistAtMs >= 0 && nowMs - st.fistAtMs > BLINK_WINDOW_MS) {
        st.fistAtMs = -1;
      }

      if (fired) {
        st.cooldownUntil = nowMs + TAP_COOLDOWN_MS;
        events.push({ normX: index.x, normY: index.y });
      }
    }

    return events;
  }
}
