import { forwardRef, useImperativeHandle, useRef } from 'react';

export interface AirCursorHandle {
  update: (
    pos: { x: number; y: number } | null,
    dwellProgress: number,
    isHovering: boolean,
    pulseStrength: number,
  ) => void;
}

const RING_R = 16;
const RING_C = 2 * Math.PI * RING_R;

export const AirCursor = forwardRef<AirCursorHandle>(function AirCursor(_, ref) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const ringRef = useRef<SVGCircleElement | null>(null);
  const pulseRef = useRef<HTMLDivElement | null>(null);

  useImperativeHandle(ref, () => ({
    update(pos, dwellProgress, isHovering, pulseStrength) {
      const root = rootRef.current;
      if (!root) return;
      if (!pos) {
        root.style.opacity = '0';
        return;
      }
      root.style.opacity = '1';
      root.style.transform = `translate3d(${pos.x}px, ${pos.y}px, 0)`;

      const ring = ringRef.current;
      if (ring) {
        const clamped = Math.max(0, Math.min(1, dwellProgress));
        ring.style.strokeDashoffset = String(RING_C * (1 - clamped));
        ring.style.opacity = isHovering ? '1' : '0.25';
      }

      const pulse = pulseRef.current;
      if (pulse) {
        if (pulseStrength > 0.01) {
          pulse.style.opacity = String(pulseStrength);
          pulse.style.transform = `translate(-50%, -50%) scale(${1 + (1 - pulseStrength) * 1.4})`;
        } else {
          pulse.style.opacity = '0';
        }
      }
    },
  }));

  return (
    <div
      ref={rootRef}
      aria-hidden="true"
      className="pointer-events-none fixed top-0 left-0 z-40 opacity-0 will-change-transform"
    >
      <div
        ref={pulseRef}
        className="absolute top-0 left-0 h-9 w-9 rounded-full border-2 border-white opacity-0"
      />
      <svg
        width="40"
        height="40"
        viewBox="0 0 40 40"
        className="-translate-x-1/2 -translate-y-1/2 -rotate-90"
      >
        <circle
          cx="20"
          cy="20"
          r={RING_R}
          fill="none"
          stroke="rgba(255,255,255,0.22)"
          strokeWidth="2.5"
        />
        <circle
          ref={ringRef}
          cx="20"
          cy="20"
          r={RING_R}
          fill="none"
          stroke="#ffffff"
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={RING_C}
          strokeDashoffset={RING_C}
        />
        <circle cx="20" cy="20" r="3.5" fill="#ffffff" />
      </svg>
    </div>
  );
});
