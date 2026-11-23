import { forwardRef, useImperativeHandle, useRef } from 'react';

export interface HoloText3DHandle {
  setTilt: (rotateXDeg: number, rotateYDeg: number) => void;
}

interface HoloText3DProps {
  text: string;
  completed: boolean;
  sheenKey: number;
}

/**
 * Centered 3D Extruded Hologram Text ("xin chào tôi là hưng"):
 * - Zero box, zero border, zero card background — floats cleanly in screen center
 * - Perspective 900px + preserve-3d tilted smoothly (±15 deg) via imperative DOM ref at 144Hz
 * - Multi-layered 12px cyan -> indigo -> deep violet 3D extrusion + per-char depth pop + metallic sheen
 */
export const HoloText3D = forwardRef<HoloText3DHandle, HoloText3DProps>(function HoloText3D(
  { text, completed, sheenKey },
  ref,
) {
  const stageRef = useRef<HTMLDivElement | null>(null);

  useImperativeHandle(ref, () => ({
    setTilt(rotateXDeg, rotateYDeg) {
      const el = stageRef.current;
      if (!el) return;
      el.style.transform = `rotateX(${rotateXDeg.toFixed(2)}deg) rotateY(${rotateYDeg.toFixed(2)}deg)`;
    },
  }));

  const chars = Array.from(text);

  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed inset-0 z-20 flex items-center justify-center px-6"
      style={{ perspective: '900px' }}
    >
      <div
        ref={stageRef}
        className="relative flex flex-col items-center will-change-transform"
        style={{ transformStyle: 'preserve-3d' }}
      >
        {chars.length === 0 ? (
          <p className="font-mono text-sm font-medium tracking-wide text-white/60">
            Nháy tay để viết chữ 3D · Nắm đấm 1.5s để xóa
          </p>
        ) : (
          <div className="relative overflow-hidden py-3">
            <h1 className="holo-3d-text text-center text-4xl font-bold tracking-tight sm:text-6xl md:text-7xl">
              {chars.map((ch, idx) => (
                <span key={`${idx}-${ch}`} className="holo-char-pop">
                  {ch === ' ' ? '\u00A0' : ch}
                </span>
              ))}
            </h1>

            {/* Metallic sheen light sweep when full sentence finishes */}
            {completed && (
              <div
                key={sheenKey}
                aria-hidden="true"
                className="holo-sheen-bar pointer-events-none absolute inset-y-0 left-0 w-1/3"
                style={{
                  background:
                    'linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.78) 50%, rgba(165,243,252,0) 100%)',
                  mixBlendMode: 'screen',
                }}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
});
