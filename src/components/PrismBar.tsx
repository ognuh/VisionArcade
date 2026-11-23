import { useEffect, type RefObject } from 'react';
import {
  BookOpen,
  Brush,
  CloudLightning,
  Contrast,
  Cpu,
  Eraser,
  Flame,
  FlipHorizontal2,
  Layers,
  Orbit,
  Scan,
  Shield,
  Sliders,
  SlidersHorizontal,
  Sun,
  Trash2,
  Wind,
  Zap,
} from 'lucide-react';
import { AIR_DRAW_COLORS } from '../prism/prismEngine';
import type { AirDrawColorId, GpuInfo, PrismModeId, PrismSettings } from '../types/vision';

interface PrismBarProps {
  settings: PrismSettings;
  onSettingsChange: (patch: Partial<PrismSettings>) => void;
  modeToast: string | null;
  gpuInfo: GpuInfo | null;
  fpsTextRef: RefObject<HTMLSpanElement | null>;
  aiFpsTextRef: RefObject<HTMLSpanElement | null>;
  modeTextRef: RefObject<HTMLSpanElement | null>;
  lightingBadgeRef: RefObject<HTMLSpanElement | null>;
  lightingTextRef: RefObject<HTMLSpanElement | null>;
  uiRootRef: RefObject<HTMLDivElement | null>;
  dockNavRef: RefObject<HTMLElement | null>;
  toolsOpen: boolean;
  onToggleTools: () => void;
  onDockActivity: () => void;
  onStepMode: (dir: -1 | 1) => void;
  onTriggerWhiteFlash: () => void;
  onClearDrawing: () => void;
  onClearNote: () => void;
}

const MODES: { id: PrismModeId; label: string; keyHint: string; icon: typeof Scan }[] = [
  { id: 'AUTO', label: 'Auto', keyHint: '1', icon: Scan },
  { id: 'GLASS', label: 'Glass', keyHint: '2', icon: Contrast },
  { id: 'THERMAL', label: 'Thermal', keyHint: '3', icon: Flame },
  { id: 'PRISM', label: 'Prism', keyHint: '4', icon: Layers },
  { id: 'GALAXY', label: 'Galaxy', keyHint: '5', icon: Orbit },
  { id: 'LIGHTNING', label: 'Lightning', keyHint: '6', icon: CloudLightning },
  { id: 'MYSTIC', label: 'Mystic', keyHint: '7', icon: Shield },
];

const DRAW_COLOR_KEYS: AirDrawColorId[] = ['CYAN', 'MAGENTA', 'GREEN', 'GOLD'];

export function PrismBar({
  settings,
  onSettingsChange,
  modeToast,
  gpuInfo,
  fpsTextRef,
  aiFpsTextRef,
  modeTextRef,
  lightingBadgeRef,
  lightingTextRef,
  uiRootRef,
  dockNavRef,
  toolsOpen,
  onToggleTools,
  onDockActivity,
  onStepMode,
  onTriggerWhiteFlash,
  onClearDrawing,
  onClearNote,
}: PrismBarProps) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === '1') onSettingsChange({ mode: 'AUTO', airDrawEnabled: false });
      else if (e.key === '2') onSettingsChange({ mode: 'GLASS', airDrawEnabled: false });
      else if (e.key === '3') onSettingsChange({ mode: 'THERMAL', airDrawEnabled: false });
      else if (e.key === '4') onSettingsChange({ mode: 'PRISM', airDrawEnabled: false });
      else if (e.key === '5') onSettingsChange({ mode: 'GALAXY', airDrawEnabled: false });
      else if (e.key === '6') onSettingsChange({ mode: 'LIGHTNING', airDrawEnabled: false });
      else if (e.key === '7') onSettingsChange({ mode: 'MYSTIC', airDrawEnabled: false });
      else if (e.key === 'ArrowRight') {
        e.preventDefault();
        onStepMode(1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        onStepMode(-1);
      } else if (e.key === 'f' || e.key === 'F') {
        if (settings.flashEnabled) {
          onTriggerWhiteFlash();
        }
      } else if (e.key === 't' || e.key === 'T') {
        onSettingsChange({ trailEnabled: !settings.trailEnabled });
      } else if (e.key === 'd' || e.key === 'D') {
        onSettingsChange({ airDrawEnabled: !settings.airDrawEnabled });
      } else if (e.key === 'h' || e.key === 'H') {
        onSettingsChange({ hdDenoise: !settings.hdDenoise });
      } else if (e.key === 'u' || e.key === 'U') {
        onSettingsChange({ cyberHud: !settings.cyberHud });
      } else if (e.key === 'm' || e.key === 'M') {
        onSettingsChange({ mirrored: !settings.mirrored });
      } else if (e.key === 'n' || e.key === 'N') {
        onSettingsChange({ noteOpen: !settings.noteOpen });
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    onSettingsChange,
    onStepMode,
    onTriggerWhiteFlash,
    settings.airDrawEnabled,
    settings.cyberHud,
    settings.flashEnabled,
    settings.hdDenoise,
    settings.mirrored,
    settings.noteOpen,
    settings.trailEnabled,
  ]);

  const gpuLabel = !gpuInfo
    ? 'GPU'
    : gpuInfo.delegate === 'CPU'
      ? 'WORKER CPU'
      : gpuInfo.tier === 'DISCRETE'
        ? 'RTX GPU'
        : 'GPU';

  return (
    <>
      {/* Top-left Pro Telemetry: VisionArcade | Render FPS | Worker AI FPS | GPU | Active Mode | Lighting Status */}
      <header className="pointer-events-none fixed top-4 left-4 z-20">
        <div className="pointer-events-auto inline-flex flex-wrap items-center gap-2 rounded-full border border-white/15 bg-black/52 px-3.5 py-1.5 backdrop-blur-md">
          <span className="text-xs font-semibold text-white">VisionArcade</span>
          <span className="h-3 w-px bg-white/20" aria-hidden="true" />
          <span className="font-mono text-xs font-semibold text-white tabular-nums">
            <span ref={fpsTextRef}>144</span> FPS
          </span>
          <span className="h-3 w-px bg-white/20" aria-hidden="true" />
          <span className="font-mono text-[11px] text-[#67e8f9] tabular-nums">
            AI <span ref={aiFpsTextRef}>30</span>
          </span>
          <span className="h-3 w-px bg-white/20" aria-hidden="true" />
          <span title={gpuInfo?.renderer ?? ''} className="font-mono text-[11px] text-white/75">
            {gpuLabel}
          </span>
          <span className="h-3 w-px bg-white/20" aria-hidden="true" />
          <span ref={modeTextRef} className="font-mono text-[11px] font-medium text-white">
            {settings.airDrawEnabled ? 'AIR DRAW' : settings.mode}
          </span>
          <span className="h-3 w-px bg-white/20" aria-hidden="true" />
          <span
            ref={lightingBadgeRef}
            className="inline-flex items-center gap-1 font-mono text-[11px] text-[#4ade80]"
          >
            <Sun className="h-3 w-3 shrink-0" aria-hidden="true" />
            <span ref={lightingTextRef}>LIGHT OK</span>
          </span>
        </div>
      </header>

      {/* Center-top 0.9s Neon HUD Toast when switching mode via Air Swipe or Arrow keys */}
      {modeToast && (
        <div
          role="status"
          aria-live="polite"
          className="pointer-events-none fixed top-5 left-1/2 z-30 -translate-x-1/2 rounded-full border border-[#22d3ee]/60 bg-black/70 px-4 py-1.5 backdrop-blur-lg"
        >
          <span className="font-mono text-xs font-semibold text-[#e0f2fe]">{modeToast}</span>
        </div>
      )}

      {/* Observed UI Container for ResizeObserver + MutationObserver Magnetic Hitbox sync */}
      <div ref={uiRootRef} className="pointer-events-none fixed inset-0 z-20">
        {/* Tools Popover Drawer above Dock */}
        {toolsOpen && (
          <div
            role="dialog"
            aria-label="Công cụ và bộ lọc"
            onPointerMove={onDockActivity}
            className="pointer-events-auto fixed bottom-18 left-1/2 flex w-[min(92vw,340px)] -translate-x-1/2 flex-col gap-2 rounded-2xl border border-white/18 bg-[#080b12]/88 p-2.5 backdrop-blur-xl"
          >
            <div className="grid grid-cols-2 gap-1.5">
              <button
                type="button"
                data-air-clickable="true"
                aria-pressed={settings.trailEnabled}
                onClick={() => onSettingsChange({ trailEnabled: !settings.trailEnabled })}
                className={`inline-flex h-9 items-center gap-2 rounded-lg px-3 text-xs font-medium transition-colors ${
                  settings.trailEnabled
                    ? 'bg-white/20 text-white'
                    : 'bg-white/5 text-white/75 hover:bg-white/12 hover:text-white'
                }`}
              >
                <Wind className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>Trail</span>
              </button>

              <button
                type="button"
                data-air-clickable="true"
                aria-pressed={settings.airDrawEnabled}
                onClick={() => onSettingsChange({ airDrawEnabled: !settings.airDrawEnabled })}
                className={`inline-flex h-9 items-center gap-2 rounded-lg px-3 text-xs font-medium transition-colors ${
                  settings.airDrawEnabled
                    ? 'bg-[#22d3ee] text-black'
                    : 'bg-white/5 text-white/75 hover:bg-white/12 hover:text-white'
                }`}
              >
                <Brush className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>Air Draw</span>
              </button>

              <button
                type="button"
                data-air-clickable="true"
                aria-pressed={settings.hdDenoise}
                onClick={() => onSettingsChange({ hdDenoise: !settings.hdDenoise })}
                className={`inline-flex h-9 items-center gap-2 rounded-lg px-3 text-xs font-medium transition-colors ${
                  settings.hdDenoise
                    ? 'bg-white/20 text-[#67e8f9]'
                    : 'bg-white/5 text-white/75 hover:bg-white/12 hover:text-white'
                }`}
              >
                <Sliders className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>HD+ Denoise</span>
              </button>

              <button
                type="button"
                data-air-clickable="true"
                aria-pressed={settings.cyberHud}
                onClick={() => onSettingsChange({ cyberHud: !settings.cyberHud })}
                className={`inline-flex h-9 items-center gap-2 rounded-lg px-3 text-xs font-medium transition-colors ${
                  settings.cyberHud
                    ? 'bg-white/20 text-[#a5f3fc]'
                    : 'bg-white/5 text-white/75 hover:bg-white/12 hover:text-white'
                }`}
              >
                <Cpu className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>Cyber HUD</span>
              </button>

              <button
                type="button"
                data-air-clickable="true"
                aria-pressed={settings.mirrored}
                onClick={() => onSettingsChange({ mirrored: !settings.mirrored })}
                className={`${
                  settings.noteOpen ? '' : 'col-span-2'
                } inline-flex h-9 items-center justify-center gap-2 rounded-lg px-3 text-xs font-medium transition-colors ${
                  settings.mirrored
                    ? 'bg-white/18 text-white'
                    : 'bg-white/5 text-white/70 hover:bg-white/12 hover:text-white'
                }`}
              >
                <FlipHorizontal2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>Mirror</span>
              </button>

              {settings.noteOpen && (
                <button
                  type="button"
                  data-air-clickable="true"
                  onClick={onClearNote}
                  className="inline-flex h-9 items-center justify-center gap-2 rounded-lg bg-white/8 px-3 text-xs font-medium text-white/90 transition-colors hover:bg-white/18"
                >
                  <Trash2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                  <span>Xóa chữ 3D</span>
                </button>
              )}
            </div>

            {settings.airDrawEnabled && (
              <div className="flex items-center justify-between border-t border-white/10 pt-2">
                <div className="flex items-center gap-1.5">
                  {DRAW_COLOR_KEYS.map((cid) => {
                    const c = AIR_DRAW_COLORS[cid];
                    const active = settings.drawColor === cid;
                    return (
                      <button
                        key={cid}
                        type="button"
                        data-air-clickable="true"
                        aria-label={c.label}
                        aria-pressed={active}
                        onClick={() => onSettingsChange({ drawColor: cid })}
                        className={`inline-flex h-8 w-8 items-center justify-center rounded-lg border transition-colors ${
                          active ? 'border-white bg-white/18' : 'border-white/15 bg-white/5'
                        }`}
                      >
                        <span
                          className="block h-3.5 w-3.5 rounded-xs"
                          style={{ backgroundColor: c.hex }}
                          aria-hidden="true"
                        />
                      </button>
                    );
                  })}
                </div>

                <button
                  type="button"
                  data-air-clickable="true"
                  onClick={onClearDrawing}
                  className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-white/8 px-2.5 text-[11px] font-medium text-white/90 transition-colors hover:bg-white/18"
                >
                  <Eraser className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>Xóa nét</span>
                </button>
              </div>
            )}
          </div>
        )}

        {/* Single slim VisionOS/macOS Floating Dock (~44px height) with Auto-Wake spring */}
        <nav
          ref={dockNavRef}
          aria-label="Thanh Dock điều khiển hiệu ứng"
          onPointerEnter={onDockActivity}
          onPointerMove={onDockActivity}
          style={{ transform: 'translateX(-50%) scale(1)' }}
          className="pointer-events-auto fixed bottom-4 left-1/2 flex h-11 max-w-[96vw] items-center gap-1 whitespace-nowrap rounded-full border border-white/20 bg-black/48 px-1.5 backdrop-blur-xl transition-[opacity,transform] duration-300 ease-out"
        >
          {/* Left Group: Modes */}
          <div className="flex shrink-0 items-center gap-0.5" role="group" aria-label="Chế độ hiệu ứng">
            {MODES.map((m) => {
              const Icon = m.icon;
              const active = !settings.airDrawEnabled && settings.mode === m.id;
              return (
                <button
                  key={m.id}
                  type="button"
                  data-air-clickable="true"
                  aria-pressed={active}
                  onClick={() => onSettingsChange({ mode: m.id, airDrawEnabled: false })}
                  title={`${m.label} (Phím ${m.keyHint})`}
                  className={`inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-xs font-medium transition-colors ${
                    active
                      ? 'bg-white text-black'
                      : 'text-white/78 hover:bg-white/12 hover:text-white'
                  }`}
                >
                  <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                  <span className="hidden md:inline">{m.label}</span>
                </button>
              );
            })}
          </div>

          <span className="mx-0.5 h-5 w-px shrink-0 bg-white/20" aria-hidden="true" />

          {/* Right Group: Tools Popover + Standalone White Flash + 3D Note */}
          <div className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              data-air-clickable="true"
              aria-expanded={toolsOpen}
              onClick={onToggleTools}
              title="Mở bảng công cụ (Trail, Air Draw, HD+ Denoise, Cyber HUD, Mirror)"
              aria-label="Mở bảng công cụ"
              className={`inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors ${
                toolsOpen || settings.airDrawEnabled
                  ? 'bg-white/22 text-[#67e8f9]'
                  : 'text-white/80 hover:bg-white/12 hover:text-white'
              }`}
            >
              <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
            </button>

            <button
              type="button"
              data-air-clickable="true"
              aria-pressed={settings.flashEnabled}
              onClick={() => onSettingsChange({ flashEnabled: !settings.flashEnabled })}
              title="Bật/tắt Flash trắng toàn màn hình (Búng tay hoặc phím F khi đang bật)"
              aria-label="Bật/tắt Flash trắng toàn màn hình"
              className={`inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors ${
                settings.flashEnabled
                  ? 'bg-[#fde047] text-black'
                  : 'text-white/80 hover:bg-white/12 hover:text-white'
              }`}
            >
              <Zap className="h-3.5 w-3.5" aria-hidden="true" />
            </button>

            <button
              type="button"
              data-air-clickable="true"
              aria-pressed={settings.noteOpen}
              onClick={() => onSettingsChange({ noteOpen: !settings.noteOpen })}
              title="Bật/tắt Chữ 3D Hologram (Phím N)"
              aria-label="Bật/tắt Chữ 3D Hologram"
              className={`inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors ${
                settings.noteOpen
                  ? 'bg-[#22d3ee] text-black'
                  : 'text-white/80 hover:bg-white/12 hover:text-white'
              }`}
            >
              <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          </div>
        </nav>
      </div>
    </>
  );
}
