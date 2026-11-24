import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import {
  BookOpen,
  ChevronDown,
  CloudLightning,
  Contrast,
  Flame,
  Footprints,
  Layers,
  Orbit,
  Scan,
  Shield,
  SlidersHorizontal,
  Sparkles,
  Waves,
} from 'lucide-react';
import { AIR_DRAW_COLORS } from '../prism/prismEngine';
import type {
  AiEngineMode,
  AirDrawColorId,
  CameraDeviceItem,
  CameraPresetId,
  FpsTarget,
  GraphicsQuality,
  PrismModeId,
  PrismSettings,
} from '../types/vision';

interface PrismBarProps {
  settings: PrismSettings;
  onSettingsChange: (patch: Partial<PrismSettings>) => void;
  modeToast: { text: string; key: number } | null;
  cameras: CameraDeviceItem[];
  selectedCameraId: string;
  cameraPreset: CameraPresetId;
  activeAiDelegate: 'GPU' | 'CPU';
  onSelectCamera: (deviceId: string) => void;
  onSelectCameraPreset: (preset: CameraPresetId) => void;
  fpsTextRef: RefObject<HTMLSpanElement | null>;
  camFpsTextRef: RefObject<HTMLSpanElement | null>;
  aiFpsTextRef: RefObject<HTMLSpanElement | null>;
  modeTextRef: RefObject<HTMLSpanElement | null>;
  uiRootRef: RefObject<HTMLDivElement | null>;
  dockNavRef: RefObject<HTMLElement | null>;
  toolsOpen: boolean;
  onToggleTools: () => void;
  onDockActivity: () => void;
  onStepMode: (dir: -1 | 1) => void;
  onClearDrawing: () => void;
  onClearNote: () => void;
}

const MODES: { id: PrismModeId; label: string; keyHint: string; icon: typeof Scan }[] = [
  { id: 'AUTO', label: 'Auto', keyHint: '1', icon: Scan },
  { id: 'GLASS', label: 'Glass', keyHint: '2', icon: Contrast },
  { id: 'THERMAL', label: 'Thermal', keyHint: '3', icon: Flame },
  { id: 'PRISM', label: 'Prism', keyHint: '4', icon: Layers },
  { id: 'COSMOS', label: 'Cosmos', keyHint: '5', icon: Orbit },
  { id: 'LIGHTNING', label: 'Lightning', keyHint: '6', icon: CloudLightning },
  { id: 'MYSTIC', label: 'Mystic', keyHint: '7', icon: Shield },
  { id: 'RIPPLE', label: 'Ripple', keyHint: '8', icon: Waves },
  { id: 'FLUID', label: 'Fluid', keyHint: '9', icon: Sparkles },
  { id: 'RUNNER', label: 'Runner', keyHint: '0', icon: Footprints },
];

const CAMERA_PRESETS: { id: CameraPresetId; label: string }[] = [
  { id: 'AUTO', label: 'Auto' },
  { id: '720P60', label: '720p 60' },
  { id: '720P30', label: '720p 30' },
  { id: '1080P60', label: '1080p 60' },
];

const QUALITY_MODES: { id: GraphicsQuality; label: string }[] = [
  { id: 'ULTRA', label: 'Ultra' },
  { id: 'HIGH', label: 'High' },
];

const FPS_TARGETS: { id: FpsTarget; label: string }[] = [
  { id: 'MAX', label: 'Max Hz' },
  { id: 120, label: '120' },
  { id: 60, label: '60' },
];

const AI_ENGINE_MODES: { id: AiEngineMode; label: string }[] = [
  { id: 'GPU', label: 'GPU' },
  { id: 'AUTO', label: 'Auto' },
  { id: 'CPU', label: 'CPU' },
];

const DRAW_COLOR_KEYS: AirDrawColorId[] = ['CYAN', 'MAGENTA', 'GREEN', 'GOLD'];

interface ToggleRowProps {
  label: string;
  checked: boolean;
  onToggle: () => void;
}

function StudioToggleRow({ label, checked, onToggle }: ToggleRowProps) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-xs font-normal text-white/85">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        data-air-clickable="true"
        onClick={onToggle}
        className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors duration-200 ${
          checked ? 'bg-white' : 'bg-white/15 hover:bg-white/22'
        }`}
      >
        <span
          aria-hidden="true"
          className={`pointer-events-none inline-block h-4 w-4 rounded-full transition-transform duration-200 ${
            checked ? 'translate-x-[18px] bg-neutral-950' : 'translate-x-[2px] bg-white/85'
          }`}
        />
      </button>
    </div>
  );
}

export function PrismBar({
  settings,
  onSettingsChange,
  modeToast,
  cameras,
  selectedCameraId,
  cameraPreset,
  activeAiDelegate,
  onSelectCamera,
  onSelectCameraPreset,
  fpsTextRef,
  camFpsTextRef,
  aiFpsTextRef,
  modeTextRef,
  uiRootRef,
  dockNavRef,
  toolsOpen,
  onToggleTools,
  onDockActivity,
  onStepMode,
  onClearDrawing,
  onClearNote,
}: PrismBarProps) {
  const modeBtnRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const [pillRect, setPillRect] = useState<{ x: number; w: number; visible: boolean }>({
    x: 0,
    w: 0,
    visible: false,
  });

  const activeModeIndex = settings.airDrawEnabled
    ? -1
    : MODES.findIndex((m) => m.id === settings.mode);

  useLayoutEffect(() => {
    const syncPill = () => {
      if (activeModeIndex < 0) {
        setPillRect((prev) => (prev.visible ? { ...prev, visible: false } : prev));
        return;
      }
      const btn = modeBtnRefs.current[activeModeIndex];
      if (!btn) return;
      const x = btn.offsetLeft;
      const w = btn.offsetWidth;
      setPillRect((prev) =>
        prev.x === x && prev.w === w && prev.visible ? prev : { x, w, visible: true },
      );
    };

    syncPill();
    window.addEventListener('resize', syncPill);
    return () => window.removeEventListener('resize', syncPill);
  }, [activeModeIndex]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }
      if (e.key === '1') onSettingsChange({ mode: 'AUTO', airDrawEnabled: false });
      else if (e.key === '2') onSettingsChange({ mode: 'GLASS', airDrawEnabled: false });
      else if (e.key === '3') onSettingsChange({ mode: 'THERMAL', airDrawEnabled: false });
      else if (e.key === '4') onSettingsChange({ mode: 'PRISM', airDrawEnabled: false });
      else if (e.key === '5') onSettingsChange({ mode: 'COSMOS', airDrawEnabled: false });
      else if (e.key === '6') onSettingsChange({ mode: 'LIGHTNING', airDrawEnabled: false });
      else if (e.key === '7') onSettingsChange({ mode: 'MYSTIC', airDrawEnabled: false });
      else if (e.key === '8') onSettingsChange({ mode: 'RIPPLE', airDrawEnabled: false });
      else if (e.key === '9') onSettingsChange({ mode: 'FLUID', airDrawEnabled: false });
      else if (e.key === '0') onSettingsChange({ mode: 'RUNNER', airDrawEnabled: false });
      else if (e.key === 'ArrowRight') {
        e.preventDefault();
        onStepMode(1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        onStepMode(-1);
      } else if (e.key === 't' || e.key === 'T') {
        onSettingsChange({ trailEnabled: !settings.trailEnabled });
      } else if (e.key === 'd' || e.key === 'D') {
        onSettingsChange({ airDrawEnabled: !settings.airDrawEnabled });
      } else if (e.key === 'u' || e.key === 'U') {
        onSettingsChange({ cyberHud: !settings.cyberHud });
      } else if (e.key === 'm' || e.key === 'M') {
        onSettingsChange({ mirrored: !settings.mirrored });
      } else if (e.key === 'n' || e.key === 'N') {
        onSettingsChange({ noteOpen: !settings.noteOpen });
      } else if (e.key === 'Escape' && toolsOpen) {
        onToggleTools();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    onSettingsChange,
    onStepMode,
    onToggleTools,
    settings.airDrawEnabled,
    settings.cyberHud,
    settings.mirrored,
    settings.noteOpen,
    settings.trailEnabled,
    toolsOpen,
  ]);

  return (
    <>
      {/* Top-Left Minimalist Pro Telemetry Pill */}
      <header className="pointer-events-none fixed top-4 left-4 z-20">
        <div
          data-testid="telemetry-pill"
          className="pointer-events-auto inline-flex h-7 items-center gap-2 rounded-full border border-white/12 bg-neutral-950/60 px-3 font-mono text-[11px] backdrop-blur-2xl"
        >
          <span className="font-sans text-[11px] font-semibold tracking-tight text-white/95">
            VisionArcade
          </span>
          <span className="text-white/25" aria-hidden="true">
            ·
          </span>
          <span className="inline-flex items-center gap-1.5 text-white/90 tabular-nums">
            <span
              aria-hidden="true"
              className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.75)]"
            />
            <span>
              <span ref={fpsTextRef}>120</span> FPS
            </span>
          </span>
          <span className="text-white/25" aria-hidden="true">
            ·
          </span>
          <span className="text-white/60 tabular-nums">
            CAM <span ref={camFpsTextRef}>60</span>
          </span>
          <span className="text-white/25" aria-hidden="true">
            ·
          </span>
          <span className="text-white/65 tabular-nums">
            AI <span ref={aiFpsTextRef}>60</span> ({activeAiDelegate})
          </span>
          <span className="text-white/25" aria-hidden="true">
            ·
          </span>
          <span
            ref={modeTextRef}
            className="font-medium tracking-[0.05em] text-white/90 uppercase"
          >
            {settings.airDrawEnabled ? 'AIR DRAW' : settings.mode}
          </span>
        </div>
      </header>

      {/* Center-Top 1.0s Monochrome Glass Mode Banner */}
      {modeToast && (
        <div
          key={modeToast.key}
          role="status"
          aria-live="polite"
          className="mode-banner-reveal pointer-events-none fixed top-5 left-1/2 z-30 rounded-full border border-white/15 bg-neutral-950/70 px-4 py-1.5 backdrop-blur-2xl"
        >
          <span className="font-mono text-[11px] font-medium tracking-[0.14em] text-white/95 uppercase">
            {modeToast.text}
          </span>
        </div>
      )}

      {/* Observed UI Root for Air-Cursor Magnetic Hitboxes */}
      <div ref={uiRootRef} className="pointer-events-none fixed inset-0 z-20">
        {/* Right-Side Floating Inspector Panel (Never Covers Face) */}
        {toolsOpen && (
          <aside
            role="dialog"
            aria-label="Studio Settings"
            data-testid="studio-inspector"
            onPointerMove={onDockActivity}
            className="inspector-panel-enter pointer-events-auto fixed right-4 bottom-20 z-30 flex max-h-[calc(100vh-110px)] w-[292px] flex-col gap-2.5 overflow-y-auto rounded-2xl border border-white/12 bg-neutral-950/65 p-3.5 shadow-[0_24px_64px_rgba(0,0,0,0.55)] backdrop-blur-2xl"
          >
            {/* Section 1: Camera Source & Max FPS Presets */}
            <section className="flex flex-col gap-2 border-b border-white/8 pb-2.5">
              <div className="flex items-center justify-between">
                <h2 className="text-[10px] font-medium tracking-[0.14em] text-white/45 uppercase">
                  Camera Source
                </h2>
                <span className="font-mono text-[10px] text-white/35">60 FPS Ready</span>
              </div>

              <div className="relative">
                <select
                  aria-label="Camera"
                  value={selectedCameraId}
                  onChange={(e) => onSelectCamera(e.target.value)}
                  className="h-8 w-full appearance-none rounded-lg border border-white/10 bg-white/[0.06] pr-7 pl-2.5 text-xs text-white/90 transition-colors hover:bg-white/[0.1] focus:border-white/30 focus:outline-none"
                >
                  {cameras.length === 0 ? (
                    <option value="" className="bg-neutral-900 text-white">
                      Default Camera
                    </option>
                  ) : (
                    cameras.map((cam) => (
                      <option
                        key={cam.deviceId}
                        value={cam.deviceId}
                        className="bg-neutral-900 text-white"
                      >
                        {cam.label}
                      </option>
                    ))
                  )}
                </select>
                <ChevronDown
                  className="pointer-events-none absolute top-1/2 right-2.5 h-3.5 w-3.5 -translate-y-1/2 text-white/50"
                  aria-hidden="true"
                />
              </div>

              <div
                role="group"
                aria-label="Camera FPS Preset"
                className="grid grid-cols-2 gap-0.5 rounded-lg border border-white/8 bg-white/[0.05] p-0.5"
              >
                {CAMERA_PRESETS.map((preset) => {
                  const active = cameraPreset === preset.id;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      data-air-clickable="true"
                      aria-pressed={active}
                      onClick={() => onSelectCameraPreset(preset.id)}
                      className={`inline-flex h-6 items-center justify-center rounded-md text-[11px] font-medium transition-colors duration-200 ${
                        active
                          ? 'bg-white text-black shadow-xs'
                          : 'text-white/60 hover:text-white'
                      }`}
                    >
                      {preset.label}
                    </button>
                  );
                })}
              </div>
            </section>

            {/* Section 2: Performance & Graphics Quality */}
            <section className="flex flex-col gap-2 border-b border-white/8 pb-2.5">
              <div className="flex items-center justify-between">
                <h2 className="text-[10px] font-medium tracking-[0.14em] text-white/45 uppercase">
                  Graphics Quality
                </h2>
                <span className="font-mono text-[10px] text-white/40">
                  {settings.graphicsQuality === 'ULTRA' ? '15K · 512p' : '9K · 384p'}
                </span>
              </div>

              <div
                role="group"
                aria-label="Graphics Quality"
                className="grid grid-cols-2 gap-0.5 rounded-lg border border-white/8 bg-white/[0.05] p-0.5"
              >
                {QUALITY_MODES.map((q) => {
                  const active = settings.graphicsQuality === q.id;
                  return (
                    <button
                      key={q.id}
                      type="button"
                      data-air-clickable="true"
                      aria-pressed={active}
                      onClick={() => onSettingsChange({ graphicsQuality: q.id })}
                      className={`inline-flex h-6 items-center justify-center rounded-md text-[11px] font-medium transition-colors duration-200 ${
                        active
                          ? 'bg-white text-black shadow-xs'
                          : 'text-white/60 hover:text-white'
                      }`}
                    >
                      {q.label}
                    </button>
                  );
                })}
              </div>

              <div className="flex items-center justify-between pt-0.5">
                <h2 className="text-[10px] font-medium tracking-[0.14em] text-white/45 uppercase">
                  FPS Target
                </h2>
                <span className="font-mono text-[10px] text-white/40">
                  {settings.fpsTarget === 'MAX' ? 'Native Hz' : `${settings.fpsTarget} FPS`}
                </span>
              </div>

              <div
                role="group"
                aria-label="FPS Target"
                className="grid grid-cols-3 gap-0.5 rounded-lg border border-white/8 bg-white/[0.05] p-0.5"
              >
                {FPS_TARGETS.map((t) => {
                  const active = settings.fpsTarget === t.id;
                  return (
                    <button
                      key={String(t.id)}
                      type="button"
                      data-air-clickable="true"
                      aria-pressed={active}
                      onClick={() => onSettingsChange({ fpsTarget: t.id })}
                      className={`inline-flex h-6 items-center justify-center rounded-md text-[11px] font-medium transition-colors duration-200 ${
                        active
                          ? 'bg-white text-black shadow-xs'
                          : 'text-white/60 hover:text-white'
                      }`}
                    >
                      {t.label}
                    </button>
                  );
                })}
              </div>
            </section>

            {/* Section 3: AI Engine Delegate Selector */}
            <section className="flex flex-col gap-1.5 border-b border-white/8 pb-2.5">
              <div className="flex items-center justify-between">
                <h2 className="text-[10px] font-medium tracking-[0.14em] text-white/45 uppercase">
                  AI Engine
                </h2>
                <span className="font-mono text-[10px] text-white/40">{activeAiDelegate}</span>
              </div>

              <div
                role="group"
                aria-label="AI Engine Delegate"
                className="grid grid-cols-3 gap-0.5 rounded-lg border border-white/8 bg-white/[0.05] p-0.5"
              >
                {AI_ENGINE_MODES.map((eng) => {
                  const active = settings.aiEngineMode === eng.id;
                  return (
                    <button
                      key={eng.id}
                      type="button"
                      data-air-clickable="true"
                      aria-pressed={active}
                      onClick={() => onSettingsChange({ aiEngineMode: eng.id })}
                      className={`inline-flex h-6 items-center justify-center rounded-md text-[11px] font-medium transition-colors duration-200 ${
                        active
                          ? 'bg-white text-black shadow-xs'
                          : 'text-white/60 hover:text-white'
                      }`}
                    >
                      {eng.label}
                    </button>
                  );
                })}
              </div>
            </section>

            {/* Section 4: Studio Controls */}
            <section className="flex flex-col gap-1">
              <h2 className="mb-0.5 text-[10px] font-medium tracking-[0.14em] text-white/45 uppercase">
                Studio Controls
              </h2>
              <StudioToggleRow
                label="Trail"
                checked={settings.trailEnabled}
                onToggle={() => onSettingsChange({ trailEnabled: !settings.trailEnabled })}
              />
              <StudioToggleRow
                label="Air Draw"
                checked={settings.airDrawEnabled}
                onToggle={() => onSettingsChange({ airDrawEnabled: !settings.airDrawEnabled })}
              />
              <StudioToggleRow
                label="Cyber HUD"
                checked={settings.cyberHud}
                onToggle={() => onSettingsChange({ cyberHud: !settings.cyberHud })}
              />
              <StudioToggleRow
                label="Mirror"
                checked={settings.mirrored}
                onToggle={() => onSettingsChange({ mirrored: !settings.mirrored })}
              />
              <StudioToggleRow
                label="SFX Audio"
                checked={settings.sfxEnabled}
                onToggle={() => onSettingsChange({ sfxEnabled: !settings.sfxEnabled })}
              />
            </section>

            {/* Air Draw Palette (revealed when Air Draw is active) */}
            {settings.airDrawEnabled && (
              <section className="flex flex-col gap-2 border-t border-white/8 pt-2.5">
                <div className="flex items-center justify-between">
                  <h2 className="text-[10px] font-medium tracking-[0.14em] text-white/45 uppercase">
                    Air Draw Palette
                  </h2>
                  <button
                    type="button"
                    data-air-clickable="true"
                    onClick={onClearDrawing}
                    className="rounded px-1.5 py-0.5 text-[11px] font-medium text-white/60 transition-colors hover:text-white"
                  >
                    Clear
                  </button>
                </div>
                <div className="flex items-center gap-3 px-0.5 py-0.5">
                  {DRAW_COLOR_KEYS.map((key) => {
                    const color = AIR_DRAW_COLORS[key];
                    const selected = settings.drawColor === key;
                    return (
                      <button
                        key={key}
                        type="button"
                        data-air-clickable="true"
                        aria-label={`Color ${color.label}`}
                        aria-pressed={selected}
                        onClick={() => onSettingsChange({ drawColor: key })}
                        className={`h-5 w-5 rounded-full transition-opacity duration-150 ${
                          selected
                            ? 'ring-2 ring-white ring-offset-2 ring-offset-neutral-950'
                            : 'opacity-70 hover:opacity-100'
                        }`}
                        style={{ backgroundColor: color.hex }}
                      />
                    );
                  })}
                </div>
              </section>
            )}

            {/* Optional Note 3D Quick Clear */}
            {settings.noteOpen && (
              <section className="flex items-center justify-between border-t border-white/8 pt-2.5">
                <span className="text-[10px] font-medium tracking-[0.14em] text-white/45 uppercase">
                  Note 3D
                </span>
                <button
                  type="button"
                  data-air-clickable="true"
                  onClick={onClearNote}
                  className="rounded px-1.5 py-0.5 text-[11px] font-medium text-white/60 transition-colors hover:text-white"
                >
                  Clear
                </button>
              </section>
            )}
          </aside>
        )}

        {/* Bottom Centered VisionOS Floating Dock */}
        <nav
          ref={dockNavRef}
          aria-label="Effect Modes and Tools"
          data-testid="floating-dock"
          onPointerEnter={onDockActivity}
          onPointerMove={onDockActivity}
          style={{ transform: 'translateX(-50%) scale(1)', opacity: 1 }}
          className="pointer-events-auto fixed bottom-5 left-1/2 flex h-11 max-w-[96vw] items-center gap-1 rounded-full border border-white/12 bg-neutral-950/60 px-2 shadow-[0_16px_48px_rgba(0,0,0,0.48)] backdrop-blur-2xl transition-[opacity,transform] duration-300 ease-out"
        >
          {/* Core Mode Buttons with Smooth Sliding Active Pill */}
          <div className="relative flex items-center gap-0.5">
            <span
              aria-hidden="true"
              style={{
                transform: `translate3d(${pillRect.x}px, 0, 0)`,
                width: `${pillRect.w}px`,
                opacity: pillRect.visible ? 1 : 0,
              }}
              className="pointer-events-none absolute top-0 bottom-0 left-0 rounded-full bg-white shadow-xs transition-[transform,width,opacity] duration-250 ease-out"
            />
            {MODES.map((m, idx) => {
              const Icon = m.icon;
              const active = !settings.airDrawEnabled && settings.mode === m.id;
              return (
                <button
                  key={m.id}
                  ref={(el) => {
                    modeBtnRefs.current[idx] = el;
                  }}
                  type="button"
                  data-air-clickable="true"
                  data-dock-item={m.id}
                  aria-pressed={active}
                  title={`${m.label} (${m.keyHint})`}
                  onClick={() => onSettingsChange({ mode: m.id, airDrawEnabled: false })}
                  className={`relative z-10 inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-xs transition-colors duration-200 ${
                    active ? 'font-medium text-black' : 'text-white/65 hover:text-white'
                  }`}
                >
                  <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                  <span className="hidden lg:inline">{m.label}</span>
                </button>
              );
            })}
          </div>

          <span className="mx-1 h-5 w-px shrink-0 bg-white/15" aria-hidden="true" />

          {/* Compact Utility Icons: Note 3D & Settings */}
          <button
            type="button"
            data-air-clickable="true"
            data-dock-item="NOTE_3D"
            aria-pressed={settings.noteOpen}
            aria-label="Note 3D"
            title="Note 3D (N)"
            onClick={() => onSettingsChange({ noteOpen: !settings.noteOpen })}
            className={`inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors duration-200 ${
              settings.noteOpen
                ? 'bg-white text-black'
                : 'text-white/65 hover:bg-white/10 hover:text-white'
            }`}
          >
            <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
          </button>

          <button
            type="button"
            data-air-clickable="true"
            data-dock-item="SETTINGS"
            aria-expanded={toolsOpen}
            aria-label="Settings"
            title="Settings"
            onClick={onToggleTools}
            className={`inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors duration-200 ${
              toolsOpen
                ? 'bg-white text-black'
                : 'text-white/65 hover:bg-white/10 hover:text-white'
            }`}
          >
            <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </nav>
      </div>
    </>
  );
}
