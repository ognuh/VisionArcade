import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, Loader2, RefreshCw } from 'lucide-react';
import { AirCursor, type AirCursorHandle } from './components/AirCursor';
import { HoloText3D, type HoloText3DHandle } from './components/HoloText3D';
import { PrismBar } from './components/PrismBar';
import { OpticalPrismEngine } from './prism/prismEngine';
import {
  MODE_CYCLE_ORDER,
  type CycleModeId,
  type GestureType,
  type GpuInfo,
  type Landmark3D,
  type PrismSettings,
} from './types/vision';
import { AirSwipeDetector } from './vision/airSwipe';
import { FingerSnapDetector } from './vision/fingerSnap';
import { classifyHandGesture, GestureStabilizer } from './vision/gesture';
import { HandBlinkDetector } from './vision/handBlink';
import { HandTrackerClient, type LightingStatus } from './vision/handTrackerClient';

type EngineStatus = 'loading' | 'ready' | 'error';

interface CachedButtonRect {
  btn: HTMLButtonElement;
  left: number;
  right: number;
  top: number;
  bottom: number;
  cx: number;
  cy: number;
}

const DWELL_TRIGGER_MS = 650;
const DWELL_COOLDOWN_MS = 800;
const DOCK_AUTO_FADE_MS = 2500;
const MAGNETIC_HIT_PADDING_PX = 14;
const MAGNETIC_PULL = 0.38;
const FPS_UPDATE_INTERVAL_MS = 500;
const EMA_ALPHA = 2 / 31; // 30-frame EMA smoothing
const MODE_TOAST_MS = 900;

// Full-Screen White Flash timing: 40ms peak white + 380ms exponential decay
const FLASH_HOLD_SEC = 0.04;
const FLASH_DECAY_SEC = 0.38;
const FLASH_TOTAL_SEC = FLASH_HOLD_SEC + FLASH_DECAY_SEC;

const TARGET_HOLO_PHRASE = 'xin chào tôi là hưng';
const FIST_CLEAR_HOLD_SEC = 1.5;

const MODE_DISPLAY_NAME: Record<CycleModeId, string> = {
  GLASS: 'MODE // GLASS TRIANGLE',
  THERMAL: 'MODE // THERMAL MASK',
  PRISM: 'MODE // CHROMATIC PRISM',
  GALAXY: 'MODE // GALAXY 3D',
  LIGHTNING: 'MODE // PLASMA LIGHTNING',
  MYSTIC: 'MODE // MYSTIC SHIELD',
};

const INITIAL_SETTINGS: PrismSettings = {
  mode: 'AUTO',
  mirrored: true,
  trailEnabled: false,
  airDrawEnabled: false,
  drawColor: 'CYAN',
  hdDenoise: true,
  cyberHud: true,
  noteOpen: false,
  flashEnabled: false,
};

type VideoWithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number) => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

interface ExtendedTrackCapabilities extends MediaTrackCapabilities {
  focusMode?: string[];
  exposureMode?: string[];
}

interface ExtendedTrackSettings extends MediaTrackSettings {
  focusDistance?: number;
}

export default function App() {
  const [engineStatus, setEngineStatus] = useState<EngineStatus>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);
  const [modeToast, setModeToast] = useState<string | null>(null);
  const [toolsOpen, setToolsOpen] = useState<boolean>(false);
  const [settings, setSettings] = useState<PrismSettings>(INITIAL_SETTINGS);
  const [retryToken, setRetryToken] = useState(0);

  // Centered 3D Extruded Hologram Text state
  const [holoText, setHoloText] = useState<string>('');
  const [holoCompleted, setHoloCompleted] = useState<boolean>(false);
  const [holoSheenKey, setHoloSheenKey] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const glCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const glowCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fpsTextRef = useRef<HTMLSpanElement | null>(null);
  const aiFpsTextRef = useRef<HTMLSpanElement | null>(null);
  const modeTextRef = useRef<HTMLSpanElement | null>(null);
  const lightingBadgeRef = useRef<HTMLSpanElement | null>(null);
  const lightingTextRef = useRef<HTMLSpanElement | null>(null);
  const uiRootRef = useRef<HTMLDivElement | null>(null);
  const dockNavRef = useRef<HTMLElement | null>(null);
  const flashOverlayRef = useRef<HTMLDivElement | null>(null);
  const airCursorRef = useRef<AirCursorHandle | null>(null);
  const holoTextRef = useRef<HoloText3DHandle | null>(null);

  const settingsRef = useRef<PrismSettings>(settings);
  const toolsOpenRef = useRef<boolean>(toolsOpen);
  const engineRef = useRef<OpticalPrismEngine | null>(null);
  const stabilizersRef = useRef([new GestureStabilizer(1), new GestureStabilizer(1)]);
  const snapDetectorRef = useRef(new FingerSnapDetector());
  const swipeDetectorRef = useRef(new AirSwipeDetector());
  const blinkDetectorRef = useRef(new HandBlinkDetector());

  // Cached Magnetic DOMRects (+14px padding) synced automatically by ResizeObserver & MutationObserver
  const cachedButtonsRef = useRef<CachedButtonRect[]>([]);

  const lastActiveModeRef = useRef<CycleModeId>('GLASS');
  const lastDockActivityRef = useRef<number>(performance.now());
  const dockAwakeRef = useRef<boolean>(true);
  const dockNearCursorRef = useRef<boolean>(false);
  const toastTimerRef = useRef<number | null>(null);
  const whiteFlashTimeRef = useRef<number>(-1);
  const holoCursorIdxRef = useRef<number>(0);
  const holoStepToggleRef = useRef<boolean>(false);

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    toolsOpenRef.current = toolsOpen;
  }, [toolsOpen]);

  /** Recomputes cached +14px magnetic button DOMRects (zero layout thrashing inside rAF). */
  const refreshCachedUiRects = useCallback(() => {
    const root = uiRootRef.current ?? document;
    const buttons = root.querySelectorAll<HTMLButtonElement>('[data-air-clickable="true"]');
    const next: CachedButtonRect[] = [];
    for (let i = 0; i < buttons.length; i += 1) {
      const btn = buttons[i];
      if (btn.disabled) continue;
      const r = btn.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      next.push({
        btn,
        left: r.left - MAGNETIC_HIT_PADDING_PX,
        right: r.right + MAGNETIC_HIT_PADDING_PX,
        top: r.top - MAGNETIC_HIT_PADDING_PX,
        bottom: r.bottom + MAGNETIC_HIT_PADDING_PX,
        cx: r.left + r.width * 0.5,
        cy: r.top + r.height * 0.5,
      });
    }
    cachedButtonsRef.current = next;
  }, []);

  // Attach ResizeObserver & MutationObserver to UI container so button rects stay 100% synced
  useEffect(() => {
    const root = uiRootRef.current;
    if (!root) return;
    let rafHandle = 0;
    const scheduleSync = () => {
      if (rafHandle) cancelAnimationFrame(rafHandle);
      rafHandle = requestAnimationFrame(refreshCachedUiRects);
    };

    scheduleSync();
    const ro = new ResizeObserver(scheduleSync);
    ro.observe(root);
    const mo = new MutationObserver(scheduleSync);
    mo.observe(root, { childList: true, subtree: true, attributes: true });

    return () => {
      if (rafHandle) cancelAnimationFrame(rafHandle);
      ro.disconnect();
      mo.disconnect();
    };
  }, [refreshCachedUiRects]);

  const applyDockStyle = useCallback((awake: boolean, boosted: boolean) => {
    const nav = dockNavRef.current;
    if (!nav) return;
    nav.style.opacity = awake ? '1' : '0.25';
    nav.style.transform = boosted
      ? 'translateX(-50%) scale(1.03)'
      : 'translateX(-50%) scale(1)';
  }, []);

  const markDockActivity = useCallback(
    (boost = true) => {
      lastDockActivityRef.current = performance.now();
      if (!dockAwakeRef.current || dockNearCursorRef.current !== boost) {
        dockAwakeRef.current = true;
        dockNearCursorRef.current = boost;
        applyDockStyle(true, boost);
      }
    },
    [applyDockStyle],
  );

  const showModeToast = useCallback((mode: CycleModeId) => {
    setModeToast(MODE_DISPLAY_NAME[mode]);
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = window.setTimeout(() => {
      setModeToast(null);
      toastTimerRef.current = null;
    }, MODE_TOAST_MS);
  }, []);

  const handleSettingsChange = useCallback(
    (patch: Partial<PrismSettings>) => {
      markDockActivity(true);
      setSettings((prev) => {
        if (patch.mirrored !== undefined && patch.mirrored !== prev.mirrored) {
          engineRef.current?.resetSmoothing();
        }
        if (patch.mode && patch.mode !== 'AUTO' && patch.mode !== prev.mode) {
          showModeToast(patch.mode);
        }
        return { ...prev, ...patch };
      });
    },
    [markDockActivity, showModeToast],
  );

  const handleToggleTools = useCallback(() => {
    markDockActivity(true);
    setToolsOpen((v) => !v);
  }, [markDockActivity]);

  const handleStepMode = useCallback(
    (dir: -1 | 1) => {
      engineRef.current?.spawnSwipeSwoosh(dir);
      setSettings((prev) => {
        const currentEffective: CycleModeId =
          prev.mode === 'AUTO' ? lastActiveModeRef.current : prev.mode;
        const idx = MODE_CYCLE_ORDER.indexOf(currentEffective);
        const len = MODE_CYCLE_ORDER.length;
        const nextMode = MODE_CYCLE_ORDER[(idx + dir + len) % len];
        showModeToast(nextMode);
        return { ...prev, mode: nextMode, airDrawEnabled: false };
      });
    },
    [showModeToast],
  );

  const handleTriggerWhiteFlash = useCallback(() => {
    whiteFlashTimeRef.current = 0;
    if (flashOverlayRef.current) {
      flashOverlayRef.current.style.opacity = '1';
    }
  }, []);

  const handleClearDrawing = useCallback(() => {
    markDockActivity(true);
    engineRef.current?.clearDrawing();
  }, [markDockActivity]);

  const handleClearNote = useCallback(() => {
    holoCursorIdxRef.current = 0;
    setHoloText('');
    setHoloCompleted(false);
  }, []);

  const advanceHoloTyping = useCallback(() => {
    if (holoCursorIdxRef.current >= TARGET_HOLO_PHRASE.length) {
      setHoloSheenKey((k) => k + 1);
      return;
    }
    const step = holoStepToggleRef.current ? 2 : 3;
    holoStepToggleRef.current = !holoStepToggleRef.current;
    const nextIdx = Math.min(TARGET_HOLO_PHRASE.length, holoCursorIdxRef.current + step);
    holoCursorIdxRef.current = nextIdx;
    setHoloText(TARGET_HOLO_PHRASE.slice(0, nextIdx));

    if (nextIdx >= TARGET_HOLO_PHRASE.length) {
      setHoloCompleted(true);
      setHoloSheenKey((k) => k + 1);
    }
  }, []);

  useEffect(() => {
    const onPointerMove = (e: PointerEvent) => {
      if (e.clientY >= window.innerHeight * 0.75) {
        markDockActivity(true);
      }
    };
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    return () => window.removeEventListener('pointermove', onPointerMove);
  }, [markDockActivity]);

  /** Lock both WebGL2 and Canvas 2D strictly to dpr = 1 for steady 144Hz fill-rate. */
  const syncCanvases = useCallback(() => {
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);

    const glCanvas = glCanvasRef.current;
    if (glCanvas && (glCanvas.width !== w || glCanvas.height !== h)) {
      glCanvas.width = w;
      glCanvas.height = h;
    }

    const glowCanvas = glowCanvasRef.current;
    if (glowCanvas && (glowCanvas.width !== w || glowCanvas.height !== h)) {
      glowCanvas.width = w;
      glowCanvas.height = h;
    }

    refreshCachedUiRects();
  }, [refreshCachedUiRects]);

  useEffect(() => {
    syncCanvases();
    window.addEventListener('resize', syncCanvases);
    return () => window.removeEventListener('resize', syncCanvases);
  }, [syncCanvases]);

  useEffect(() => {
    let cancelled = false;
    let rafId = 0;
    let rvfcId = 0;
    let focusLockTimerId: number | null = null;
    let mediaStream: MediaStream | null = null;
    let prismEngine: OpticalPrismEngine | null = null;
    let trackerClient: HandTrackerClient | null = null;
    let hoveredBtn: HTMLButtonElement | null = null;

    setEngineStatus('loading');
    setErrorMessage(null);
    stabilizersRef.current.forEach((s) => s.reset());
    snapDetectorRef.current.reset();
    swipeDetectorRef.current.reset();
    blinkDetectorRef.current.reset();

    const clearHoverAttr = () => {
      if (hoveredBtn) {
        hoveredBtn.removeAttribute('data-air-hover');
        hoveredBtn = null;
      }
    };

    const updateLightingDom = (status: LightingStatus) => {
      const badge = lightingBadgeRef.current;
      const text = lightingTextRef.current;
      if (!badge || !text) return;
      if (status === 'LOW_LIGHT_BOOST') {
        badge.style.color = '#fde047';
        text.textContent = 'LOW-LIGHT BOOST ON';
      } else if (status === 'HIGHLIGHT_TAME') {
        badge.style.color = '#fb923c';
        text.textContent = 'HIGHLIGHT TAME';
      } else {
        badge.style.color = '#4ade80';
        text.textContent = 'LIGHT OK';
      }
    };

    async function boot() {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Trình duyệt không hỗ trợ mở webcam (getUserMedia).');
        }
        const glCanvas = glCanvasRef.current;
        if (!glCanvas) throw new Error('Không tìm thấy khung dựng WebGL2.');

        // Shared latest AI state updated asynchronously by the Web Worker (zero main-thread stall!)
        let latestHands: Landmark3D[][] = [];
        let latestGestures: GestureType[] = ['IDLE', 'IDLE'];
        let lastLighting: LightingStatus | null = null;
        let lastAiFpsUpdateMs = 0;

        trackerClient = new HandTrackerClient();

        const onWorkerResult = (packet: {
          landmarks: Landmark3D[][];
          lighting: LightingStatus;
          avgLuma: number;
          aiFps: number;
        }) => {
          if (cancelled) return;
          const now = performance.now();
          latestHands = packet.landmarks;

          const stab0 = stabilizersRef.current[0];
          const stab1 = stabilizersRef.current[1];
          latestGestures = [
            stab0.update(classifyHandGesture(latestHands[0], stab0.current)),
            stab1.update(classifyHandGesture(latestHands[1], stab1.current)),
          ];

          if (now - lastAiFpsUpdateMs >= FPS_UPDATE_INTERVAL_MS) {
            if (aiFpsTextRef.current) {
              aiFpsTextRef.current.textContent = String(packet.aiFps).padStart(2, '0');
            }
            lastAiFpsUpdateMs = now;
          }

          if (packet.lighting !== lastLighting) {
            lastLighting = packet.lighting;
            updateLightingDom(packet.lighting);
          }

          const curSettings = settingsRef.current;

          // Standalone Full-Screen White Flash on Finger Snap ONLY when Flash button is ON
          if (curSettings.flashEnabled) {
            const snaps = snapDetectorRef.current.update(latestHands, now);
            if (snaps.length > 0) {
              handleTriggerWhiteFlash();
            }
          }

          // Air Swipe 2.0 (280ms sliding window, non-fist hand, not in Air Draw)
          if (!curSettings.airDrawEnabled) {
            const swipeDir = swipeDetectorRef.current.update(
              latestHands,
              latestGestures,
              now,
              curSettings.mirrored,
            );
            if (swipeDir !== 0) {
              handleStepMode(swipeDir);
            }
          }

          // Centered 3D Hologram Text rapid hand-blink typing ("xin chào tôi là hưng")
          if (curSettings.noteOpen && latestHands.length > 0) {
            const blinkEvents = blinkDetectorRef.current.update(latestHands, latestGestures, now);
            if (blinkEvents.length > 0) {
              const viewW = glCanvasRef.current?.width || window.innerWidth;
              const viewH = glCanvasRef.current?.height || window.innerHeight;
              const vidW = videoRef.current?.videoWidth || 1280;
              const vidH = videoRef.current?.videoHeight || 720;
              for (let i = 0; i < blinkEvents.length; i += 1) {
                advanceHoloTyping();
                engineRef.current?.spawnNoteSpark(
                  blinkEvents[i].normX,
                  blinkEvents[i].normY,
                  viewW * 0.5,
                  viewH * 0.5,
                  viewW,
                  viewH,
                  vidW,
                  vidH,
                  curSettings.mirrored,
                );
              }
            }
          }
        };

        const [stream, detectedGpu] = await Promise.all([
          navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: 'user',
              width: { ideal: 1280 },
              height: { ideal: 720 },
              frameRate: { ideal: 30, max: 30 },
            },
            audio: false,
          }),
          trackerClient.init(onWorkerResult),
        ]);

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          trackerClient.dispose();
          return;
        }

        syncCanvases();
        prismEngine = new OpticalPrismEngine(glCanvas);
        engineRef.current = prismEngine;
        mediaStream = stream;
        setGpuInfo(detectedGpu);

        const video = videoRef.current as VideoWithRvfc | null;
        if (!video) return;
        video.srcObject = stream;
        await video.play();
        if (cancelled) return;

        // Lock Auto-Focus after 1.5s initial settle to stop continuous focus hunting in low light
        const videoTrack = stream.getVideoTracks()[0];
        if (videoTrack && typeof videoTrack.getCapabilities === 'function') {
          focusLockTimerId = window.setTimeout(() => {
            if (cancelled || videoTrack.readyState !== 'live') return;
            try {
              const caps = videoTrack.getCapabilities() as ExtendedTrackCapabilities;
              const focusModes = caps.focusMode;
              if (Array.isArray(focusModes) && focusModes.length > 0) {
                const advanced: Record<string, unknown>[] = [];
                if (focusModes.includes('single-shot')) {
                  advanced.push({ focusMode: 'single-shot' });
                } else if (focusModes.includes('manual')) {
                  const curSettings =
                    typeof videoTrack.getSettings === 'function'
                      ? (videoTrack.getSettings() as ExtendedTrackSettings)
                      : {};
                  if (typeof curSettings.focusDistance === 'number') {
                    advanced.push({
                      focusMode: 'manual',
                      focusDistance: curSettings.focusDistance,
                    });
                  } else {
                    advanced.push({ focusMode: 'manual' });
                  }
                }
                if (advanced.length > 0) {
                  void videoTrack.applyConstraints({
                    advanced: advanced as MediaTrackConstraintSet[],
                  }).catch(() => {});
                }
              }
            } catch {
              // Camera driver does not expose UVC focus lock
            }
          }, 1500);
        }

        // Upload initial frame immediately so WebGL2 never starts blank
        prismEngine.uploadVideoFrame(video);

        syncCanvases();
        setEngineStatus('ready');

        const activeEngine = prismEngine;
        const activeTracker = trackerClient;
        const overlayCtx = glowCanvasRef.current?.getContext('2d', { desynchronized: true }) ?? null;

        // Decoupled Camera Feed: requestVideoFrameCallback uploads texture & feeds Worker ONLY on fresh frames
        let usesRvfc = false;
        let lastFallbackVideoTime = -1;

        if (typeof video.requestVideoFrameCallback === 'function') {
          usesRvfc = true;
          const onVideoFrame = (now: number) => {
            if (cancelled) return;
            activeEngine.uploadVideoFrame(video);
            activeTracker.submitFrame(video, now);
            rvfcId = video.requestVideoFrameCallback!(onVideoFrame);
          };
          rvfcId = video.requestVideoFrameCallback(onVideoFrame);
        }

        let lastFrameTime = performance.now();
        let emaDt = 1 / 144;
        let lastFpsDisplayMs = lastFrameTime;

        // VisionOS Magnetic Air-Cursor dwell-to-click state (0.65s dwell, 800ms cooldown)
        let dwellMs = 0;
        let dwellCooldownUntil = 0;
        let clickPulse = 0;
        let lastIsHoveringUi = false;
        let lastDisplayedStatus = '';

        // 3D Hologram Text smooth parallax & 1.5s Fist-to-Clear timer
        let curTiltX = 0;
        let curTiltY = 0;
        let fistHoldSec = 0;

        // Main Thread 120–144Hz Pure Render Loop (Zero AI Blocking, Zero setState, Zero Layout Thrashing)
        const renderLoop = (now: number) => {
          if (cancelled) return;
          rafId = requestAnimationFrame(renderLoop);

          const dtSeconds = Math.min(Math.max((now - lastFrameTime) / 1000, 0.001), 0.1);
          lastFrameTime = now;

          emaDt = emaDt * (1 - EMA_ALPHA) + dtSeconds * EMA_ALPHA;
          if (now - lastFpsDisplayMs >= FPS_UPDATE_INTERVAL_MS) {
            if (fpsTextRef.current) {
              fpsTextRef.current.textContent = String(
                Math.round(1 / Math.max(emaDt, 0.001)),
              ).padStart(2, '0');
            }
            lastFpsDisplayMs = now;
          }

          // Fallback video submission if browser lacks requestVideoFrameCallback
          if (!usesRvfc && video.readyState >= 2 && video.currentTime !== lastFallbackVideoTime) {
            lastFallbackVideoTime = video.currentTime;
            activeEngine.uploadVideoFrame(video);
            activeTracker.submitFrame(video, now);
          }

          // Update Full-Screen White Flash (40ms peak #FFFFFF -> 380ms exponential decay)
          if (whiteFlashTimeRef.current >= 0) {
            whiteFlashTimeRef.current += dtSeconds;
            const ft = whiteFlashTimeRef.current;
            const el = flashOverlayRef.current;
            if (ft >= FLASH_TOTAL_SEC) {
              whiteFlashTimeRef.current = -1;
              if (el) el.style.opacity = '0';
            } else if (ft < FLASH_HOLD_SEC) {
              if (el) el.style.opacity = '1';
            } else {
              const decayT = (ft - FLASH_HOLD_SEC) / FLASH_DECAY_SEC;
              const op = Math.exp(-5.5 * decayT) * (1 - decayT);
              if (el) el.style.opacity = op.toFixed(4);
            }
          }

          const glCanvasEl = glCanvasRef.current;
          if (!glCanvasEl) return;

          const curSettings = settingsRef.current;
          const viewW = glCanvasEl.width;
          const viewH = glCanvasEl.height;
          const vidW = video.videoWidth || 1280;
          const vidH = video.videoHeight || 720;

          const frameOut = activeEngine.renderFrame(
            dtSeconds,
            viewW,
            viewH,
            vidW,
            vidH,
            latestHands,
            latestGestures,
            curSettings,
            lastIsHoveringUi,
            overlayCtx,
          );

          lastActiveModeRef.current = frameOut.activeMode;
          const statusLabel = curSettings.airDrawEnabled ? 'AIR DRAW' : frameOut.activeMode;
          if (statusLabel !== lastDisplayedStatus && modeTextRef.current) {
            modeTextRef.current.textContent = statusLabel;
            lastDisplayedStatus = statusLabel;
          }

          // 3D Hologram Text ±15 deg parallax + 1.5s Fist-to-Clear
          if (curSettings.noteOpen) {
            let targetX = 0;
            let targetY = 0;
            if (frameOut.palmScreenNorm) {
              targetY = (frameOut.palmScreenNorm.x - 0.5) * 30;
              targetX = (0.5 - frameOut.palmScreenNorm.y) * 30;
            }
            const smooth = 1 - Math.exp(-9.5 * dtSeconds);
            curTiltX += (targetX - curTiltX) * smooth;
            curTiltY += (targetY - curTiltY) * smooth;
            holoTextRef.current?.setTilt(curTiltX, curTiltY);

            if (frameOut.isLeadFist && holoCursorIdxRef.current > 0) {
              fistHoldSec += dtSeconds;
              if (fistHoldSec >= FIST_CLEAR_HOLD_SEC) {
                fistHoldSec = 0;
                handleClearNote();
              }
            } else {
              fistHoldSec = 0;
            }
          } else {
            fistHoldSec = 0;
          }

          // VisionOS Magnetic Air-Cursor (+14px hitbox, 38% magnetic pull, 0.65s dwell, 800ms cooldown)
          if (clickPulse > 0) {
            clickPulse = Math.max(0, clickPulse - dtSeconds * 3.8);
          }

          const tip = frameOut.indexTipPx;
          if (tip) {
            // Auto-Wake Dock whenever index finger enters bottom 25% of viewport
            const nearBottomZone = tip.y >= viewH * 0.75;
            if (nearBottomZone) {
              markDockActivity(true);
            }

            let hitRect: CachedButtonRect | null = null;
            const cached = cachedButtonsRef.current;
            for (let i = 0; i < cached.length; i += 1) {
              const r = cached[i];
              if (tip.x >= r.left && tip.x <= r.right && tip.y >= r.top && tip.y <= r.bottom) {
                hitRect = r;
                break;
              }
            }

            const btn = hitRect ? hitRect.btn : null;
            lastIsHoveringUi = Boolean(btn);

            if (btn !== hoveredBtn) {
              clearHoverAttr();
              if (btn) {
                btn.setAttribute('data-air-hover', 'true');
                hoveredBtn = btn;
                markDockActivity(true);
              }
              dwellMs = 0;
            }

            // Apply gentle magnetic damping toward button center while inside +14px hitbox
            const displayPos = hitRect
              ? {
                  x: tip.x + (hitRect.cx - tip.x) * MAGNETIC_PULL,
                  y: tip.y + (hitRect.cy - tip.y) * MAGNETIC_PULL,
                }
              : tip;

            let progress = 0;
            if (hoveredBtn && now >= dwellCooldownUntil) {
              dwellMs += dtSeconds * 1000;
              progress = Math.min(1, dwellMs / DWELL_TRIGGER_MS);
              if (dwellMs >= DWELL_TRIGGER_MS) {
                hoveredBtn.click();
                dwellCooldownUntil = now + DWELL_COOLDOWN_MS;
                dwellMs = 0;
                clickPulse = 1;
                progress = 0;
              }
            } else {
              dwellMs = 0;
            }

            airCursorRef.current?.update(displayPos, progress, Boolean(hoveredBtn), clickPulse);
          } else {
            lastIsHoveringUi = false;
            clearHoverAttr();
            dwellMs = 0;
            airCursorRef.current?.update(null, 0, false, 0);
          }

          // Imperative VisionOS Auto-Fade after 2.5s inactivity
          const shouldBeAwake =
            toolsOpenRef.current || now - lastDockActivityRef.current <= DOCK_AUTO_FADE_MS;
          if (shouldBeAwake !== dockAwakeRef.current) {
            dockAwakeRef.current = shouldBeAwake;
            dockNearCursorRef.current = false;
            applyDockStyle(shouldBeAwake, false);
          }
        };

        rafId = requestAnimationFrame(renderLoop);
      } catch (err) {
        if (cancelled) return;
        setErrorMessage(
          err instanceof Error ? err.message : 'Không thể mở webcam hoặc khởi tạo WebGL2.',
        );
        setEngineStatus('error');
      }
    }

    boot();

    return () => {
      cancelled = true;
      clearHoverAttr();
      if (focusLockTimerId !== null) {
        window.clearTimeout(focusLockTimerId);
        focusLockTimerId = null;
      }
      if (toastTimerRef.current !== null) {
        window.clearTimeout(toastTimerRef.current);
        toastTimerRef.current = null;
      }
      if (rafId) cancelAnimationFrame(rafId);
      const vid = videoRef.current as VideoWithRvfc | null;
      if (rvfcId && vid?.cancelVideoFrameCallback) {
        vid.cancelVideoFrameCallback(rvfcId);
      }
      if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
      if (trackerClient) {
        trackerClient.dispose();
        trackerClient = null;
      }
      if (prismEngine) {
        prismEngine.dispose();
        if (engineRef.current === prismEngine) {
          engineRef.current = null;
        }
      }
    };
  }, [
    advanceHoloTyping,
    applyDockStyle,
    handleClearNote,
    handleStepMode,
    handleTriggerWhiteFlash,
    markDockActivity,
    retryToken,
    syncCanvases,
  ]);

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-black">
      {/* Raw video hidden cleanly (opacity-0 pointer-events-none) so WebGL2 never overlaps with raw video */}
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={`pointer-events-none fixed inset-0 h-full w-full object-cover opacity-0 ${
          settings.mirrored ? 'scale-x-[-1]' : ''
        }`}
      />

      {/* Layer 1 (Zero-Copy, dpr=1): WebGL2 GPU HD+ Denoise / CAS + Optical Polygons */}
      <canvas
        ref={glCanvasRef}
        className="pointer-events-none fixed inset-0 z-10 h-full w-full"
      />

      {/* Layer 2 (Zero-Copy, dpr=1): Transparent Canvas 2D for src/fx/ + Cyber HUD + Air Draw */}
      <canvas
        ref={glowCanvasRef}
        className="pointer-events-none fixed inset-0 z-10 h-full w-full"
      />

      {/* Centered 3D Extruded Hologram Text ("xin chào tôi là hưng") — zero box/card */}
      {settings.noteOpen && (
        <HoloText3D
          ref={holoTextRef}
          text={holoText}
          completed={holoCompleted}
          sheenKey={holoSheenKey}
        />
      )}

      {/* Standalone Full-Screen White Flash Layer (40ms peak #FFFFFF -> 380ms exponential decay) */}
      <div
        ref={flashOverlayRef}
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 z-30 bg-white opacity-0 will-change-[opacity]"
      />

      <PrismBar
        settings={settings}
        onSettingsChange={handleSettingsChange}
        modeToast={modeToast}
        gpuInfo={gpuInfo}
        fpsTextRef={fpsTextRef}
        aiFpsTextRef={aiFpsTextRef}
        modeTextRef={modeTextRef}
        lightingBadgeRef={lightingBadgeRef}
        lightingTextRef={lightingTextRef}
        uiRootRef={uiRootRef}
        dockNavRef={dockNavRef}
        toolsOpen={toolsOpen}
        onToggleTools={handleToggleTools}
        onDockActivity={() => markDockActivity(true)}
        onStepMode={handleStepMode}
        onTriggerWhiteFlash={handleTriggerWhiteFlash}
        onClearDrawing={handleClearDrawing}
        onClearNote={handleClearNote}
      />

      <AirCursor ref={airCursorRef} />

      {engineStatus === 'loading' && (
        <div
          role="status"
          aria-live="polite"
          className="fixed inset-0 z-40 flex items-center justify-center bg-black/90 p-4"
        >
          <div className="flex max-w-sm flex-col items-center gap-3 rounded-xl border border-white/12 bg-[#0d1017] px-7 py-6 text-center">
            <Loader2 className="h-6 w-6 animate-spin text-[#38bdf8]" aria-hidden="true" />
            <h1 className="text-sm font-semibold text-white">
              Đang khởi chạy VisionArcade
            </h1>
            <p className="text-xs text-white/65">
              Đang nạp MediaPipe HandLandmarker trên Web Worker và bộ lọc HD+ GPU.
            </p>
          </div>
        </div>
      )}

      {engineStatus === 'error' && (
        <div
          role="alert"
          className="fixed inset-0 z-40 flex items-center justify-center bg-black/92 p-4"
        >
          <div className="flex max-w-md flex-col items-center gap-3.5 rounded-xl border border-white/12 bg-[#0d1017] px-7 py-6 text-center">
            <Camera className="h-6 w-6 text-[#38bdf8]" aria-hidden="true" />
            <h1 className="text-sm font-semibold text-white">Không thể khởi chạy camera</h1>
            <p className="text-xs text-white/65">
              {errorMessage ?? 'Hãy cấp quyền webcam cho trình duyệt rồi thử lại.'}
            </p>
            <button
              type="button"
              onClick={() => setRetryToken((t) => t + 1)}
              className="mt-1 inline-flex h-10 items-center gap-2 rounded-full bg-white px-5 text-xs font-semibold text-black transition-colors hover:bg-white/90"
            >
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Thử lại</span>
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
