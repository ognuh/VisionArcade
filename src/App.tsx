import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react';
import { Camera, Crosshair, Loader2, Play, RefreshCw, RotateCcw } from 'lucide-react';
import { AirCursor, type AirCursorHandle } from './components/AirCursor';
import { HoloText3D, type HoloText3DHandle } from './components/HoloText3D';
import { PrismBar } from './components/PrismBar';
import type { RunnerHudSnapshot } from './fx/runnerGame';
import { OpticalPrismEngine } from './prism/prismEngine';
import {
  MODE_CYCLE_ORDER,
  type AiEngineMode,
  type CameraDeviceItem,
  type CameraPresetId,
  type CycleModeId,
  type FpsTarget,
  type GestureType,
  type GraphicsQuality,
  type Landmark3D,
  type PrismSettings,
  type TrackerTaskMode,
} from './types/vision';
import { classifyHandGesture, GestureStabilizer } from './vision/gesture';
import { GestureManager, type SwipeIndicatorState } from './vision/gestureManager';
import { HandBlinkDetector } from './vision/handBlink';
import { HandTrackerClient, type TrackingResultPacket } from './vision/handTrackerClient';
import {
  createMockCameraStream,
  generateMockHands,
  generateMockPose,
  type MockPoseId,
} from './vision/mockHands';

interface VisionTestBridge {
  setPose: (pose: MockPoseId) => void;
  setMode: (mode: PrismSettings['mode']) => void;
  setQuality: (quality: GraphicsQuality) => void;
  setFpsTarget: (fpsTarget: FpsTarget) => void;
  setNoteOpen: (open: boolean) => void;
  setToolsOpen: (open: boolean) => void;
  setAirDraw: (enabled: boolean) => void;
  startRunner: (skipCountdown?: boolean) => void;
  recalibrateRunner: () => void;
  getRunnerSnapshot: () => RunnerHudSnapshot | null;
  getFluidParticleCount: () => number;
  prewarmCosmos: () => void;
  resetPerf: () => void;
  getPerf: () => {
    avgFps: number;
    maxRenderMs: number;
    avgRenderMs: number;
    samples: number;
  };
  getRippleMeta: () => {
    isHalfFloat: boolean;
    fboComplete: boolean;
    simWidth: number;
    simHeight: number;
  } | null;
  runLiveWorkerTest: (
    engineMode?: AiEngineMode,
    frames?: number,
    taskMode?: TrackerTaskMode,
  ) => Promise<{
    isWorker: boolean;
    taskMode: TrackerTaskMode;
    completedFrames: number;
    steadyAvgInferMs: number;
    steadyMaxInferMs: number;
    aiFps: number;
    delegate: 'GPU' | 'CPU';
  }>;
  getHoveredUiItem: () => string | null;
  getState: () => {
    mode: PrismSettings['mode'];
    activeMode: CycleModeId;
    pose: MockPoseId;
    noteOpen: boolean;
    toolsOpen: boolean;
    airDrawEnabled: boolean;
    graphicsQuality: GraphicsQuality;
    fpsTarget: FpsTarget;
    holoText: string;
    engineStatus: EngineStatus;
  };
}

declare global {
  interface Window {
    __VISION_TEST__?: VisionTestBridge;
  }
}

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

const CAMERA_PRESET_STORAGE_KEY = 'visionarcade.cameraPreset';
const SFX_STORAGE_KEY = 'visionarcade.sfxEnabled';
const AI_ENGINE_STORAGE_KEY = 'visionarcade.aiEngineMode';
const QUALITY_STORAGE_KEY = 'visionarcade.graphicsQuality';
const FPS_TARGET_STORAGE_KEY = 'visionarcade.fpsTarget';
const LEGACY_STORAGE_KEYS = [
  'visionarcade.arcadeHighScore',
  'visionarcade.faceMaskMode',
  'visionarcade.faceMaskStrength',
  'visionarcade.fpsCap',
];

const DWELL_TRIGGER_MS = 650;
const DWELL_COOLDOWN_MS = 800;
const DOCK_AUTO_FADE_MS = 2500;
const MAGNETIC_HIT_PADDING_PX = 14;
const MAGNETIC_PULL = 0.38;
const FPS_UPDATE_INTERVAL_MS = 450;
const EMA_ALPHA = 2 / 41;
const MODE_TOAST_MS = 1000;

const TARGET_HOLO_PHRASE = 'xin chào tôi là hưng';
const FIST_CLEAR_HOLD_SEC = 1.5;

const MODE_DISPLAY_NAME: Record<CycleModeId, string> = {
  GLASS: 'GLASS',
  THERMAL: 'THERMAL',
  PRISM: 'PRISM',
  COSMOS: 'COSMOS',
  LIGHTNING: 'LIGHTNING',
  MYSTIC: 'MYSTIC',
  RIPPLE: 'RIPPLE',
  FLUID: 'FLUID',
  RUNNER: 'RUNNER',
};

const EMPTY_RUNNER_SNAP: RunnerHudSnapshot = {
  phase: 'READY',
  score: 0,
  coins: 0,
  bestScore: 0,
  speedMultiplier: 1.0,
  countdownNum: 0,
  lane: 0,
  actionLabel: 'READY',
  hasBody: false,
  isCalibrated: false,
  handsUpProgress: 0,
  distance: 0,
};

function purgeLegacyStorage(): void {
  try {
    for (let i = 0; i < LEGACY_STORAGE_KEYS.length; i += 1) {
      window.localStorage.removeItem(LEGACY_STORAGE_KEYS[i]);
    }
  } catch {
    // ignore storage errors
  }
}

function loadSavedCameraPreset(): CameraPresetId {
  try {
    const v = window.localStorage.getItem(CAMERA_PRESET_STORAGE_KEY);
    if (v === 'AUTO' || v === '720P60' || v === '720P30' || v === '1080P60') return v;
    if (v === '720P') return '720P60';
    if (v === '1080P') return '1080P60';
  } catch {
    // ignore
  }
  return 'AUTO';
}

function saveCameraPreset(preset: CameraPresetId): void {
  try {
    window.localStorage.setItem(CAMERA_PRESET_STORAGE_KEY, preset);
  } catch {
    // ignore
  }
}

function loadSavedSfxEnabled(): boolean {
  try {
    const v = window.localStorage.getItem(SFX_STORAGE_KEY);
    if (v === '0') return false;
  } catch {
    // ignore
  }
  return true;
}

function saveSfxEnabled(enabled: boolean): void {
  try {
    window.localStorage.setItem(SFX_STORAGE_KEY, enabled ? '1' : '0');
  } catch {
    // ignore
  }
}

function loadSavedAiEngineMode(): AiEngineMode {
  try {
    const v = window.localStorage.getItem(AI_ENGINE_STORAGE_KEY);
    if (v === 'GPU' || v === 'AUTO' || v === 'CPU') return v;
  } catch {
    // ignore
  }
  return 'GPU';
}

function saveAiEngineMode(mode: AiEngineMode): void {
  try {
    window.localStorage.setItem(AI_ENGINE_STORAGE_KEY, mode);
  } catch {
    // ignore
  }
}

function loadSavedQuality(): GraphicsQuality {
  try {
    const v = window.localStorage.getItem(QUALITY_STORAGE_KEY);
    if (v === 'ULTRA' || v === 'HIGH') return v;
  } catch {
    // ignore
  }
  return 'ULTRA';
}

function saveQuality(q: GraphicsQuality): void {
  try {
    window.localStorage.setItem(QUALITY_STORAGE_KEY, q);
  } catch {
    // ignore
  }
}

function loadSavedFpsTarget(): FpsTarget {
  try {
    const v = window.localStorage.getItem(FPS_TARGET_STORAGE_KEY);
    if (v === 'MAX') return 'MAX';
    if (v === '120') return 120;
    if (v === '60') return 60;
  } catch {
    // ignore
  }
  return 'MAX';
}

function saveFpsTarget(target: FpsTarget): void {
  try {
    window.localStorage.setItem(FPS_TARGET_STORAGE_KEY, String(target));
  } catch {
    // ignore
  }
}

function createInitialSettings(): PrismSettings {
  return {
    mode: 'AUTO',
    mirrored: true,
    trailEnabled: false,
    airDrawEnabled: false,
    drawColor: 'CYAN',
    cyberHud: true,
    noteOpen: false,
    sfxEnabled: loadSavedSfxEnabled(),
    aiEngineMode: loadSavedAiEngineMode(),
    graphicsQuality: loadSavedQuality(),
    fpsTarget: loadSavedFpsTarget(),
  };
}

type VideoWithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number) => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

interface ExtendedTrackCapabilities extends MediaTrackCapabilities {
  focusMode?: string[];
}

interface ExtendedTrackSettings extends MediaTrackSettings {
  focusDistance?: number;
}

/**
 * 60 FPS priority webcam constraints:
 * - Requests ideal 60 FPS with min 25 FPS so 60fps sources (USB webcam / Phone / OBS) lock true 60 FPS,
 *   while 30fps laptop webcams smoothly lock 30 FPS and never drop to 10fps Windows Media Foundation profiles.
 */
function getVideoConstraintsForPreset(preset: CameraPresetId): MediaTrackConstraints {
  switch (preset) {
    case '720P60':
      return { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, min: 25 } };
    case '720P30':
      return { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, min: 25 } };
    case '1080P60':
      return { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 60, min: 25 } };
    case 'AUTO':
    default:
      return { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, min: 25 } };
  }
}

async function findObsCameraDeviceId(): Promise<string | undefined> {
  if (!navigator.mediaDevices?.enumerateDevices) return undefined;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const obs = devices.find(
      (d) => d.kind === 'videoinput' && d.deviceId && /obs/i.test(d.label || ''),
    );
    return obs?.deviceId || undefined;
  } catch {
    return undefined;
  }
}

async function openCameraStream(
  preferredDeviceId?: string,
  preset: CameraPresetId = 'AUTO',
): Promise<MediaStream> {
  const baseVideo = getVideoConstraintsForPreset(preset);
  const deviceFilter: MediaTrackConstraints = preferredDeviceId
    ? { deviceId: { exact: preferredDeviceId } }
    : { facingMode: 'user' };

  const attempts: MediaTrackConstraints[] = [
    { ...baseVideo, ...deviceFilter },
    { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, min: 25 }, ...deviceFilter },
    { width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: 30, min: 25 }, ...deviceFilter },
    { width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: 30 }, ...deviceFilter },
  ];

  if (preferredDeviceId) {
    // Allow OBS Virtual Camera to open with its native output profile before falling back to default cam
    attempts.push(
      { deviceId: { exact: preferredDeviceId } },
      { ...baseVideo, facingMode: 'user' },
      { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, min: 25 }, facingMode: 'user' },
      { width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: 30, min: 25 }, facingMode: 'user' },
    );
  }

  let lastErr: unknown;
  for (let i = 0; i < attempts.length; i += 1) {
    try {
      return await navigator.mediaDevices.getUserMedia({
        video: attempts[i],
        audio: false,
      });
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error('Unable to open camera stream.');
}

/** Defaults to OBS Virtual Camera whenever available (handles both pre-granted and first-time permission). */
async function openDefaultObsCameraStream(preset: CameraPresetId): Promise<MediaStream> {
  const preObsId = await findObsCameraDeviceId();
  const initialStream = await openCameraStream(preObsId, preset);
  if (preObsId) return initialStream;

  // On first visit labels were empty until permission was granted; re-check for OBS now
  const postObsId = await findObsCameraDeviceId();
  const activeId = initialStream.getVideoTracks()[0]?.getSettings?.().deviceId;
  if (postObsId && postObsId !== activeId) {
    try {
      const obsStream = await openCameraStream(postObsId, preset);
      initialStream.getTracks().forEach((t) => t.stop());
      return obsStream;
    } catch {
      return initialStream;
    }
  }
  return initialStream;
}

function scheduleSafeFocusLock(
  stream: MediaStream,
  isCancelled: () => boolean,
): number | null {
  try {
    const videoTrack = stream.getVideoTracks()[0];
    if (!videoTrack || typeof videoTrack.getCapabilities !== 'function') return null;

    return window.setTimeout(() => {
      if (isCancelled() || videoTrack.readyState !== 'live') return;
      try {
        const caps = videoTrack.getCapabilities() as ExtendedTrackCapabilities;
        const focusModes = caps?.focusMode;
        if (!Array.isArray(focusModes) || focusModes.length === 0) return;

        const advanced: Record<string, unknown>[] = [];
        if (focusModes.includes('single-shot')) {
          advanced.push({ focusMode: 'single-shot' });
        } else if (focusModes.includes('manual')) {
          const curSettings =
            typeof videoTrack.getSettings === 'function'
              ? (videoTrack.getSettings() as ExtendedTrackSettings)
              : {};
          if (typeof curSettings.focusDistance === 'number') {
            advanced.push({ focusMode: 'manual', focusDistance: curSettings.focusDistance });
          } else {
            advanced.push({ focusMode: 'manual' });
          }
        }
        if (advanced.length > 0) {
          void videoTrack
            .applyConstraints({ advanced: advanced as MediaTrackConstraintSet[] })
            .catch(() => {});
        }
      } catch {
        // ignore UVC focus unsupported
      }
    }, 1500);
  } catch {
    return null;
  }
}

export default function App() {
  const [engineStatus, setEngineStatus] = useState<EngineStatus>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [modeToast, setModeToast] = useState<{ text: string; key: number } | null>(null);
  const [toolsOpen, setToolsOpen] = useState<boolean>(false);
  const [settings, setSettings] = useState<PrismSettings>(createInitialSettings);
  const [cameras, setCameras] = useState<CameraDeviceItem[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [cameraPreset, setCameraPreset] = useState<CameraPresetId>(loadSavedCameraPreset);
  const [activeAiDelegate, setActiveAiDelegate] = useState<'GPU' | 'CPU'>('GPU');
  const [retryToken, setRetryToken] = useState(0);
  const [runnerHud, setRunnerHud] = useState<RunnerHudSnapshot>(EMPTY_RUNNER_SNAP);

  // Centered 3D Extruded Hologram Text state
  const [holoText, setHoloText] = useState<string>('');
  const [holoCompleted, setHoloCompleted] = useState<boolean>(false);
  const [holoSheenKey, setHoloSheenKey] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const glCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cosmosCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const glowCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const pipCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fpsTextRef = useRef<HTMLSpanElement | null>(null);
  const camFpsTextRef = useRef<HTMLSpanElement | null>(null);
  const aiFpsTextRef = useRef<HTMLSpanElement | null>(null);
  const modeTextRef = useRef<HTMLSpanElement | null>(null);
  const uiRootRef = useRef<HTMLDivElement | null>(null);
  const dockNavRef = useRef<HTMLElement | null>(null);
  const airCursorRef = useRef<AirCursorHandle | null>(null);
  const holoTextRef = useRef<HoloText3DHandle | null>(null);

  const settingsRef = useRef<PrismSettings>(settings);
  const toolsOpenRef = useRef<boolean>(toolsOpen);
  const engineRef = useRef<OpticalPrismEngine | null>(null);
  const trackerClientRef = useRef<HandTrackerClient | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const focusTimerRef = useRef<number | null>(null);
  const stabilizersRef = useRef([new GestureStabilizer(1), new GestureStabilizer(1)]);
  const gestureManagerRef = useRef(new GestureManager());
  const swipeIndicatorRef = useRef<SwipeIndicatorState>({
    active: false,
    dir: 0,
    progress: 0,
    normX: 0.5,
    normY: 0.5,
  });
  const hoveringUiRef = useRef<boolean>(false);
  const hoveredUiItemRef = useRef<string | null>(null);
  const blinkDetectorRef = useRef(new HandBlinkDetector());
  const cachedButtonsRef = useRef<CachedButtonRect[]>([]);

  const lastActiveModeRef = useRef<CycleModeId>('GLASS');
  const lastDockActivityRef = useRef<number>(performance.now());
  const dockAwakeRef = useRef<boolean>(true);
  const dockNearCursorRef = useRef<boolean>(false);
  const toastTimerRef = useRef<number | null>(null);
  const toastSeqRef = useRef<number>(0);
  const holoCursorIdxRef = useRef<number>(0);
  const holoStepToggleRef = useRef<boolean>(false);
  const mockPoseRef = useRef<MockPoseId>('QUAD_TWO_HANDS');
  const mockPoseStartRef = useRef<number>(performance.now());
  const engineStatusRef = useRef<EngineStatus>('loading');
  const holoTextValueRef = useRef<string>('');

  useEffect(() => {
    purgeLegacyStorage();
  }, []);

  useEffect(() => {
    engineStatusRef.current = engineStatus;
  }, [engineStatus]);

  useEffect(() => {
    holoTextValueRef.current = holoText;
  }, [holoText]);

  useEffect(() => {
    settingsRef.current = settings;
    const targetTask: TrackerTaskMode =
      settings.mode === 'RUNNER' && !settings.airDrawEnabled ? 'POSE' : 'HAND';
    trackerClientRef.current?.setTaskMode(targetTask);
    engineRef.current?.setQuality(settings.graphicsQuality);
  }, [settings]);

  useEffect(() => {
    toolsOpenRef.current = toolsOpen;
  }, [toolsOpen]);

  const refreshCameraList = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices
        .filter((d) => d.kind === 'videoinput')
        .map((d, idx) => ({
          deviceId: d.deviceId,
          label: d.label?.trim() || `Camera ${idx + 1}`,
        }));
      setCameras(videoDevices);
    } catch {
      // ignore enumeration errors
    }
  }, []);

  useEffect(() => {
    const md = navigator.mediaDevices;
    if (!md?.addEventListener) return;
    const onChange = () => void refreshCameraList();
    md.addEventListener('devicechange', onChange);
    return () => md.removeEventListener('devicechange', onChange);
  }, [refreshCameraList]);

  const applyCameraConfig = useCallback(
    async (deviceId: string, preset: CameraPresetId) => {
      const video = videoRef.current;
      if (!video) return;

      try {
        if (focusTimerRef.current !== null) {
          window.clearTimeout(focusTimerRef.current);
          focusTimerRef.current = null;
        }
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((t) => t.stop());
          mediaStreamRef.current = null;
        }

        const newStream = await openCameraStream(deviceId || undefined, preset);
        mediaStreamRef.current = newStream;
        video.srcObject = newStream;
        await video.play();

        engineRef.current?.resetSmoothing();
        engineRef.current?.uploadVideoFrame(video);

        const activeTrackId = newStream.getVideoTracks()[0]?.getSettings?.().deviceId;
        if (activeTrackId) {
          setSelectedCameraId(activeTrackId);
        }

        focusTimerRef.current = scheduleSafeFocusLock(newStream, () => false);
        void refreshCameraList();
      } catch (err) {
        console.warn('Unable to switch camera configuration:', err);
      }
    },
    [refreshCameraList],
  );

  const handleSelectCamera = useCallback(
    (deviceId: string) => {
      if (!deviceId) return;
      setSelectedCameraId(deviceId);
      void applyCameraConfig(deviceId, cameraPreset);
    },
    [applyCameraConfig, cameraPreset],
  );

  const handleSelectCameraPreset = useCallback(
    (preset: CameraPresetId) => {
      saveCameraPreset(preset);
      setCameraPreset(preset);
      void applyCameraConfig(selectedCameraId, preset);
    },
    [applyCameraConfig, selectedCameraId],
  );

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
    nav.style.opacity = awake ? '1' : '0.32';
    nav.style.transform = boosted
      ? 'translateX(-50%) scale(1.02)'
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

  const showModeToast = useCallback((label: string) => {
    toastSeqRef.current += 1;
    setModeToast({ text: label, key: toastSeqRef.current });
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
        if (patch.sfxEnabled !== undefined && patch.sfxEnabled !== prev.sfxEnabled) {
          saveSfxEnabled(patch.sfxEnabled);
        }
        if (patch.aiEngineMode !== undefined && patch.aiEngineMode !== prev.aiEngineMode) {
          saveAiEngineMode(patch.aiEngineMode);
          trackerClientRef.current?.setEngineMode(patch.aiEngineMode);
        }
        if (patch.graphicsQuality !== undefined && patch.graphicsQuality !== prev.graphicsQuality) {
          saveQuality(patch.graphicsQuality);
          engineRef.current?.setQuality(patch.graphicsQuality);
        }
        if (patch.fpsTarget !== undefined && patch.fpsTarget !== prev.fpsTarget) {
          saveFpsTarget(patch.fpsTarget);
        }
        if (patch.mode && patch.mode !== 'AUTO' && patch.mode !== prev.mode) {
          showModeToast(MODE_DISPLAY_NAME[patch.mode]);
          if (patch.mode === 'RUNNER') {
            engineRef.current?.resetRunner();
          }
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
        showModeToast(MODE_DISPLAY_NAME[nextMode]);
        if (nextMode === 'RUNNER') {
          engineRef.current?.resetRunner();
        }
        return { ...prev, mode: nextMode, airDrawEnabled: false };
      });
    },
    [showModeToast],
  );

  const handleClearDrawing = useCallback(() => {
    markDockActivity(true);
    engineRef.current?.clearDrawing();
  }, [markDockActivity]);

  const handleClearNote = useCallback(() => {
    holoCursorIdxRef.current = 0;
    setHoloText('');
    setHoloCompleted(false);
  }, []);

  const handleDropGlb = useCallback(
    (e: DragEvent<HTMLElement>) => {
      e.preventDefault();
      const file = e.dataTransfer?.files?.[0];
      if (!file || !file.name.toLowerCase().endsWith('.glb')) return;
      void engineRef.current
        ?.loadCustomGlbFromFile(file)
        .then(() => {
          setSettings((prev) => ({ ...prev, mode: 'COSMOS', airDrawEnabled: false }));
          showModeToast(`GLB · ${file.name.replace(/\.glb$/i, '')}`);
        })
        .catch(() => {
          showModeToast('INVALID GLB');
        });
    },
    [showModeToast],
  );

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
    let mockIntervalId = 0;
    let stopMockStream: (() => void) | null = null;
    let prismEngine: OpticalPrismEngine | null = null;
    let trackerClient: HandTrackerClient | null = null;
    let hoveredBtn: HTMLButtonElement | null = null;

    const searchParams = new URLSearchParams(window.location.search);
    const host = window.location.hostname;
    const isLocalHost = host === 'localhost' || host === '127.0.0.1' || host === '[::1]';
    const isTestMode = isLocalHost && searchParams.get('test') === '1';
    const VALID_MOCK_POSES: ReadonlySet<string> = new Set<MockPoseId>([
      'NONE',
      'ONE_FINGER',
      'TWO_FINGERS',
      'OPEN_ONE',
      'FIST_ONE',
      'OPEN_TWO_HANDS',
      'QUAD_TWO_HANDS',
      'AIR_SWIPE_RIGHT',
      'AIR_SWIPE_LEFT',
      'AIR_SWIPE_RETURN',
      'DOCK_HOVER_CENTER',
      'RIPPLE_WAVE_BURST',
      'FLUID_INTERACT',
      'HAND_BLINK',
      'FIST_BOTH',
      'RUNNER_STAND',
      'RUNNER_HANDS_UP',
      'RUNNER_LEFT',
      'RUNNER_RIGHT',
      'RUNNER_JUMP',
      'RUNNER_CROUCH',
      'RUNNER_DEMO_SEQ',
    ]);
    const requestedPose = searchParams.get('pose');
    const initialTestPose: MockPoseId =
      requestedPose && VALID_MOCK_POSES.has(requestedPose)
        ? (requestedPose as MockPoseId)
        : 'QUAD_TWO_HANDS';
    if (isTestMode) {
      mockPoseRef.current = initialTestPose;
      mockPoseStartRef.current = performance.now();
    }

    setEngineStatus('loading');
    setErrorMessage(null);
    stabilizersRef.current.forEach((s) => s.reset());
    gestureManagerRef.current.reset();
    blinkDetectorRef.current.reset();

    const clearHoverAttr = () => {
      if (hoveredBtn) {
        hoveredBtn.removeAttribute('data-air-hover');
        hoveredBtn = null;
      }
      hoveredUiItemRef.current = null;
    };

    async function boot() {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Camera access (getUserMedia) is not supported in this browser.');
        }
        const glCanvas = glCanvasRef.current;
        if (!glCanvas) throw new Error('WebGL2 viewport failed to mount.');

        let latestHands: Landmark3D[][] = [];
        let latestPose: Landmark3D[] | null = null;
        let latestGestures: GestureType[] = ['IDLE', 'IDLE'];
        let handPacketSeq = 0;
        let lastAiFpsUpdateMs = 0;

        trackerClient = new HandTrackerClient();
        trackerClientRef.current = trackerClient;

        const onWorkerResult = (packet: TrackingResultPacket) => {
          if (cancelled) return;
          const now = performance.now();
          if (packet.taskMode === 'POSE') {
            latestPose = packet.poseLandmarks;
            latestHands = [];
          } else {
            latestHands = packet.landmarks;
            latestPose = null;
          }
          handPacketSeq += 1;

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

          const curSettings = settingsRef.current;

          const gestureDecision = gestureManagerRef.current.update({
            hands: latestHands,
            gestures: latestGestures,
            nowMs: now,
            settings: curSettings,
            activeMode: lastActiveModeRef.current,
            isHoveringUi: hoveringUiRef.current,
            cosmosBlackHoleBusy: engineRef.current?.isCosmosBlackHoleBusy() ?? false,
          });
          swipeIndicatorRef.current = gestureDecision.swipeIndicator;
          if (gestureDecision.swipeDir !== 0) {
            handleStepMode(gestureDecision.swipeDir);
          }

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

        const savedPreset = loadSavedCameraPreset();
        const initialTask: TrackerTaskMode =
          settingsRef.current.mode === 'RUNNER' ? 'POSE' : 'HAND';
        let stream: MediaStream;

        let perfSamples = 0;
        let perfFpsSum = 0;
        let perfRenderSumMs = 0;
        let perfMaxRenderMs = 0;

        if (isTestMode) {
          const mock = createMockCameraStream();
          stream = mock.stream;
          stopMockStream = mock.stop;
          window.__VISION_TEST__ = {
            setPose: (pose: MockPoseId) => {
              mockPoseRef.current = pose;
              mockPoseStartRef.current = performance.now();
            },
            setMode: (mode: PrismSettings['mode']) => {
              setSettings((prev) => ({
                ...prev,
                mode,
                airDrawEnabled: false,
              }));
            },
            setQuality: (graphicsQuality: GraphicsQuality) => {
              setSettings((prev) => ({ ...prev, graphicsQuality }));
            },
            setFpsTarget: (fpsTarget: FpsTarget) => {
              setSettings((prev) => ({ ...prev, fpsTarget }));
            },
            setNoteOpen: (open: boolean) => {
              setSettings((prev) => ({ ...prev, noteOpen: open }));
            },
            setToolsOpen: (open: boolean) => {
              setToolsOpen(open);
            },
            setAirDraw: (airDrawEnabled: boolean) => {
              setSettings((prev) => ({ ...prev, airDrawEnabled }));
            },
            startRunner: (skipCountdown = false) => {
              engineRef.current?.startRunner(skipCountdown);
            },
            recalibrateRunner: () => {
              engineRef.current?.recalibrateRunner();
            },
            getRunnerSnapshot: () => engineRef.current?.getRunnerSnapshot() ?? null,
            getFluidParticleCount: () => engineRef.current?.getFluidParticleCount() ?? 0,
            prewarmCosmos: () => {
              engineRef.current?.prewarmCosmosNow();
            },
            resetPerf: () => {
              perfSamples = 0;
              perfFpsSum = 0;
              perfRenderSumMs = 0;
              perfMaxRenderMs = 0;
            },
            getPerf: () => ({
              avgFps: perfSamples > 0 ? Math.round((perfFpsSum / perfSamples) * 10) / 10 : 0,
              maxRenderMs: Math.round(perfMaxRenderMs * 100) / 100,
              avgRenderMs:
                perfSamples > 0 ? Math.round((perfRenderSumMs / perfSamples) * 100) / 100 : 0,
              samples: perfSamples,
            }),
            getRippleMeta: () => engineRef.current?.getRippleMeta() ?? null,
            runLiveWorkerTest: async (
              testEngineMode: AiEngineMode = 'GPU',
              targetFrames = 24,
              taskMode: TrackerTaskMode = 'HAND',
            ) => {
              const vid = videoRef.current;
              if (!vid) throw new Error('Video element missing');
              const benchClient = new HandTrackerClient();
              let count = 0;
              let steadyCount = 0;
              let steadySumMs = 0;
              let steadyMaxMs = 0;
              let lastFps = 0;
              let lastDelegate: 'GPU' | 'CPU' = 'GPU';
              let timerId = 0;

              return new Promise((resolve, reject) => {
                const finish = () => {
                  window.clearTimeout(timeoutId);
                  window.clearInterval(timerId);
                  const meta = benchClient.getTelemetry();
                  benchClient.dispose();
                  resolve({
                    isWorker: meta.isWorker,
                    taskMode: meta.taskMode,
                    completedFrames: count,
                    steadyAvgInferMs:
                      steadyCount > 0 ? Math.round((steadySumMs / steadyCount) * 10) / 10 : 0,
                    steadyMaxInferMs: Math.round(steadyMaxMs * 10) / 10,
                    aiFps: lastFps,
                    delegate: lastDelegate,
                  });
                };

                const timeoutId = window.setTimeout(finish, 14000);

                benchClient
                  .init(
                    (pkt) => {
                      count += 1;
                      if (count > 3 && pkt.inferMs > 0) {
                        steadyCount += 1;
                        steadySumMs += pkt.inferMs;
                        if (pkt.inferMs > steadyMaxMs) steadyMaxMs = pkt.inferMs;
                      }
                      lastFps = pkt.aiFps;
                      lastDelegate = pkt.delegate;
                      if (count >= targetFrames) {
                        finish();
                      }
                    },
                    testEngineMode,
                    undefined,
                    taskMode,
                  )
                  .then(() => {
                    timerId = window.setInterval(() => {
                      benchClient.submitFrame(vid, performance.now());
                    }, 16);
                  })
                  .catch((err) => {
                    window.clearTimeout(timeoutId);
                    benchClient.dispose();
                    reject(err);
                  });
              });
            },
            getHoveredUiItem: () => hoveredUiItemRef.current,
            getState: () => ({
              mode: settingsRef.current.mode,
              activeMode: lastActiveModeRef.current,
              pose: mockPoseRef.current,
              noteOpen: settingsRef.current.noteOpen,
              toolsOpen: toolsOpenRef.current,
              airDrawEnabled: settingsRef.current.airDrawEnabled,
              graphicsQuality: settingsRef.current.graphicsQuality,
              fpsTarget: settingsRef.current.fpsTarget,
              holoText: holoTextValueRef.current,
              engineStatus: engineStatusRef.current,
            }),
          };
        } else {
          const [camStream] = await Promise.all([
            openDefaultObsCameraStream(savedPreset),
            trackerClient.init(
              onWorkerResult,
              settingsRef.current.aiEngineMode,
              (delegate) => {
                if (!cancelled) setActiveAiDelegate(delegate);
              },
              initialTask,
            ),
          ]);
          stream = camStream;
        }

        if (cancelled) {
          if (stopMockStream) stopMockStream();
          else stream.getTracks().forEach((t) => t.stop());
          trackerClient.dispose();
          return;
        }

        syncCanvases();
        prismEngine = new OpticalPrismEngine(glCanvas, cosmosCanvasRef.current);
        prismEngine.setQuality(settingsRef.current.graphicsQuality);
        engineRef.current = prismEngine;
        mediaStreamRef.current = stream;

        if (!isTestMode) {
          const activeDeviceId = stream.getVideoTracks()[0]?.getSettings?.().deviceId;
          if (activeDeviceId) {
            setSelectedCameraId(activeDeviceId);
          }
          void refreshCameraList();
        }

        const video = videoRef.current as VideoWithRvfc | null;
        if (!video) return;
        video.srcObject = stream;
        await video.play();
        if (cancelled) return;

        if (!isTestMode) {
          focusTimerRef.current = scheduleSafeFocusLock(stream, () => cancelled);
        }
        prismEngine.uploadVideoFrame(video);

        // Glass Preloader compiles all 3D + Bloom + Dissolve + Runner shaders before reveal (0ms switch hitch)
        prismEngine.prewarmCosmosNow();

        syncCanvases();
        setEngineStatus('ready');

        if (isTestMode) {
          // 60Hz synthetic mock stream (both Hand & Full-Body Runner Pose)
          mockIntervalId = window.setInterval(() => {
            if (cancelled) return;
            const elapsedSec = (performance.now() - mockPoseStartRef.current) / 1000;
            const curTask: TrackerTaskMode =
              settingsRef.current.mode === 'RUNNER' && !settingsRef.current.airDrawEnabled
                ? 'POSE'
                : 'HAND';
            if (curTask === 'POSE') {
              onWorkerResult({
                taskMode: 'POSE',
                landmarks: [],
                poseLandmarks: generateMockPose(mockPoseRef.current, elapsedSec),
                aiFps: 60,
                inferMs: 5,
                delegate: 'GPU',
              });
            } else {
              onWorkerResult({
                taskMode: 'HAND',
                landmarks: generateMockHands(mockPoseRef.current, elapsedSec),
                poseLandmarks: null,
                aiFps: 60,
                inferMs: 5,
                delegate: 'GPU',
              });
            }
          }, 16);
        }

        const activeEngine = prismEngine;
        const activeTracker = trackerClient;
        const overlayCtx = glowCanvasRef.current?.getContext('2d', { desynchronized: true }) ?? null;

        let usesRvfc = false;
        let lastFallbackVideoTime = -1;
        let lastCamFrameMs = 0;
        let camEmaDt = 1 / 60;
        let lastCamDisplayMs = 0;

        const recordCamCadence = (now: number) => {
          if (lastCamFrameMs > 0) {
            const cdt = Math.min(0.25, Math.max(0.004, (now - lastCamFrameMs) / 1000));
            camEmaDt = camEmaDt * 0.85 + cdt * 0.15;
          }
          lastCamFrameMs = now;
          if (now - lastCamDisplayMs >= FPS_UPDATE_INTERVAL_MS && camFpsTextRef.current) {
            camFpsTextRef.current.textContent = String(
              Math.min(120, Math.round(1 / Math.max(camEmaDt, 0.001))),
            ).padStart(2, '0');
            lastCamDisplayMs = now;
          }
        };

        if (typeof video.requestVideoFrameCallback === 'function') {
          usesRvfc = true;
          const onVideoFrame = (now: number) => {
            if (cancelled) return;
            recordCamCadence(now);
            activeEngine.uploadVideoFrame(video);
            if (!isTestMode) {
              activeTracker.submitFrame(video, now);
            }
            rvfcId = video.requestVideoFrameCallback!(onVideoFrame);
          };
          rvfcId = video.requestVideoFrameCallback(onVideoFrame);
        }

        let lastFrameTime = performance.now();
        let emaDt = 1 / 120;
        let lastFpsDisplayMs = lastFrameTime;
        let nextRenderTime = lastFrameTime;
        let lastRunnerUiSyncMs = 0;

        let dwellMs = 0;
        let dwellCooldownUntil = 0;
        let clickPulse = 0;
        let lastIsHoveringUi = false;
        let lastDisplayedStatus = '';

        let curTiltX = 0;
        let curTiltY = 0;
        let fistHoldSec = 0;

        // Max Refresh Rate Native Render Loop (Max Hz / 120 / 60 user selectable)
        const renderLoop = (now: number) => {
          if (cancelled) return;
          rafId = requestAnimationFrame(renderLoop);

          const curSettings = settingsRef.current;

          if (!usesRvfc && video.readyState >= 2 && video.currentTime !== lastFallbackVideoTime) {
            lastFallbackVideoTime = video.currentTime;
            recordCamCadence(now);
            activeEngine.uploadVideoFrame(video);
            if (!isTestMode) {
              activeTracker.submitFrame(video, now);
            }
          }

          if (curSettings.fpsTarget !== 'MAX') {
            const targetInterval = 1000 / curSettings.fpsTarget;
            if (now < nextRenderTime - 0.25) {
              return;
            }
            nextRenderTime += targetInterval;
            if (now - nextRenderTime > 24) {
              nextRenderTime = now + targetInterval;
            }
          } else {
            nextRenderTime = now;
          }

          const dtSeconds = Math.min(Math.max((now - lastFrameTime) / 1000, 0.001), 0.1);
          lastFrameTime = now;

          emaDt = emaDt * (1 - EMA_ALPHA) + dtSeconds * EMA_ALPHA;
          if (now - lastFpsDisplayMs >= FPS_UPDATE_INTERVAL_MS) {
            if (fpsTextRef.current) {
              fpsTextRef.current.textContent = String(
                Math.min(240, Math.round(1 / Math.max(emaDt, 0.001))),
              ).padStart(2, '0');
            }
            lastFpsDisplayMs = now;
          }

          const glCanvasEl = glCanvasRef.current;
          if (!glCanvasEl) return;

          const viewW = glCanvasEl.width;
          const viewH = glCanvasEl.height;
          const vidW = video.videoWidth || 1280;
          const vidH = video.videoHeight || 720;

          const pipCtx =
            curSettings.mode === 'RUNNER' && !curSettings.airDrawEnabled
              ? pipCanvasRef.current?.getContext('2d') ?? null
              : null;

          const renderStartMs = performance.now();
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
            swipeIndicatorRef.current,
            handPacketSeq,
            latestPose,
            pipCtx,
            video,
          );
          const renderCostMs = performance.now() - renderStartMs;
          if (isTestMode) {
            perfSamples += 1;
            perfFpsSum += 1 / Math.max(dtSeconds, 0.001);
            perfRenderSumMs += renderCostMs;
            if (renderCostMs > perfMaxRenderMs) perfMaxRenderMs = renderCostMs;
          }

          lastActiveModeRef.current = frameOut.activeMode;
          const statusLabel = curSettings.airDrawEnabled ? 'AIR DRAW' : frameOut.activeMode;
          if (statusLabel !== lastDisplayedStatus && modeTextRef.current) {
            modeTextRef.current.textContent = statusLabel;
            lastDisplayedStatus = statusLabel;
          }

          // Throttled Runner HUD state sync (~15Hz — crisp score/countdown updates with near-zero React cost)
          if (frameOut.activeMode === 'RUNNER' && now - lastRunnerUiSyncMs >= 65) {
            lastRunnerUiSyncMs = now;
            const snap = activeEngine.getRunnerSnapshot();
            if (snap) {
              setRunnerHud((prev) =>
                prev.phase === snap.phase &&
                prev.score === snap.score &&
                prev.coins === snap.coins &&
                prev.bestScore === snap.bestScore &&
                prev.speedMultiplier === snap.speedMultiplier &&
                prev.countdownNum === snap.countdownNum &&
                prev.hasBody === snap.hasBody &&
                Math.abs(prev.handsUpProgress - snap.handsUpProgress) < 0.04
                  ? prev
                  : snap,
              );
            }
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

          // VisionOS Magnetic Air-Cursor
          if (clickPulse > 0) {
            clickPulse = Math.max(0, clickPulse - dtSeconds * 3.8);
          }

          const tip = frameOut.indexTipPx;
          if (tip) {
            if (tip.y >= viewH * 0.75) {
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
            hoveringUiRef.current = lastIsHoveringUi;

            if (btn !== hoveredBtn) {
              clearHoverAttr();
              if (btn) {
                btn.setAttribute('data-air-hover', 'true');
                hoveredBtn = btn;
                hoveredUiItemRef.current =
                  btn.getAttribute('data-dock-item') ?? btn.getAttribute('aria-label') ?? btn.textContent;
                markDockActivity(true);
              }
              dwellMs = 0;
            }

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
            hoveringUiRef.current = false;
            clearHoverAttr();
            dwellMs = 0;
            airCursorRef.current?.update(null, 0, false, 0);
          }

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
          err instanceof Error ? err.message : 'Unable to start camera or WebGL2 renderer.',
        );
        setEngineStatus('error');
      }
    }

    boot();

    return () => {
      cancelled = true;
      clearHoverAttr();
      if (focusTimerRef.current !== null) {
        window.clearTimeout(focusTimerRef.current);
        focusTimerRef.current = null;
      }
      if (toastTimerRef.current !== null) {
        window.clearTimeout(toastTimerRef.current);
        toastTimerRef.current = null;
      }
      if (rafId) cancelAnimationFrame(rafId);
      if (mockIntervalId) window.clearInterval(mockIntervalId);
      if (stopMockStream) {
        stopMockStream();
        stopMockStream = null;
      }
      if (isTestMode) {
        delete window.__VISION_TEST__;
      }
      const vid = videoRef.current as VideoWithRvfc | null;
      if (rvfcId && vid?.cancelVideoFrameCallback) {
        vid.cancelVideoFrameCallback(rvfcId);
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
        mediaStreamRef.current = null;
      }
      if (trackerClient) {
        trackerClient.dispose();
        if (trackerClientRef.current === trackerClient) {
          trackerClientRef.current = null;
        }
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
    markDockActivity,
    refreshCameraList,
    retryToken,
    syncCanvases,
  ]);

  const isRunnerMode = settings.mode === 'RUNNER' && !settings.airDrawEnabled;

  return (
    <main
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDropGlb}
      className="relative h-screen w-screen overflow-hidden bg-black"
    >
      {/* Hidden raw video source */}
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={`pointer-events-none fixed inset-0 h-full w-full object-cover opacity-0 ${
          settings.mirrored ? 'scale-x-[-1]' : ''
        }`}
      />

      {/* Layer 1: WebGL2 Webcam Pass-Through (+ Ultra Liquid Glass Ripple) + 3D Crystal Lenses */}
      <canvas
        ref={glCanvasRef}
        className="pointer-events-none fixed inset-0 z-10 h-full w-full"
      />

      {/* Layer 2: Shared Three.js Stage (Cosmos + 3D Lightning + 15k Fluid + Ultra 3D Runner) */}
      <canvas
        ref={cosmosCanvasRef}
        className="pointer-events-none fixed inset-0 z-10 h-full w-full"
      />

      {/* Layer 3: Transparent Canvas 2D (Mystic 3D Runes, Cyber HUD, Speed Lines, Air Draw & Trails) */}
      <canvas
        ref={glowCanvasRef}
        className="pointer-events-none fixed inset-0 z-10 h-full w-full"
      />

      {settings.noteOpen && (
        <HoloText3D
          ref={holoTextRef}
          text={holoText}
          completed={holoCompleted}
          sheenKey={holoSheenKey}
        />
      )}

      {/* Runner Pro Studio Mono HUD (Top-Right) */}
      {isRunnerMode && (
        <div
          data-testid="runner-hud"
          className="pointer-events-auto fixed top-4 right-4 z-20 flex items-center gap-2"
        >
          <div className="inline-flex h-7 items-center gap-3 rounded-full border border-white/12 bg-neutral-950/65 px-3.5 font-mono text-[11px] text-white/90 tabular-nums backdrop-blur-2xl">
            <span>
              <span className="text-white/45">SCORE </span>
              {String(runnerHud.score).padStart(5, '0')}
            </span>
            <span className="text-white/20">·</span>
            <span className="text-amber-300">
              <span className="text-white/45">COINS </span>
              {runnerHud.coins}
            </span>
            <span className="text-white/20">·</span>
            <span className="text-cyan-300">x{runnerHud.speedMultiplier.toFixed(1)}</span>
            <span className="text-white/20">·</span>
            <span className="text-white/70">
              <span className="text-white/45">BEST </span>
              {runnerHud.bestScore}
            </span>
          </div>
          <button
            type="button"
            onClick={() => engineRef.current?.recalibrateRunner()}
            title="Recalibrate standing posture"
            className="inline-flex h-7 items-center gap-1.5 rounded-full border border-white/12 bg-neutral-950/65 px-2.5 font-mono text-[10px] font-medium text-white/80 backdrop-blur-2xl transition-colors hover:bg-white/15 hover:text-white"
          >
            <Crosshair className="h-3 w-3" aria-hidden="true" />
            <span>Recalibrate</span>
          </button>
        </div>
      )}

      {/* Runner Bottom-Right Camera PiP Body Guide (220x124px, shifts left if Inspector is open) */}
      {isRunnerMode && (
        <div
          data-testid="runner-pip"
          className={`pointer-events-none fixed bottom-20 z-20 overflow-hidden rounded-xl border border-white/15 bg-neutral-950/75 shadow-[0_12px_36px_rgba(0,0,0,0.6)] backdrop-blur-xl transition-[right] duration-250 ${
            toolsOpen ? 'right-[316px]' : 'right-4'
          }`}
          style={{ width: 220, height: 124 }}
        >
          <canvas ref={pipCanvasRef} width={220} height={124} className="block h-full w-full" />
        </div>
      )}

      {/* Runner Start / 3-2-1 Calibration / Game Over Frosted Overlay */}
      {isRunnerMode && runnerHud.phase !== 'PLAYING' && (
        <div className="pointer-events-none fixed inset-0 z-20 flex items-center justify-center p-4">
          {runnerHud.phase === 'CALIBRATING' ? (
            <div className="flex flex-col items-center gap-2 rounded-2xl border border-white/15 bg-neutral-950/70 px-8 py-6 text-center backdrop-blur-2xl">
              <span className="font-mono text-[10px] tracking-[0.2em] text-cyan-300 uppercase">
                Calibrating Shoulders · Stand Center
              </span>
              <span className="font-mono text-6xl font-semibold text-white tabular-nums">
                {runnerHud.countdownNum || 1}
              </span>
            </div>
          ) : (
            <div className="pointer-events-auto flex w-full max-w-xs flex-col items-center gap-3.5 rounded-2xl border border-white/14 bg-neutral-950/75 px-6 py-5 text-center shadow-[0_24px_64px_rgba(0,0,0,0.6)] backdrop-blur-2xl">
              <span className="font-mono text-[10px] tracking-[0.18em] text-white/50 uppercase">
                {runnerHud.phase === 'GAME_OVER' ? 'Run Complete' : 'Full-Body Neon Runner'}
              </span>
              <h2 className="text-lg font-semibold tracking-tight text-white">
                {runnerHud.phase === 'GAME_OVER'
                  ? `Score ${runnerHud.score}`
                  : 'Neon Subway Runner'}
              </h2>
              {runnerHud.phase === 'GAME_OVER' ? (
                <p className="font-mono text-xs text-white/70 tabular-nums">
                  COINS {runnerHud.coins} · BEST {runnerHud.bestScore}
                </p>
              ) : (
                <p className="text-xs leading-relaxed text-white/60">
                  Lean or step Left / Center / Right · Jump over barriers · Duck under laser gates.
                </p>
              )}

              <button
                type="button"
                data-testid="runner-start-btn"
                onClick={() => engineRef.current?.startRunner(false)}
                className="inline-flex h-9 w-full items-center justify-center gap-2 rounded-full bg-white px-5 text-xs font-medium text-black transition-colors hover:bg-white/90"
              >
                {runnerHud.phase === 'GAME_OVER' ? (
                  <>
                    <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                    <span>Play Again</span>
                  </>
                ) : (
                  <>
                    <Play className="h-3.5 w-3.5 fill-current" aria-hidden="true" />
                    <span>Start Run</span>
                  </>
                )}
              </button>

              {/* Touchless Overhead Both-Hands-Up Hold Indicator (1.2s) */}
              <div className="flex w-full flex-col gap-1.5 pt-0.5">
                <div className="flex items-center justify-between font-mono text-[10px] text-white/50">
                  <span>Or raise both hands up</span>
                  <span>{Math.round(runnerHud.handsUpProgress * 100)}%</span>
                </div>
                <div className="h-1 w-full overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-cyan-400 transition-[width] duration-75"
                    style={{ width: `${Math.round(runnerHud.handsUpProgress * 100)}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <PrismBar
        settings={settings}
        onSettingsChange={handleSettingsChange}
        modeToast={modeToast}
        cameras={cameras}
        selectedCameraId={selectedCameraId}
        cameraPreset={cameraPreset}
        activeAiDelegate={activeAiDelegate}
        onSelectCamera={handleSelectCamera}
        onSelectCameraPreset={handleSelectCameraPreset}
        fpsTextRef={fpsTextRef}
        camFpsTextRef={camFpsTextRef}
        aiFpsTextRef={aiFpsTextRef}
        modeTextRef={modeTextRef}
        uiRootRef={uiRootRef}
        dockNavRef={dockNavRef}
        toolsOpen={toolsOpen}
        onToggleTools={handleToggleTools}
        onDockActivity={() => markDockActivity(true)}
        onStepMode={handleStepMode}
        onClearDrawing={handleClearDrawing}
        onClearNote={handleClearNote}
      />

      <AirCursor ref={airCursorRef} />

      {engineStatus === 'loading' && (
        <div
          role="status"
          aria-live="polite"
          className="fixed inset-0 z-40 flex items-center justify-center bg-black/88 p-4 backdrop-blur-xl"
        >
          <div className="flex max-w-sm flex-col items-center gap-3 rounded-2xl border border-white/12 bg-neutral-950/80 px-7 py-6 text-center backdrop-blur-2xl">
            <Loader2 className="h-5 w-5 animate-spin text-white/85" aria-hidden="true" />
            <h1 className="text-sm font-medium text-white">Starting VisionArcade</h1>
            <p className="text-xs text-white/55">
              Compiling Ultra Cinema shaders, GPU Worker pipeline, and 3D stage.
            </p>
          </div>
        </div>
      )}

      {engineStatus === 'error' && (
        <div
          role="alert"
          className="fixed inset-0 z-40 flex items-center justify-center bg-black/90 p-4 backdrop-blur-xl"
        >
          <div className="flex max-w-md flex-col items-center gap-3.5 rounded-2xl border border-white/12 bg-neutral-950/80 px-7 py-6 text-center backdrop-blur-2xl">
            <Camera className="h-5 w-5 text-white/80" aria-hidden="true" />
            <h1 className="text-sm font-medium text-white">Camera Unavailable</h1>
            <p className="text-xs text-white/55">
              {errorMessage ?? 'Grant camera permission in your browser and try again.'}
            </p>
            <button
              type="button"
              onClick={() => setRetryToken((t) => t + 1)}
              className="mt-1 inline-flex h-9 items-center gap-2 rounded-full bg-white px-5 text-xs font-medium text-black transition-colors hover:bg-white/90"
            >
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Retry</span>
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
