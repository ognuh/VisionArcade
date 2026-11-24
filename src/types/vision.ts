export interface Landmark3D {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export type GestureType = 'IDLE' | 'OPEN' | 'PINCH' | 'FIST' | 'POINT';

export type GpuTier = 'DISCRETE' | 'INTEGRATED' | 'SOFTWARE' | 'UNKNOWN';

export interface GpuInfo {
  delegate: 'GPU' | 'CPU';
  renderer: string;
  tier: GpuTier;
  fallbackReason?: string;
}

export type AiEngineMode = 'GPU' | 'AUTO' | 'CPU';

export type TrackerTaskMode = 'HAND' | 'POSE';

export type GraphicsQuality = 'ULTRA' | 'HIGH';

export type FpsTarget = 'MAX' | 120 | 60;

export type PrismModeId =
  | 'AUTO'
  | 'GLASS'
  | 'THERMAL'
  | 'PRISM'
  | 'COSMOS'
  | 'LIGHTNING'
  | 'MYSTIC'
  | 'RIPPLE'
  | 'FLUID'
  | 'RUNNER';

export type CycleModeId = Exclude<PrismModeId, 'AUTO'>;

export const MODE_CYCLE_ORDER: readonly CycleModeId[] = [
  'GLASS',
  'THERMAL',
  'PRISM',
  'COSMOS',
  'LIGHTNING',
  'MYSTIC',
  'RIPPLE',
  'FLUID',
  'RUNNER',
];

export type AirDrawColorId = 'CYAN' | 'MAGENTA' | 'GREEN' | 'GOLD';

export type CameraPresetId = 'AUTO' | '720P60' | '720P30' | '1080P60';

export interface CameraDeviceItem {
  deviceId: string;
  label: string;
}

export interface PrismSettings {
  mode: PrismModeId;
  mirrored: boolean;
  trailEnabled: boolean;
  airDrawEnabled: boolean;
  drawColor: AirDrawColorId;
  cyberHud: boolean;
  noteOpen: boolean;
  sfxEnabled: boolean;
  aiEngineMode: AiEngineMode;
  graphicsQuality: GraphicsQuality;
  fpsTarget: FpsTarget;
}
