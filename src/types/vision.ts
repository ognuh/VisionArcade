export interface Landmark3D {
  x: number;
  y: number;
  z: number;
}

export type GestureType = 'IDLE' | 'OPEN' | 'PINCH' | 'FIST' | 'POINT';

export type GpuTier = 'DISCRETE' | 'INTEGRATED' | 'SOFTWARE' | 'UNKNOWN';

export interface GpuInfo {
  delegate: 'GPU' | 'CPU';
  renderer: string;
  tier: GpuTier;
  fallbackReason?: string;
}

export type PrismModeId =
  | 'AUTO'
  | 'GLASS'
  | 'THERMAL'
  | 'PRISM'
  | 'COSMOS'
  | 'LIGHTNING'
  | 'MYSTIC';

export type CycleModeId = Exclude<PrismModeId, 'AUTO'>;

export const MODE_CYCLE_ORDER: readonly CycleModeId[] = [
  'GLASS',
  'THERMAL',
  'PRISM',
  'COSMOS',
  'LIGHTNING',
  'MYSTIC',
];

export type AirDrawColorId = 'CYAN' | 'MAGENTA' | 'GREEN' | 'GOLD';

export type CameraPresetId = 'AUTO' | '720P' | '1080P';

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
  hdDenoise: boolean;
  cyberHud: boolean;
  noteOpen: boolean;
  sfxEnabled: boolean;
}
