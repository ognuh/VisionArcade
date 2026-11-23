import type { GestureType, Landmark3D } from '../types/vision';

function dist2D(a: Landmark3D, b: Landmark3D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function isFingerExtended(
  landmarks: Landmark3D[],
  mcpIdx: number,
  pipIdx: number,
  tipIdx: number,
): boolean {
  const wrist = landmarks[0];
  const mcp = landmarks[mcpIdx];
  const pip = landmarks[pipIdx];
  const tip = landmarks[tipIdx];

  const wristToTip = dist2D(wrist, tip);
  const wristToPip = dist2D(wrist, pip);
  const mcpToTip = dist2D(mcp, tip);
  const mcpToPip = dist2D(mcp, pip);

  // Check joint straightness via vector cosine at PIP so curled fingers don't misclassify.
  const v1x = pip.x - mcp.x;
  const v1y = pip.y - mcp.y;
  const v2x = tip.x - pip.x;
  const v2y = tip.y - pip.y;
  const len1 = Math.hypot(v1x, v1y) || 0.0001;
  const len2 = Math.hypot(v2x, v2y) || 0.0001;
  const straightness = (v1x * v2x + v1y * v2y) / (len1 * len2);

  return wristToTip > wristToPip * 1.08 && mcpToTip > mcpToPip * 1.1 && straightness > 0.35;
}

/**
 * Classifies hand gesture with Schmitt-trigger hysteresis on PINCH so dragging
 * straight lines does not drop mid-stroke when thumb/index distance fluctuates.
 */
export function classifyHandGesture(
  landmarks: Landmark3D[] | undefined,
  prevGesture: GestureType = 'IDLE',
): GestureType {
  if (!landmarks || landmarks.length < 21) {
    return 'IDLE';
  }

  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const palmScale = Math.max(dist2D(wrist, middleMcp), 0.001);

  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const pinchRatio = dist2D(thumbTip, indexTip) / palmScale;

  // Schmitt-trigger hysteresis: tighter to enter PINCH, wider to release.
  const pinchThreshold = prevGesture === 'PINCH' ? 0.44 : 0.31;
  if (pinchRatio < pinchThreshold) {
    return 'PINCH';
  }

  const indexOpen = isFingerExtended(landmarks, 5, 6, 8);
  const middleOpen = isFingerExtended(landmarks, 9, 10, 12);
  const ringOpen = isFingerExtended(landmarks, 13, 14, 16);
  const pinkyOpen = isFingerExtended(landmarks, 17, 18, 20);

  const openCount =
    (indexOpen ? 1 : 0) +
    (middleOpen ? 1 : 0) +
    (ringOpen ? 1 : 0) +
    (pinkyOpen ? 1 : 0);

  if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
    return 'POINT';
  }

  if (openCount === 0) {
    return 'FIST';
  }

  if (openCount >= 3) {
    return 'OPEN';
  }

  return indexOpen && middleOpen ? 'OPEN' : 'FIST';
}

export class GestureStabilizer {
  private stableGesture: GestureType = 'IDLE';
  private candidateGesture: GestureType = 'IDLE';
  private candidateFrames = 0;
  private readonly requiredFrames: number;

  constructor(requiredFrames = 2) {
    this.requiredFrames = requiredFrames;
  }

  public get current(): GestureType {
    return this.stableGesture;
  }

  public update(rawGesture: GestureType): GestureType {
    if (rawGesture === this.stableGesture) {
      this.candidateGesture = rawGesture;
      this.candidateFrames = 0;
      return this.stableGesture;
    }

    if (rawGesture === this.candidateGesture) {
      this.candidateFrames += 1;
      if (this.candidateFrames >= this.requiredFrames) {
        this.stableGesture = rawGesture;
        this.candidateFrames = 0;
      }
    } else {
      this.candidateGesture = rawGesture;
      this.candidateFrames = 1;
    }

    return this.stableGesture;
  }

  public reset(): void {
    this.stableGesture = 'IDLE';
    this.candidateGesture = 'IDLE';
    this.candidateFrames = 0;
  }
}
