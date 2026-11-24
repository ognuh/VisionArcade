/**
 * Procedural Web Audio API SFX Engine for Plasma Lightning & Thunder Discharge:
 * - Zero external audio files; 100% synthesized in real time.
 * - High-voltage electric crackle impulses modulated by live arc intensity.
 * - Deep resonant sub-thunder boom on Fist-to-Open Thunder Discharge.
 * - Safe against browser autoplay restrictions (lazy unlock, zero console warnings).
 */
export class LightningAudioEngine {
  private ctx: AudioContext | null = null;
  private noiseBuffer: AudioBuffer | null = null;
  private enabled = false;
  private nextCrackleMs = 0;

  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (enabled) {
      this.ensureContext();
    }
  }

  private ensureContext(): AudioContext | null {
    if (!this.enabled) return null;
    if (this.ctx) {
      if (this.ctx.state === 'suspended') {
        void this.ctx.resume().catch(() => {});
      }
      return this.ctx.state === 'running' ? this.ctx : null;
    }

    try {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return null;
      const ctx = new AudioCtx();
      this.ctx = ctx;

      // Pre-generate 1.2s white noise buffer once (Zero-GC during playback)
      const len = Math.floor(ctx.sampleRate * 1.2);
      const buf = ctx.createBuffer(1, len, ctx.sampleRate);
      const data = buf.getChannelData(0);
      for (let i = 0; i < len; i += 1) {
        data[i] = Math.random() * 2 - 1;
      }
      this.noiseBuffer = buf;

      if (ctx.state === 'suspended') {
        void ctx.resume().catch(() => {});
      }
      return ctx.state === 'running' ? ctx : null;
    } catch {
      return null;
    }
  }

  /** Emits subtle high-voltage electric crackle snaps scaled by arc intensity (0..1). */
  public tickCrackle(intensity: number, nowMs: number): void {
    if (!this.enabled || intensity < 0.15 || nowMs < this.nextCrackleMs) return;
    const ctx = this.ensureContext();
    if (!ctx || !this.noiseBuffer) return;

    const interval = Math.max(42, 165 - intensity * 110) + Math.random() * 65;
    this.nextCrackleMs = nowMs + interval;

    try {
      const now = ctx.currentTime;
      const src = ctx.createBufferSource();
      src.buffer = this.noiseBuffer;

      const bandpass = ctx.createBiquadFilter();
      bandpass.type = 'bandpass';
      bandpass.frequency.setValueAtTime(2600 + Math.random() * 3400, now);
      bandpass.Q.setValueAtTime(4.5, now);

      const gain = ctx.createGain();
      const peak = Math.min(0.11, 0.025 + intensity * 0.065) * (0.6 + Math.random() * 0.4);
      const dur = 0.018 + Math.random() * 0.028;
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(peak, now + 0.003);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + dur);

      src.connect(bandpass);
      bandpass.connect(gain);
      gain.connect(ctx.destination);

      const offset = Math.random() * 0.9;
      src.start(now, offset, dur + 0.01);
    } catch {
      // ignore transient audio graph errors
    }
  }

  /** Triggers a soft deep sub-thunder boom + high-voltage snap on Fist-to-Open Thunder Discharge. */
  public playThunderDischarge(power = 1): void {
    if (!this.enabled) return;
    const ctx = this.ensureContext();
    if (!ctx || !this.noiseBuffer) return;

    try {
      const now = ctx.currentTime;
      const p = Math.max(0.4, Math.min(1.3, power));

      // 1. Sub-bass punch (115Hz -> 34Hz pitch drop)
      const osc = ctx.createOscillator();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(115, now);
      osc.frequency.exponentialRampToValueAtTime(34, now + 0.36);

      const subGain = ctx.createGain();
      subGain.gain.setValueAtTime(0.0001, now);
      subGain.gain.linearRampToValueAtTime(0.22 * p, now + 0.015);
      subGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.48);

      osc.connect(subGain);
      subGain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.5);

      // 2. Filtered thunder crack & roll
      const noise = ctx.createBufferSource();
      noise.buffer = this.noiseBuffer;

      const lpf = ctx.createBiquadFilter();
      lpf.type = 'lowpass';
      lpf.frequency.setValueAtTime(2200, now);
      lpf.frequency.exponentialRampToValueAtTime(160, now + 0.55);

      const nGain = ctx.createGain();
      nGain.gain.setValueAtTime(0.0001, now);
      nGain.gain.linearRampToValueAtTime(0.18 * p, now + 0.012);
      nGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.62);

      noise.connect(lpf);
      lpf.connect(nGain);
      nGain.connect(ctx.destination);
      noise.start(now, 0, 0.65);
    } catch {
      // ignore audio errors
    }
  }

  public dispose(): void {
    if (this.ctx) {
      void this.ctx.close().catch(() => {});
      this.ctx = null;
      this.noiseBuffer = null;
    }
  }
}
