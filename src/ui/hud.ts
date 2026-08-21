// FPS + GPU stats overlay (the original's drawDebug/drawInfo).
export class Hud {
  private fpsEl = document.getElementById('fps')!;
  private gpuEl = document.getElementById('gpu-stats')!;
  private lastTime = performance.now();
  private smoothedFps = 0;
  private frames = 0;
  private lastUpdate = 0;

  setCounts(numParticles: number, numStrands: number): void {
    document.getElementById('num-particles')!.textContent = numParticles.toLocaleString('en-US');
    document.getElementById('num-strands')!.textContent = numStrands.toLocaleString('en-US');
  }

  // call once per frame; updates DOM ~4x/sec
  tick(gpuStats?: { simMs: number | null; renderMs: number | null }): void {
    const now = performance.now();
    const dt = now - this.lastTime;
    this.lastTime = now;
    const fps = 1000 / Math.max(dt, 1e-3);
    this.smoothedFps = this.smoothedFps === 0 ? fps : this.smoothedFps * 0.95 + fps * 0.05;
    this.frames++;

    if (now - this.lastUpdate > 250) {
      this.lastUpdate = now;
      this.fpsEl.textContent = this.smoothedFps.toFixed(1);
      if (gpuStats && (gpuStats.simMs !== null || gpuStats.renderMs !== null)) {
        const sim = gpuStats.simMs !== null ? gpuStats.simMs.toFixed(2) : '-';
        const render = gpuStats.renderMs !== null ? gpuStats.renderMs.toFixed(2) : '-';
        this.gpuEl.textContent = `gpu sim    ${sim} ms\ngpu render ${render} ms`;
      } else {
        this.gpuEl.textContent = '';
      }
    }
  }
}

export function showBanner(message?: string): void {
  const banner = document.getElementById('banner')!;
  if (message) document.getElementById('banner-message')!.textContent = message;
  banner.hidden = false;
}
