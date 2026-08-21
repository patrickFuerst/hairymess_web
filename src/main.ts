// HairyMess — WebGPU port. Boot, frame loop and UI wiring.

import { MAX_DT, NUM_HAIR_PARTICLES, defaultParams, type ColorMode } from './params';
import { OrbitCamera } from './engine/camera';
import { densityCompensatedRepulsion, makeEngineConfig, type EngineConfig } from './engine/config';
import { initWebGPU, WebGpuUnavailableError } from './engine/device';
import { HairSim } from './engine/hairSim';
import {
  SPHERE_DEFAULT_STRANDS,
  createSphereModel,
  loadModel,
  type HairModel,
} from './engine/model';
import { Renderer } from './engine/renderer';
import { runSelftest } from './engine/selftest';
import { Timestamps } from './engine/timestamps';
import { createGui, type ModelName } from './ui/gui';
import { Hud, showBanner } from './ui/hud';

declare global {
  interface Window {
    __selftestPromise?: Promise<boolean>;
  }
}

const MAX_DEVICE_PIXEL_RATIO = 2;

async function main(): Promise<void> {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement | null;
  if (!canvas) throw new Error('#gpu-canvas is missing from the document');

  const query = new URLSearchParams(location.search);
  const wantSphere = query.get('model') === 'sphere';
  const wantSelftest = query.has('selftest');
  const strandsOverride = Number.parseInt(query.get('strands') ?? '', 10);
  const sphereStrands = Number.isFinite(strandsOverride) ? strandsOverride : SPHERE_DEFAULT_STRANDS;

  // ---------------------------------------------------------------- device
  let gpu;
  try {
    gpu = await initWebGPU(canvas);
  } catch (err) {
    const message =
      err instanceof WebGpuUnavailableError
        ? err.message
        : `WebGPU initialisation failed: ${err instanceof Error ? err.message : String(err)}`;
    console.error(err);
    showBanner(message);
    return;
  }
  const { device, context, format, hasTimestamps } = gpu;

  let bannerShown = false;
  const reportFatal = (message: string): void => {
    console.error(message);
    if (!bannerShown) {
      bannerShown = true;
      showBanner(message);
    }
  };
  device.onuncapturederror = (event) => {
    reportFatal(`WebGPU error: ${event.error.message}`);
  };
  void device.lost.then((info) => {
    if (info.reason !== 'destroyed') reportFatal(`WebGPU device lost: ${info.message}`);
  });

  // ---------------------------------------------------------------- model
  // The FurryBall is the video-faithful mode: 16 particles per strand, a symmetric
  // model-local voxel box that stays around the ball through its bounce, and the
  // root -> tip colour gradient. The beast keeps the checked-in build's settings.
  const sphereConfig = (): EngineConfig =>
    makeEngineConfig({
      particlesPerStrand: 16,
      bbMin: [-7, -7, -7],
      bbMax: [7, 7, 7],
    });

  let model: HairModel;
  let cfg: EngineConfig;
  let modelName: ModelName;
  if (wantSphere) {
    model = createSphereModel(sphereStrands);
    cfg = sphereConfig();
    modelName = 'sphere';
  } else {
    const url = new URL('models/beast.json', document.baseURI).href;
    try {
      model = await loadModel(url);
      cfg = makeEngineConfig();
      modelName = 'beast';
    } catch (err) {
      console.warn(`Could not load ${url}, falling back to the procedural sphere.`, err);
      model = createSphereModel(sphereStrands);
      cfg = sphereConfig();
      modelName = 'sphere';
    }
  }

  // ---------------------------------------------------------------- engine
  const params = defaultParams();
  const colorsOverride = query.get('colors');
  params.colorMode =
    colorsOverride === 'gradient' || colorsOverride === 'alternating'
      ? (colorsOverride as ColorMode)
      : modelName === 'sphere'
        ? 'gradient'
        : 'alternating';

  const sim = new HairSim(device, model, cfg, params.colorMode, reportFatal);

  if (modelName === 'sphere') {
    // Two defaults have to follow the FurryBall's much denser, twice-as-long strands.
    // The beast keeps the settings.xml values exactly.
    params.repulsion = densityCompensatedRepulsion(params.repulsion, sim.numParticles);
    // same Gauss-Seidel sweeps per link as the original's 30 iterations over 8 particles
    params.numIterations = Math.round(
      (params.numIterations * cfg.particlesPerStrand) / NUM_HAIR_PARTICLES
    );
  }
  const renderer = new Renderer(device, format, sim, cfg, reportFatal);
  const camera = new OrbitCamera(canvas, [10, 15, 10]);
  const timestamps = hasTimestamps ? new Timestamps(device) : null;

  const hud = new Hud();
  hud.setCounts(sim.numParticles, sim.numStrands);
  createGui(params, {
    model: modelName,
    onModelChange: (next) => {
      const url = new URL(location.href);
      url.searchParams.set('model', next);
      location.href = url.href; // full reload = clean reinitialisation
    },
    onColorModeChange: () => sim.resetParticles(params.colorMode),
  });

  console.log(
    `HairyMess: model "${model.name}" — ${sim.numStrands} strands (${sim.paddedStrands} padded), ` +
      `${sim.numParticles} particles at ${cfg.particlesPerStrand}/strand, ${model.jointCount} joints, ` +
      `${model.frameCount} frames, colors ${params.colorMode}, ` +
      `timestamps ${hasTimestamps ? 'on' : 'unavailable'}`
  );

  // ---------------------------------------------------------------- sizing
  const maxDimension = device.limits.maxTextureDimension2D;
  function syncCanvasSize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DEVICE_PIXEL_RATIO);
    const cssWidth = canvas!.clientWidth || window.innerWidth;
    const cssHeight = canvas!.clientHeight || window.innerHeight;
    const width = Math.max(1, Math.min(maxDimension, Math.round(cssWidth * dpr)));
    const height = Math.max(1, Math.min(maxDimension, Math.round(cssHeight * dpr)));
    if (canvas!.width !== width) canvas!.width = width;
    if (canvas!.height !== height) canvas!.height = height;
    renderer.resize(canvas!.width, canvas!.height);
  }

  // ---------------------------------------------------------------- one frame
  function runFrame(dt: number): void {
    syncCanvasSize();
    sim.update(params, dt);

    const encoder = device.createCommandEncoder({ label: 'frame' });
    sim.encode(encoder, params, timestamps?.computePassWrites);
    renderer.render(
      encoder,
      context.getCurrentTexture().createView(),
      camera,
      params,
      timestamps?.renderPassWrites
    );
    const slot = timestamps ? timestamps.resolve(encoder) : -1;
    device.queue.submit([encoder.finish()]);
    if (timestamps) timestamps.read(slot);
  }

  // ---------------------------------------------------------------- input
  window.addEventListener('keydown', (event) => {
    const target = event.target as HTMLElement | null;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.isContentEditable)) {
      return;
    }
    if (event.key === 'f' || event.key === 'F') {
      event.preventDefault();
      if (document.fullscreenElement) {
        void document.exitFullscreen();
      } else {
        void document.documentElement.requestFullscreen().catch((err) => console.warn(err));
      }
    }
  });

  // ---------------------------------------------------------------- loop
  let previous = performance.now();
  let running = true;
  function loop(now: number): void {
    if (!running) return;
    const dt = Math.min((now - previous) / 1000, MAX_DT);
    previous = now;
    runFrame(Math.max(dt, 1e-4));
    hud.tick(timestamps?.timings);
    requestAnimationFrame(loop);
  }

  if (wantSelftest) {
    window.__selftestPromise = runSelftest({ device, sim, cfg, params, runFrame }).then((passed) => {
      previous = performance.now();
      requestAnimationFrame(loop);
      return passed;
    });
  } else {
    requestAnimationFrame(loop);
  }

  window.addEventListener('beforeunload', () => {
    running = false;
  });
}

void main().catch((err) => {
  console.error(err);
  showBanner(`Startup failed: ${err instanceof Error ? err.message : String(err)}`);
});
