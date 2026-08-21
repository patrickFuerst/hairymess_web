// HairyMess — WebGPU port. Boot, frame loop and UI wiring.

import { MAX_DT, NUM_HAIR_PARTICLES, defaultParams, type ColorMode } from './params';
import { OrbitCamera } from './engine/camera';
import { densityCompensatedRepulsion, makeEngineConfig, type EngineConfig } from './engine/config';
import { initWebGPU, WebGpuUnavailableError } from './engine/device';
import { HairSim } from './engine/hairSim';
import { PointerBrush } from './engine/interaction';
import { SPHERE_DEFAULT_STRANDS, createSphereModel, loadModel, type HairModel } from './engine/model';
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

/** session keys: the adaptive strand count, its step counter, the device-lost guard */
const KEY_STRANDS = 'hairymess.sphereStrands';
const KEY_DOWNSCALES = 'hairymess.downscales';
const KEY_RELOAD_GUARD = 'hairymess.reloadGuard';

/** adaptive quality: frames to warm up before judging, and the budget it aims at */
const ADAPTIVE_FRAMES = 120;
const ADAPTIVE_BUDGET_MS = 14;
const ADAPTIVE_TARGET_MS = 12;
const ADAPTIVE_MAX_STEPS = 2;
const ADAPTIVE_MIN_STRANDS = 10_000;

function session(key: string): string | null {
  try {
    return sessionStorage.getItem(key);
  } catch {
    return null; // private mode / storage disabled — the features degrade to off
  }
}

function setSession(key: string, value: string | null): void {
  try {
    if (value === null) sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}

async function main(): Promise<void> {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement | null;
  if (!canvas) throw new Error('#gpu-canvas is missing from the document');

  const query = new URLSearchParams(location.search);
  const wantSphere = query.get('model') === 'sphere';
  const wantSelftest = query.has('selftest');
  const strandsOverride = Number.parseInt(query.get('strands') ?? '', 10);
  const explicitStrands = Number.isFinite(strandsOverride);
  // an adaptive downscale from an earlier run of this session wins over the default,
  // but never over an explicit ?strands=
  const rememberedStrands = Number.parseInt(session(KEY_STRANDS) ?? '', 10);
  const sphereStrands = explicitStrands
    ? strandsOverride
    : Number.isFinite(rememberedStrands)
      ? rememberedStrands
      : SPHERE_DEFAULT_STRANDS;

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
  const { device, context, format, hasTimestamps, canPullInVertexStage } = gpu;

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
    if (info.reason === 'destroyed') return;
    console.warn(`WebGPU device lost: ${info.message}`);
    // One automatic recovery attempt per session. The guard is cleared once a run has
    // survived its first frames, so a genuine crash loop shows the banner instead of
    // reloading for ever.
    if (session(KEY_RELOAD_GUARD) === '1') {
      reportFatal(`WebGPU device lost: ${info.message}`);
      return;
    }
    setSession(KEY_RELOAD_GUARD, '1');
    console.warn('Reloading once to recover the device.');
    location.reload();
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
  if (!canPullInVertexStage) {
    params.renderMode = 'lines';
    console.warn(
      'This device does not allow storage buffers in the vertex stage ' +
        '(maxStorageBuffersInVertexStage = 0) — the hair falls back to the line path.'
    );
  }

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
  const renderer = new Renderer(device, format, sim, cfg, canPullInVertexStage, reportFatal);
  const camera = new OrbitCamera(canvas, [10, 15, 10]);
  const brush = new PointerBrush(canvas);
  const timestamps = hasTimestamps ? new Timestamps(device) : null;

  const hud = new Hud();
  hud.setCounts(sim.numParticles, sim.numStrands);
  createGui(params, {
    model: modelName,
    ribbonsAvailable: canPullInVertexStage,
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
      `render ${params.renderMode}, ${model.colliders.length} model collider(s), ` +
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
    const aspect = canvas!.height > 0 ? canvas!.width / canvas!.height : 1;
    sim.update(params, dt, brush.update(camera, model, aspect, dt));

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

  // ---------------------------------------------------------------- adaptive quality
  // Only the FurryBall, and only when the strand count was not asked for explicitly:
  // after a warm-up, a GPU frame heavier than the budget rescales the tessellation and
  // reloads once. The chosen count is remembered for the rest of the session.
  const downscales = Number.parseInt(session(KEY_DOWNSCALES) ?? '0', 10) || 0;
  const adaptiveEnabled =
    modelName === 'sphere' &&
    !explicitStrands &&
    !wantSelftest &&
    timestamps !== null &&
    downscales < ADAPTIVE_MAX_STEPS;

  function considerDownscale(): void {
    const timings = timestamps?.timings;
    const total = (timings?.simMs ?? 0) + (timings?.renderMs ?? 0);
    if (!(total > 0)) return; // timestamps never produced a reading
    if (total <= ADAPTIVE_BUDGET_MS) {
      console.log(
        `adaptive quality: ${total.toFixed(1)} ms/frame is within budget, ` +
          `staying at ${sim.numStrands} strands`
      );
      return; // the counter only ever counts actual downscales
    }
    const next = Math.max(
      ADAPTIVE_MIN_STRANDS,
      Math.round(sim.numStrands * (ADAPTIVE_TARGET_MS / total))
    );
    if (next >= sim.numStrands) return;
    console.warn(
      `adaptive quality: ${total.toFixed(1)} ms/frame exceeds ${ADAPTIVE_BUDGET_MS} ms — ` +
        `rebuilding at ~${next} strands (was ${sim.numStrands}) and reloading`
    );
    setSession(KEY_STRANDS, String(next));
    setSession(KEY_DOWNSCALES, String(downscales + 1));
    location.reload();
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
  let frames = 0;
  function loop(now: number): void {
    if (!running) return;
    const dt = Math.min((now - previous) / 1000, MAX_DT);
    previous = now;
    runFrame(Math.max(dt, 1e-4));
    hud.tick(timestamps?.timings);
    frames++;
    // the run is healthy — let a future device loss try its one reload again
    if (frames === 5) setSession(KEY_RELOAD_GUARD, null);
    if (adaptiveEnabled && frames === ADAPTIVE_FRAMES) considerDownscale();
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
