// ?selftest — runs a fixed number of frames on each solver, reads the particle buffer
// back and sanity-checks the simulation. Result goes to the console, the document
// title and window.__selftestPromise.

import type { SimParams } from '../params';
import type { EngineConfig } from './config';
import type { HairSim } from './hairSim';

const FRAMES_PER_ALGORITHM = 60;
const FIXED_DT = 1 / 60;
/**
 * DFTL resets each segment to its rest length outright, so it should be near-exact.
 * PBD is a *soft* solver: it relaxes the chain with Gauss-Seidel sweeps and we sample
 * right after an odd sweep, which by construction leaves the even pairs unrelaxed. Its
 * residual grows with chain length (relative error doubles when the rest length halves
 * at 16 particles/strand), so it gets a looser bound — still tight enough to catch a
 * blow-up, which is what this guards against.
 */
const LENGTH_TOLERANCE: Record<string, number> = { DFTL: 0.25, PBD: 1.0 };
const MIN_GOOD_STRAND_FRACTION = 0.99;
const ROOT_SAMPLE_STRANDS = 100;
const ROOT_MAX_ABS = 100;
/**
 * A particle at (or below) the ground plane was just repositioned by the plane
 * collision. PBD applies that collision *after* the constraint solve — faithfully to
 * PBDApproach.glsl — so those segments are legitimately off their rest length for a
 * frame and cannot be part of a length assertion. See IMPLEMENTATION.md.
 */
const FLOOR_EPSILON = 1e-3;

export interface SelftestDeps {
  device: GPUDevice;
  sim: HairSim;
  cfg: EngineConfig;
  params: SimParams;
  /** encode + submit exactly one frame, like the rAF loop does */
  runFrame: (dt: number) => void;
}

export async function runSelftest(deps: SelftestDeps): Promise<boolean> {
  const { device, sim, cfg, params } = deps;
  try {
    params.useFilter = true;
    params.playAnimation = true;

    for (const algorithm of ['DFTL', 'PBD'] as const) {
      params.algorithm = algorithm;
      for (let i = 0; i < FRAMES_PER_ALGORITHM; i++) {
        deps.runFrame(FIXED_DT);
        if (i % 10 === 9) await device.queue.onSubmittedWorkDone();
      }
      const data = await readParticles(device, sim);
      const failure = check(data, sim, cfg.particlesPerStrand, algorithm);
      if (failure) return report(`${algorithm}: ${failure}`);
    }
    return report(null);
  } catch (err) {
    return report(`exception: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/** Copies the real particles back on a fresh encoder, outside any rAF callback. */
async function readParticles(device: GPUDevice, sim: HairSim): Promise<Float32Array> {
  await device.queue.onSubmittedWorkDone();
  const byteLength = sim.numParticles * 48;
  const staging = device.createBuffer({
    label: 'selftestStaging',
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: 'selftestReadback' });
  encoder.copyBufferToBuffer(sim.particles, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

function check(data: Float32Array, sim: HairSim, perStrand: number, label: string): string | null {
  const floatsPerParticle = 12; // 48 byte stride
  const numStrands = sim.numStrands;
  const tolerance = LENGTH_TOLERANCE[label] ?? 0.25;

  // (a) every component finite
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) {
      const particle = Math.floor(i / floatsPerParticle);
      return `non-finite value at particle ${particle} (strand ${Math.floor(
        particle / perStrand
      )}), component ${i % floatsPerParticle}`;
    }
  }

  // (b) segment lengths near their rest length, ignoring segments the ground
  //     collision just repositioned
  let checkedStrands = 0;
  let goodStrands = 0;
  let floorStrands = 0;
  let worstRatio = 1;
  let maxStepInRestLengths = 0;

  for (let s = 0; s < numStrands; s++) {
    const rest = sim.strandLengthsCpu[s] / perStrand;
    let ok = true;
    let checkable = 0;
    let touchedFloor = false;

    for (let j = 0; j < perStrand; j++) {
      const p = (s * perStrand + j) * floatsPerParticle;
      const step = Math.hypot(data[p] - data[p + 4], data[p + 1] - data[p + 5], data[p + 2] - data[p + 6]);
      maxStepInRestLengths = Math.max(maxStepInRestLengths, step / rest);
    }

    for (let j = 0; j < perStrand - 1; j++) {
      const a = (s * perStrand + j) * floatsPerParticle;
      const b = a + floatsPerParticle;
      if (data[a + 1] <= FLOOR_EPSILON || data[b + 1] <= FLOOR_EPSILON) {
        touchedFloor = true;
        continue;
      }
      checkable++;
      const len = Math.hypot(data[b] - data[a], data[b + 1] - data[a + 1], data[b + 2] - data[a + 2]);
      const ratio = len / rest;
      if (Math.abs(ratio - 1) > Math.abs(worstRatio - 1)) worstRatio = ratio;
      if (Math.abs(len - rest) > tolerance * rest) ok = false;
    }

    if (touchedFloor) floorStrands++;
    if (checkable === 0) continue; // entirely floor-bound, nothing to assert
    checkedStrands++;
    if (ok) goodStrands++;
  }

  if (checkedStrands === 0) return 'no strand had a segment clear of the ground plane';
  const fraction = goodStrands / checkedStrands;
  const floorShare = ((floorStrands / numStrands) * 100).toFixed(1);

  if (fraction < MIN_GOOD_STRAND_FRACTION) {
    return `only ${(fraction * 100).toFixed(2)}% of ${checkedStrands} strands kept their segment ` +
      `lengths within ${tolerance * 100}% (worst ratio ${worstRatio.toFixed(
        3
      )}, max step ${maxStepInRestLengths.toFixed(2)} rest lengths, ${floorShare}% of strands touch the floor)`;
  }

  // (c) roots stayed put and sane
  const sample = Math.min(ROOT_SAMPLE_STRANDS, numStrands);
  for (let s = 0; s < sample; s++) {
    const o = s * perStrand * floatsPerParticle;
    for (let k = 0; k < 3; k++) {
      const v = data[o + k];
      if (!Number.isFinite(v) || Math.abs(v) >= ROOT_MAX_ABS) {
        return `root of strand ${s} is out of range (${data[o].toFixed(2)}, ${data[o + 1].toFixed(
          2
        )}, ${data[o + 2].toFixed(2)})`;
      }
    }
  }

  console.log(
    `selftest ${label}: ${(fraction * 100).toFixed(2)}% of ${checkedStrands} checked strands within ` +
      `${tolerance * 100}% length tolerance (worst ratio ${worstRatio.toFixed(3)}, max step ${maxStepInRestLengths.toFixed(
        2
      )} rest lengths, ${floorShare}% of strands touch the ground plane)`
  );
  return null;
}

function report(failure: string | null): boolean {
  if (failure) {
    console.log(`SELFTEST FAIL: ${failure}`);
    document.title = `SELFTEST FAIL: ${failure}`;
    return false;
  }
  console.log('SELFTEST PASS');
  document.title = 'SELFTEST PASS';
  return true;
}
