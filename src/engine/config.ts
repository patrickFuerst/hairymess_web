// Everything the kernels are specialised on lives here, so strand/particle counts and
// grid extents flow from data instead of being hardcoded in the engine or shaders.
// (A later "video-faithful FurryBall" mode wants particlesPerStrand = 16 and a
// symmetric bounding box — that should only need a different config object.)

import {
  BB_MAX,
  BB_MIN,
  DENSITY_SCALE,
  MAX_HAIR_LENGTH,
  MIN_HAIR_LENGTH,
  NUM_HAIR_PARTICLES,
  VELOCITY_SCALE,
  VOXEL_GRID_SIZE,
  VOXEL_LOCAL_SIZE,
  WORK_GROUP_SIZE,
} from '../params';

export interface EngineConfig {
  /** particles per hair strand (8 in the shipped build) */
  particlesPerStrand: number;
  /** 1D compute workgroup size for the particle kernels */
  workGroupSize: number;
  /** derived: workGroupSize / particlesPerStrand — also the strand padding multiple */
  strandsPerGroup: number;
  gridSize: number;
  voxelLocalSize: number;
  densityScale: number;
  velocityScale: number;
  minHairLength: number;
  maxHairLength: number;
  bbMin: readonly [number, number, number];
  bbMax: readonly [number, number, number];
  /** PRNG seed for the per-strand hair length, so runs are reproducible */
  seed: number;
}

export function makeEngineConfig(overrides: Partial<EngineConfig> = {}): EngineConfig {
  const particlesPerStrand = overrides.particlesPerStrand ?? NUM_HAIR_PARTICLES;
  const workGroupSize = overrides.workGroupSize ?? WORK_GROUP_SIZE;
  const gridSize = overrides.gridSize ?? VOXEL_GRID_SIZE;
  // The original dispatches the grid kernels with 8x8x8 = 512 invocations per
  // workgroup; WebGPU guarantees only maxComputeInvocationsPerWorkgroup = 256, so the
  // default here is 4x4x4 = 64. These kernels use neither workgroup memory nor
  // barriers, so the workgroup shape is a pure scheduling choice.
  const voxelLocalSize = overrides.voxelLocalSize ?? Math.min(VOXEL_LOCAL_SIZE, 4);

  if (particlesPerStrand < 2 || workGroupSize % particlesPerStrand !== 0) {
    throw new Error(
      `workGroupSize (${workGroupSize}) must be a multiple of particlesPerStrand (${particlesPerStrand})`
    );
  }
  if (gridSize % voxelLocalSize !== 0) {
    throw new Error(`gridSize (${gridSize}) must be a multiple of voxelLocalSize (${voxelLocalSize})`);
  }

  return {
    particlesPerStrand,
    workGroupSize,
    strandsPerGroup: workGroupSize / particlesPerStrand,
    gridSize,
    voxelLocalSize,
    densityScale: overrides.densityScale ?? DENSITY_SCALE,
    velocityScale: overrides.velocityScale ?? VELOCITY_SCALE,
    minHairLength: overrides.minHairLength ?? MIN_HAIR_LENGTH,
    maxHairLength: overrides.maxHairLength ?? MAX_HAIR_LENGTH,
    bbMin: overrides.bbMin ?? BB_MIN,
    bbMax: overrides.bbMax ?? BB_MAX,
    seed: overrides.seed ?? 0x9e3779b9,
  };
}

/**
 * Particle count the shipped `settings.xml` values were tuned against: the beast's
 * 17,365 strands x 8 particles.
 */
export const REPULSION_REFERENCE_PARTICLES = 17365 * 8;

/**
 * The repulsion term uses the *non-normalised* density gradient, whose magnitude is
 * proportional to how many particles land in a voxel. The video-faithful FurryBall
 * packs ~9x more particles into a bigger box than the beast, so the settings.xml
 * repulsion would drive it about 13x harder than the original ever ran — the fur is
 * blasted straight out and the (soft) PBD constraints cannot hold it. Normalising by
 * particle count keeps the repulsion *acceleration* at the value the defaults were
 * tuned for; at the reference count this returns `base` unchanged.
 */
export function densityCompensatedRepulsion(base: number, numParticles: number): number {
  if (numParticles <= 0) return base;
  return base * (REPULSION_REFERENCE_PARTICLES / numParticles);
}

/** mulberry32 — deterministic replacement for the original's ofRandom(). */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
