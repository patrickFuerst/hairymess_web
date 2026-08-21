// Simulation parameters — defaults match the original bin/data/settings.xml.

export type Algorithm = 'DFTL' | 'PBD';

/**
 * 'alternating' — the checked-in scheme: even strands orange, odd strands teal.
 * 'gradient'    — the demo video's root -> tip gradient (default for the FurryBall).
 */
export type ColorMode = 'alternating' | 'gradient';

export interface SimParams {
  algorithm: Algorithm;
  colorMode: ColorMode;
  velocityDamping: number;
  numIterations: number;
  stiffness: number;
  friction: number;
  repulsion: number;
  ftlDamping: number;
  gravityX: number;
  gravityY: number;
  gravityZ: number;
  useFilter: boolean;
  playAnimation: boolean;
  drawFur: boolean;
  drawVoxelGrid: boolean;
  drawBoundingBox: boolean;
}

export function defaultParams(): SimParams {
  return {
    algorithm: 'DFTL',
    colorMode: 'alternating',
    velocityDamping: 0.984694,
    numIterations: 30,
    stiffness: 1.0,
    friction: 0.0663265,
    repulsion: 117.347,
    ftlDamping: 1.0,
    gravityX: 0,
    gravityY: -10,
    gravityZ: 0,
    useFilter: true,
    playAnimation: true,
    drawFur: true,
    drawVoxelGrid: false,
    drawBoundingBox: false,
  };
}

export const NUM_HAIR_PARTICLES = 8;
export const WORK_GROUP_SIZE = 64;
export const STRANDS_PER_GROUP = WORK_GROUP_SIZE / NUM_HAIR_PARTICLES;
export const VOXEL_GRID_SIZE = 64;
export const VOXEL_LOCAL_SIZE = 8;
export const MIN_HAIR_LENGTH = 1.0;
export const MAX_HAIR_LENGTH = 1.2;
export const MAX_DT = 0.02;
export const DENSITY_SCALE = 4096.0;
export const VELOCITY_SCALE = 1024.0;

// Static simulation bounding box (voxel grid extent), like the original.
export const BB_MIN: [number, number, number] = [-5, 0, -5];
export const BB_MAX: [number, number, number] = [5, 10, 5];
