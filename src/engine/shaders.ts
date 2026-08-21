// WGSL sources, string-composed with a generated constants prelude so the shader
// constants can never drift from the engine config.

import type { EngineConfig } from './config';

import commonSrc from '../shaders/common.wgsl?raw';
import simCommonSrc from '../shaders/simCommon.wgsl?raw';
import skinningSrc from '../shaders/skinning.wgsl?raw';
import voxelFillSrc from '../shaders/voxelFill.wgsl?raw';
import voxelPostSrc from '../shaders/voxelPost.wgsl?raw';
import voxelFilterSrc from '../shaders/voxelFilter.wgsl?raw';
import simDFTLSrc from '../shaders/simDFTL.wgsl?raw';
import simPBDSrc from '../shaders/simPBD.wgsl?raw';
import backgroundSrc from '../shaders/background.wgsl?raw';
import meshSrc from '../shaders/mesh.wgsl?raw';
import hairSrc from '../shaders/hair.wgsl?raw';
import floorFadeSrc from '../shaders/floorFade.wgsl?raw';
import voxelDebugSrc from '../shaders/voxelDebug.wgsl?raw';
import bboxSrc from '../shaders/bbox.wgsl?raw';

export type ShaderName =
  | 'skinning'
  | 'voxelFill'
  | 'voxelPost'
  | 'voxelFilter'
  | 'simDFTL'
  | 'simPBD'
  | 'background'
  | 'mesh'
  | 'hair'
  | 'floorFade'
  | 'voxelDebug'
  | 'bbox';

export type ShaderSources = Record<ShaderName, string>;

function f32Literal(v: number): string {
  return Number.isInteger(v) ? `${v}.0` : `${v}`;
}

export function buildShaders(cfg: EngineConfig): ShaderSources {
  const constants = `// generated from the engine config — do not edit
const GRID_SIZE: i32 = ${cfg.gridSize};
const VOXEL_LOCAL_SIZE: u32 = ${cfg.voxelLocalSize}u;
const NUM_HAIR_PARTICLES: u32 = ${cfg.particlesPerStrand}u;
const STRANDS_PER_GROUP: u32 = ${cfg.strandsPerGroup}u;
const WORK_GROUP_SIZE: u32 = ${cfg.workGroupSize}u;
const DENSITY_SCALE: f32 = ${f32Literal(cfg.densityScale)};
const VELOCITY_SCALE: f32 = ${f32Literal(cfg.velocityScale)};
`;

  const prelude = `${constants}\n${commonSrc}\n`;

  return {
    skinning: prelude + skinningSrc,
    voxelFill: prelude + voxelFillSrc,
    voxelPost: prelude + voxelPostSrc,
    voxelFilter: prelude + voxelFilterSrc,
    simDFTL: prelude + simCommonSrc + '\n' + simDFTLSrc,
    simPBD: prelude + simCommonSrc + '\n' + simPBDSrc,
    background: prelude + backgroundSrc,
    mesh: prelude + meshSrc,
    hair: prelude + hairSrc,
    floorFade: prelude + floorFadeSrc,
    voxelDebug: prelude + voxelDebugSrc,
    bbox: prelude + bboxSrc,
  };
}
