// GPU simulation: buffers, compute pipelines and every bind group (both ping-pong
// parities) built up front. Nothing here allocates during the frame loop.

import type { ColorMode, SimParams } from '../params';
import { makeRng, type EngineConfig } from './config';
import { createShaderModule } from './device';
import { PARTICLE_STRIDE, SIM_U, SKIN_U, FILTER_U, UniformScratch, gpuWrite } from './layout';
import { lerpPalette, skinWorld, type HairModel, type SkinnedSurface } from './model';
import { buildShaders } from './shaders';

/** parked position of the padding strands: far outside the voxel grid, never drawn */
const PARKED = -10000;
const RESTART_INDEX = 0xffffffff;

/** ofColor(255,190,42) and ofColor(0,163,136), the alternating strand colours */
const STRAND_COLORS: ReadonlyArray<readonly [number, number, number]> = [
  [255 / 255, 190 / 255, 42 / 255],
  [0 / 255, 163 / 255, 136 / 255],
];

/**
 * Root -> tip gradient of the original demo video (see VIDEO_REFERENCE.md):
 * #d5491d -> #b7cf4f -> #52c8bc. Those hex values were read off the *rendered* video,
 * and the hair fragment shader applies pow(1/2.2), so the stops are stored raised to
 * 2.2 — after the shader's gamma they land back on the sampled colours.
 */
const GRADIENT_STOPS: ReadonlyArray<readonly [number, number, number]> = [
  linearise(0xd5, 0x49, 0x1d),
  linearise(0xb7, 0xcf, 0x4f),
  linearise(0x52, 0xc8, 0xbc),
];

function linearise(r: number, g: number, b: number): [number, number, number] {
  return [(r / 255) ** 2.2, (g / 255) ** 2.2, (b / 255) ** 2.2];
}

function gradientColor(t: number, out: [number, number, number]): void {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (GRADIENT_STOPS.length - 1);
  const i = Math.min(GRADIENT_STOPS.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const a = GRADIENT_STOPS[i];
  const b = GRADIENT_STOPS[i + 1];
  out[0] = a[0] + (b[0] - a[0]) * f;
  out[1] = a[1] + (b[1] - a[1]) * f;
  out[2] = a[2] + (b[2] - a[2]) * f;
}

export interface SimCounts {
  numStrands: number;
  numParticles: number;
  paddedStrands: number;
  paddedParticles: number;
}

export class HairSim {
  readonly device: GPUDevice;
  readonly model: HairModel;
  readonly cfg: EngineConfig;

  readonly numStrands: number;
  readonly numParticles: number;
  readonly paddedStrands: number;
  readonly paddedParticles: number;

  // --- buffers exposed to the renderer / selftest
  readonly particles: GPUBuffer;
  readonly hairIndexBuffer: GPUBuffer;
  readonly hairIndexCount: number;
  readonly skinnedPositions: GPUBuffer;
  readonly meshIndexBuffer: GPUBuffer;
  readonly meshIndexCount: number;
  readonly simUniformBuffer: GPUBuffer;
  readonly velocityGrid: [GPUBuffer, GPUBuffer];
  readonly gradientGrid: [GPUBuffer, GPUBuffer];
  readonly densityAtomic: GPUBuffer;

  /** rest length of every real strand (CPU copy, used by the selftest) */
  readonly strandLengthsCpu: Float32Array;

  private readonly velocityAtomic: GPUBuffer;
  private readonly strandLengths: GPUBuffer;
  private readonly roots: GPUBuffer;
  private readonly bindPositions: GPUBuffer;
  private readonly skinJoints: GPUBuffer;
  private readonly skinWeights: GPUBuffer;
  private readonly palette: GPUBuffer;
  private readonly skinUniformBuffer: GPUBuffer;
  private readonly filterUniformBuffers: GPUBuffer[];

  private readonly skinPipeline: GPUComputePipeline;
  private readonly fillPipeline: GPUComputePipeline;
  private readonly postPipeline: GPUComputePipeline;
  private readonly filterPipeline: GPUComputePipeline;
  private readonly simPipelines: Record<'DFTL' | 'PBD', GPUComputePipeline>;

  private readonly skinBindGroup: GPUBindGroup;
  private readonly fillBindGroup: GPUBindGroup;
  private readonly postBindGroups: GPUBindGroup[]; // [ping]
  private readonly filterBindGroups: GPUBindGroup[]; // [ping]
  private readonly filterAxisBindGroups: GPUBindGroup[]; // [axis]
  private readonly simBindGroups: Record<'DFTL' | 'PBD', GPUBindGroup[]>; // [alg][ping]

  private readonly simScratch = new UniformScratch(SIM_U.SIZE);
  private readonly paletteScratch: Float32Array;
  /** inverse of the previous frame's model matrix (parity with the original UBO) */
  private readonly modelMatrixInverse: Float32Array;
  private readonly prevModelMatrix: Float32Array;
  private colorMode: ColorMode;

  /** index of the grid buffer holding the current (readable) values */
  private ping = 0;
  private animationPhase = 0;

  constructor(
    device: GPUDevice,
    model: HairModel,
    cfg: EngineConfig,
    colorMode: ColorMode,
    onShaderError?: (m: string) => void
  ) {
    this.device = device;
    this.model = model;
    this.cfg = cfg;
    this.colorMode = colorMode;

    const perStrand = cfg.particlesPerStrand;
    this.numStrands = model.vertexCount;
    this.numParticles = this.numStrands * perStrand;
    this.paddedStrands = Math.ceil(this.numStrands / cfg.strandsPerGroup) * cfg.strandsPerGroup;
    this.paddedParticles = this.paddedStrands * perStrand;

    // ---------------------------------------------------------------- initial state
    this.paletteScratch = new Float32Array(model.jointCount * 16);
    lerpPalette(model, 0, this.paletteScratch);
    model.updateTransform?.(0);
    const surface = skinWorld(model, this.paletteScratch);
    const init = buildInitialState(model, surface, cfg, this.paddedStrands, colorMode);
    this.strandLengthsCpu = init.strandLengthsReal;

    // ---------------------------------------------------------------- buffers
    const S = GPUBufferUsage.STORAGE;
    const CD = GPUBufferUsage.COPY_DST;

    this.particles = device.createBuffer({
      label: 'particles',
      size: this.paddedParticles * PARTICLE_STRIDE,
      usage: S | GPUBufferUsage.VERTEX | CD | GPUBufferUsage.COPY_SRC,
    });
    gpuWrite(device, this.particles, init.particles);

    this.strandLengths = device.createBuffer({
      label: 'strandLengths',
      size: this.paddedStrands * 4,
      usage: S | CD,
    });
    gpuWrite(device, this.strandLengths, init.strandLengths);

    // roots are rewritten by the skinning pass every frame; only the padded tail,
    // which the skinning dispatch does not cover, keeps this initial value.
    this.roots = device.createBuffer({
      label: 'roots',
      size: this.paddedStrands * 16,
      usage: S | CD,
    });
    gpuWrite(device, this.roots, init.roots);

    this.hairIndexBuffer = device.createBuffer({
      label: 'hairIndices',
      size: init.hairIndices.byteLength,
      usage: GPUBufferUsage.INDEX | CD,
    });
    gpuWrite(device, this.hairIndexBuffer, init.hairIndices);
    this.hairIndexCount = init.hairIndices.length;

    this.bindPositions = device.createBuffer({
      label: 'bindPositions',
      size: model.bindPositions.byteLength,
      usage: S | CD,
    });
    gpuWrite(device, this.bindPositions, model.bindPositions);

    this.skinJoints = device.createBuffer({
      label: 'skinJoints',
      size: model.skinJoints.byteLength,
      usage: S | CD,
    });
    gpuWrite(device, this.skinJoints, model.skinJoints);

    this.skinWeights = device.createBuffer({
      label: 'skinWeights',
      size: model.skinWeights.byteLength,
      usage: S | CD,
    });
    gpuWrite(device, this.skinWeights, model.skinWeights);

    this.palette = device.createBuffer({
      label: 'palette',
      size: this.paletteScratch.byteLength,
      usage: S | CD,
    });
    gpuWrite(device, this.palette, this.paletteScratch);

    this.skinnedPositions = device.createBuffer({
      label: 'skinnedPositions',
      size: model.vertexCount * 16,
      usage: S | GPUBufferUsage.VERTEX | CD,
    });

    this.meshIndexBuffer = device.createBuffer({
      label: 'meshIndices',
      size: model.indices.byteLength,
      usage: GPUBufferUsage.INDEX | CD,
    });
    gpuWrite(device, this.meshIndexBuffer, model.indices);
    this.meshIndexCount = model.indices.length;

    // clearBuffer() counts as a copy destination, and the voxel debug view pulls all
    // three grids through vertex buffers (exactly like the original's mVoxelVBO).
    const cells = cfg.gridSize ** 3;
    const gridUsage = S | CD | GPUBufferUsage.VERTEX;
    this.densityAtomic = device.createBuffer({ label: 'densityAtomic', size: cells * 4, usage: gridUsage });
    this.velocityAtomic = device.createBuffer({ label: 'velocityAtomic', size: cells * 16, usage: S | CD });
    this.velocityGrid = [
      device.createBuffer({ label: 'velocityGrid0', size: cells * 16, usage: gridUsage }),
      device.createBuffer({ label: 'velocityGrid1', size: cells * 16, usage: gridUsage }),
    ];
    this.gradientGrid = [
      device.createBuffer({ label: 'gradientGrid0', size: cells * 16, usage: gridUsage }),
      device.createBuffer({ label: 'gradientGrid1', size: cells * 16, usage: gridUsage }),
    ];

    const U = GPUBufferUsage.UNIFORM;
    this.simUniformBuffer = device.createBuffer({ label: 'simUniforms', size: SIM_U.SIZE, usage: U | CD });
    this.skinUniformBuffer = device.createBuffer({ label: 'skinUniforms', size: SKIN_U.SIZE, usage: U | CD });
    {
      const s = new UniformScratch(SKIN_U.SIZE);
      s.setU32(SKIN_U.numVerts, model.vertexCount);
      s.setU32(SKIN_U.numJoints, model.jointCount);
      gpuWrite(device, this.skinUniformBuffer, s.bytes);
    }
    this.filterUniformBuffers = [0, 1, 2].map((axis) => {
      const buf = device.createBuffer({ label: `filterAxis${axis}`, size: FILTER_U.SIZE, usage: U | CD });
      const s = new UniformScratch(FILTER_U.SIZE);
      s.setI32(FILTER_U.axis, axis);
      gpuWrite(device, buf, s.bytes);
      return buf;
    });

    this.prevModelMatrix = new Float32Array(model.modelMatrix);
    this.modelMatrixInverse = invertMat4(model.modelMatrix);

    // ---------------------------------------------------------------- pipelines
    const src = buildShaders(cfg);
    const mod = (name: keyof typeof src) => createShaderModule(device, name, src[name], onShaderError);

    const uniformEntry = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform' },
    });
    const readEntry = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'read-only-storage' },
    });
    const rwEntry = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' },
    });

    // --- skinning
    const skinLayout = device.createBindGroupLayout({
      label: 'skinLayout',
      entries: [
        uniformEntry(0),
        readEntry(1),
        readEntry(2),
        readEntry(3),
        readEntry(4),
        rwEntry(5),
        rwEntry(6),
      ],
    });
    this.skinPipeline = device.createComputePipeline({
      label: 'skinning',
      layout: device.createPipelineLayout({ bindGroupLayouts: [skinLayout] }),
      compute: { module: mod('skinning'), entryPoint: 'main' },
    });
    this.skinBindGroup = device.createBindGroup({
      label: 'skinBindGroup',
      layout: skinLayout,
      entries: [
        { binding: 0, resource: { buffer: this.skinUniformBuffer } },
        { binding: 1, resource: { buffer: this.bindPositions } },
        { binding: 2, resource: { buffer: this.skinJoints } },
        { binding: 3, resource: { buffer: this.skinWeights } },
        { binding: 4, resource: { buffer: this.palette } },
        { binding: 5, resource: { buffer: this.roots } },
        { binding: 6, resource: { buffer: this.skinnedPositions } },
      ],
    });

    // --- voxel fill
    const fillLayout = device.createBindGroupLayout({
      label: 'voxelFillLayout',
      entries: [uniformEntry(0), readEntry(1), rwEntry(2), rwEntry(3)],
    });
    this.fillPipeline = device.createComputePipeline({
      label: 'voxelFill',
      layout: device.createPipelineLayout({ bindGroupLayouts: [fillLayout] }),
      compute: { module: mod('voxelFill'), entryPoint: 'main' },
    });
    this.fillBindGroup = device.createBindGroup({
      label: 'voxelFillBindGroup',
      layout: fillLayout,
      entries: [
        { binding: 0, resource: { buffer: this.simUniformBuffer } },
        { binding: 1, resource: { buffer: this.particles } },
        { binding: 2, resource: { buffer: this.densityAtomic } },
        { binding: 3, resource: { buffer: this.velocityAtomic } },
      ],
    });

    // --- voxel post
    const postLayout = device.createBindGroupLayout({
      label: 'voxelPostLayout',
      entries: [uniformEntry(0), readEntry(1), readEntry(2), rwEntry(3), rwEntry(4)],
    });
    this.postPipeline = device.createComputePipeline({
      label: 'voxelPost',
      layout: device.createPipelineLayout({ bindGroupLayouts: [postLayout] }),
      compute: { module: mod('voxelPost'), entryPoint: 'main' },
    });
    this.postBindGroups = [0, 1].map((ping) =>
      device.createBindGroup({
        label: `voxelPostBindGroup${ping}`,
        layout: postLayout,
        entries: [
          { binding: 0, resource: { buffer: this.simUniformBuffer } },
          { binding: 1, resource: { buffer: this.densityAtomic } },
          { binding: 2, resource: { buffer: this.velocityAtomic } },
          { binding: 3, resource: { buffer: this.velocityGrid[1 - ping] } },
          { binding: 4, resource: { buffer: this.gradientGrid[1 - ping] } },
        ],
      })
    );

    // --- voxel filter (group 0 = ping-pong, group 1 = axis)
    const filterLayout = device.createBindGroupLayout({
      label: 'voxelFilterLayout',
      entries: [readEntry(0), readEntry(1), rwEntry(2), rwEntry(3)],
    });
    const filterAxisLayout = device.createBindGroupLayout({
      label: 'voxelFilterAxisLayout',
      entries: [uniformEntry(0)],
    });
    this.filterPipeline = device.createComputePipeline({
      label: 'voxelFilter',
      layout: device.createPipelineLayout({ bindGroupLayouts: [filterLayout, filterAxisLayout] }),
      compute: { module: mod('voxelFilter'), entryPoint: 'main' },
    });
    this.filterBindGroups = [0, 1].map((ping) =>
      device.createBindGroup({
        label: `voxelFilterBindGroup${ping}`,
        layout: filterLayout,
        entries: [
          { binding: 0, resource: { buffer: this.velocityGrid[ping] } },
          { binding: 1, resource: { buffer: this.gradientGrid[ping] } },
          { binding: 2, resource: { buffer: this.velocityGrid[1 - ping] } },
          { binding: 3, resource: { buffer: this.gradientGrid[1 - ping] } },
        ],
      })
    );
    this.filterAxisBindGroups = this.filterUniformBuffers.map((buffer, axis) =>
      device.createBindGroup({
        label: `voxelFilterAxis${axis}`,
        layout: filterAxisLayout,
        entries: [{ binding: 0, resource: { buffer } }],
      })
    );

    // --- simulation (two algorithms x two parities)
    const simLayout = device.createBindGroupLayout({
      label: 'simLayout',
      entries: [uniformEntry(0), rwEntry(1), readEntry(2), readEntry(3), readEntry(4), readEntry(5)],
    });
    const simPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [simLayout] });
    this.simPipelines = {
      DFTL: device.createComputePipeline({
        label: 'simDFTL',
        layout: simPipelineLayout,
        compute: { module: mod('simDFTL'), entryPoint: 'main' },
      }),
      PBD: device.createComputePipeline({
        label: 'simPBD',
        layout: simPipelineLayout,
        compute: { module: mod('simPBD'), entryPoint: 'main' },
      }),
    };
    const simGroups = [0, 1].map((ping) =>
      device.createBindGroup({
        label: `simBindGroup${ping}`,
        layout: simLayout,
        entries: [
          { binding: 0, resource: { buffer: this.simUniformBuffer } },
          { binding: 1, resource: { buffer: this.particles } },
          { binding: 2, resource: { buffer: this.strandLengths } },
          { binding: 3, resource: { buffer: this.roots } },
          { binding: 4, resource: { buffer: this.velocityGrid[ping] } },
          { binding: 5, resource: { buffer: this.gradientGrid[ping] } },
        ],
      })
    );
    // both algorithms share one layout, so they share the bind groups too
    this.simBindGroups = { DFTL: simGroups, PBD: simGroups };
  }

  get counts(): SimCounts {
    return {
      numStrands: this.numStrands,
      numParticles: this.numParticles,
      paddedStrands: this.paddedStrands,
      paddedParticles: this.paddedParticles,
    };
  }

  /** grid buffer index the sim (and the debug view) reads this frame */
  get readIndex(): number {
    return this.ping;
  }

  get phase(): number {
    return this.animationPhase;
  }

  /**
   * Rebuilds the particle buffer at the model's current pose — used when the colour
   * scheme changes. Hair lengths are regenerated from the same seed, so only the
   * colours (and the reset to straight strands) change.
   */
  resetParticles(colorMode: ColorMode): void {
    this.colorMode = colorMode;
    lerpPalette(this.model, this.animationPhase, this.paletteScratch);
    this.model.updateTransform?.(this.animationPhase);
    const surface = skinWorld(this.model, this.paletteScratch);
    const init = buildInitialState(this.model, surface, this.cfg, this.paddedStrands, colorMode);
    gpuWrite(this.device, this.particles, init.particles);
    gpuWrite(this.device, this.strandLengths, init.strandLengths);
  }

  get currentColorMode(): ColorMode {
    return this.colorMode;
  }

  /** Advances the animation clock and uploads the per-frame uniforms + joint palette. */
  update(params: SimParams, dt: number): void {
    const model = this.model;
    if (params.playAnimation) {
      this.animationPhase = (this.animationPhase + dt) % model.duration;
      if (this.animationPhase < 0) this.animationPhase += model.duration;
    }
    lerpPalette(model, this.animationPhase, this.paletteScratch);
    gpuWrite(this.device, this.palette, this.paletteScratch);

    // rigid placement (the FurryBall's bounce + roll); static for the beast
    invertMat4Into(this.prevModelMatrix, this.modelMatrixInverse);
    model.updateTransform?.(this.animationPhase);
    this.prevModelMatrix.set(model.modelMatrix);

    const s = this.simScratch;
    const cfg = this.cfg;
    const t = model.simTranslation;
    s.setMat4(SIM_U.modelMatrix, model.modelMatrix);
    s.setMat4(SIM_U.modelMatrixPrevInverted, this.modelMatrixInverse);
    s.setVec4(SIM_U.modelTranslation, t[0], t[1], t[2], 0);
    s.setVec4(SIM_U.gravity, params.gravityX, params.gravityY, params.gravityZ, 0);
    s.setVec4(SIM_U.minBB, cfg.bbMin[0], cfg.bbMin[1], cfg.bbMin[2], 0);
    s.setVec4(SIM_U.maxBB, cfg.bbMax[0], cfg.bbMax[1], cfg.bbMax[2], 0);
    s.setF32(SIM_U.velocityDamping, params.velocityDamping);
    s.setI32(SIM_U.numIterationsPBD, Math.max(1, Math.round(params.numIterations)));
    s.setF32(SIM_U.stiffness, params.stiffness);
    s.setF32(SIM_U.friction, params.friction);
    s.setF32(SIM_U.repulsion, params.repulsion);
    s.setF32(SIM_U.ftlDamping, params.ftlDamping);
    s.setF32(SIM_U.deltaTime, Math.max(dt, 1e-5));
    s.setI32(SIM_U.gridSize, cfg.gridSize);
    s.setU32(SIM_U.numVerticesPerStrand, cfg.particlesPerStrand);
    s.setU32(SIM_U.numStrandsPerThreadGroup, cfg.strandsPerGroup);
    s.setU32(SIM_U.numStrands, this.numStrands);
    s.setU32(SIM_U.pad0, 0);
    gpuWrite(this.device, this.simUniformBuffer, s.bytes);
  }

  /** One clear + one compute pass with every dispatch of the frame graph. */
  encode(encoder: GPUCommandEncoder, params: SimParams, timestampWrites?: GPUComputePassTimestampWrites): void {
    const cfg = this.cfg;
    encoder.clearBuffer(this.densityAtomic);
    encoder.clearBuffer(this.velocityAtomic);

    const pass = encoder.beginComputePass(
      timestampWrites ? { label: 'compute', timestampWrites } : { label: 'compute' }
    );

    // 1. skinning -> roots + mesh positions
    pass.setPipeline(this.skinPipeline);
    pass.setBindGroup(0, this.skinBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.model.vertexCount / cfg.workGroupSize));

    // 2. voxel grid scatter
    const particleGroups = this.paddedParticles / cfg.workGroupSize;
    pass.setPipeline(this.fillPipeline);
    pass.setBindGroup(0, this.fillBindGroup);
    pass.dispatchWorkgroups(particleGroups);

    // 3. decode + gradient, writes the other parity
    const gridGroups = cfg.gridSize / cfg.voxelLocalSize;
    pass.setPipeline(this.postPipeline);
    pass.setBindGroup(0, this.postBindGroups[this.ping]);
    pass.dispatchWorkgroups(gridGroups, gridGroups, gridGroups);
    this.ping ^= 1;

    // 4. separable box filter, one pass per axis
    if (params.useFilter) {
      pass.setPipeline(this.filterPipeline);
      for (let axis = 0; axis < 3; axis++) {
        pass.setBindGroup(0, this.filterBindGroups[this.ping]);
        pass.setBindGroup(1, this.filterAxisBindGroups[axis]);
        pass.dispatchWorkgroups(gridGroups, gridGroups, gridGroups);
        this.ping ^= 1;
      }
    }

    // 5. hair solver
    pass.setPipeline(this.simPipelines[params.algorithm]);
    pass.setBindGroup(0, this.simBindGroups[params.algorithm][this.ping]);
    pass.dispatchWorkgroups(particleGroups);

    pass.end();
  }
}

// ---------------------------------------------------------------- initial particle state

interface InitialState {
  particles: Float32Array;
  strandLengths: Float32Array;
  strandLengthsReal: Float32Array;
  roots: Float32Array;
  hairIndices: Uint32Array;
}

/**
 * Port of the ofApp::setup() particle loop: a random length per strand, particles
 * spaced along the vertex normal, root fixed. Colours are either the checked-in
 * alternating scheme or the video's root -> tip gradient.
 */
function buildInitialState(
  model: HairModel,
  frame0: SkinnedSurface,
  cfg: EngineConfig,
  paddedStrands: number,
  colorMode: ColorMode
): InitialState {
  const perStrand = cfg.particlesPerStrand;
  const numStrands = model.vertexCount;
  const paddedParticles = paddedStrands * perStrand;

  const particles = new Float32Array(paddedParticles * (PARTICLE_STRIDE / 4));
  const strandLengths = new Float32Array(paddedStrands);
  const strandLengthsReal = new Float32Array(numStrands);
  const roots = new Float32Array(paddedStrands * 4);
  const hairIndices = new Uint32Array(numStrands * (perStrand + 1));

  const rng = makeRng(cfg.seed);
  const lengthRange = cfg.maxHairLength - cfg.minHairLength;

  const gradient: [number, number, number] = [0, 0, 0];
  const useGradient = colorMode === 'gradient';
  const lastIndex = Math.max(1, perStrand - 1);

  let ii = 0;
  for (let i = 0; i < numStrands; i++) {
    const hairLength = cfg.minHairLength + rng() * lengthRange;
    const color = STRAND_COLORS[i % STRAND_COLORS.length];

    const vx = frame0.positions[i * 3 + 0];
    const vy = frame0.positions[i * 3 + 1];
    const vz = frame0.positions[i * 3 + 2];
    const nx = frame0.normals[i * 3 + 0];
    const ny = frame0.normals[i * 3 + 1];
    const nz = frame0.normals[i * 3 + 2];
    const step = hairLength / perStrand;

    for (let j = 0; j < perStrand; j++) {
      const p = (i * perStrand + j) * 12;
      const x = vx + nx * step * j;
      const y = vy + ny * step * j;
      const z = vz + nz * step * j;
      particles[p + 0] = x;
      particles[p + 1] = y;
      particles[p + 2] = z;
      particles[p + 3] = 1;
      particles[p + 4] = x; // prevPos = pos
      particles[p + 5] = y;
      particles[p + 6] = z;
      particles[p + 7] = 1;
      if (useGradient) {
        gradientColor(j / lastIndex, gradient);
        particles[p + 8] = gradient[0];
        particles[p + 9] = gradient[1];
        particles[p + 10] = gradient[2];
      } else {
        particles[p + 8] = color[0];
        particles[p + 9] = color[1];
        particles[p + 10] = color[2];
      }
      particles[p + 11] = j === 0 ? 1 : 0; // fix flag lives in color.w
      hairIndices[ii++] = i * perStrand + j;
    }
    hairIndices[ii++] = RESTART_INDEX;

    strandLengths[i] = hairLength;
    strandLengthsReal[i] = hairLength;
  }

  // padding strands: every particle fixed and parked outside the grid
  for (let i = numStrands; i < paddedStrands; i++) {
    strandLengths[i] = cfg.minHairLength;
    for (let j = 0; j < perStrand; j++) {
      const p = (i * perStrand + j) * 12;
      particles[p + 0] = PARKED;
      particles[p + 1] = PARKED;
      particles[p + 2] = PARKED;
      particles[p + 3] = 1;
      particles[p + 4] = PARKED;
      particles[p + 5] = PARKED;
      particles[p + 6] = PARKED;
      particles[p + 7] = 1;
      particles[p + 11] = 1; // fixed
    }
  }

  // roots: real entries are overwritten by the skinning pass every frame, the padded
  // tail (which the dispatch does not cover) keeps the parked position forever.
  for (let i = 0; i < paddedStrands; i++) {
    const o = i * 4;
    const real = i < numStrands;
    roots[o + 0] = real ? 0 : PARKED;
    roots[o + 1] = real ? 0 : PARKED;
    roots[o + 2] = real ? 0 : PARKED;
    roots[o + 3] = 1;
  }

  return { particles, strandLengths, strandLengthsReal, roots, hairIndices };
}

// ---------------------------------------------------------------- mat4 inverse (column-major)

function invertMat4(m: ArrayLike<number>): Float32Array {
  const out = new Float32Array(16);
  invertMat4Into(m, out);
  return out;
}

function invertMat4Into(m: ArrayLike<number>, out: Float32Array): void {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) {
    out.fill(0);
    out[0] = 1;
    out[5] = 1;
    out[10] = 1;
    out[15] = 1;
    return;
  }
  det = 1.0 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
}
