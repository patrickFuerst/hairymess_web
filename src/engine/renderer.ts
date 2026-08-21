// Everything is drawn in a single 4x MSAA render pass, in the original's order:
// background gradient -> shaded body mesh -> hair -> floor stencil mask ->
// mirrored hair reflection -> optional voxel grid, bounding box and collider overlays.
//
// The hair has two interchangeable paths (params.renderMode):
//   'ribbons' — 6 vertices per strand segment, pulled out of the particle storage
//               buffer in the vertex stage, expanded to a camera-facing quad of
//               constant pixel width and shaded with a Kajiya-Kay model. Default.
//   'lines'   — the original 1px line strips. Also the automatic fallback on devices
//               that will not grant maxStorageBuffersInVertexStage >= 1.

import type { SimParams } from '../params';
import type { OrbitCamera } from './camera';
import type { EngineConfig } from './config';
import { createShaderModule } from './device';
import type { HairSim } from './hairSim';
import { BG_U, PARTICLE_SHADE_OFFSET, PARTICLE_STRIDE, RENDER_U, UniformScratch } from './layout';
import { mat4Identity } from './model';
import { buildShaders } from './shaders';

const SAMPLE_COUNT = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus-stencil8';
const STENCIL_REF = 1;

/** ofColor::lightGray -> ofColor::white, the circular background gradient */
const BG_CENTER: [number, number, number] = [211 / 255, 211 / 255, 211 / 255];
const BG_EDGE: [number, number, number] = [1, 1, 1];
/** ofFloatColor::whiteSmoke, the reflection tint */
const WHITE_SMOKE: [number, number, number] = [245 / 255, 245 / 255, 245 / 255];
/**
 * 60 x 60 floor plane, rotated 45 degrees about Y. The original (and the demo video)
 * used 30 x 30; doubling it lets the mirrored fur spread over far more ground, which
 * is what `reflectionFade()` in common.wgsl was re-tuned for.
 */
const FLOOR_HALF_SIZE = 30;
/** vertices colliders.wgsl emits per capsule: 3 rings x 24 segments x 2 + 4 axial x 2 */
const COLLIDER_VERTS = 24 * 2 * 3 + 8;

const ALPHA_BLEND: GPUBlendState = {
  color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
};

interface Stage {
  pipeline: GPURenderPipeline;
  bindGroup: GPUBindGroup;
  uniform: GPUBuffer;
}

interface UniformValues {
  viewProj: Float32Array;
  model: Float32Array;
  color: readonly [number, number, number];
  eye: Float32Array;
  alpha?: number;
  shading?: boolean;
  strandWidth?: number;
  fade?: boolean;
}

export class Renderer {
  private readonly device: GPUDevice;
  private readonly format: GPUTextureFormat;
  private readonly sim: HairSim;
  private readonly cfg: EngineConfig;

  private msaaTexture: GPUTexture | null = null;
  private depthTexture: GPUTexture | null = null;
  private msaaView: GPUTextureView | null = null;
  private depthView: GPUTextureView | null = null;
  private width = 0;
  private height = 0;

  private readonly background: Stage;
  private readonly mesh: Stage;
  private readonly hairLines: Stage;
  private readonly reflectionLines: Stage;
  private readonly hairRibbons: Stage | null;
  private readonly reflectionRibbons: Stage | null;
  private readonly floor: Stage;
  private readonly voxel: Stage;
  private readonly bbox: Stage;
  private readonly colliders: Stage;

  private readonly floorVertices: GPUBuffer;
  private readonly scratch = new UniformScratch(RENDER_U.SIZE);
  private readonly bgScratch = new UniformScratch(BG_U.SIZE);
  private readonly identity = mat4Identity();
  private readonly mirror: Float32Array;
  /** 6 vertices per segment, (particlesPerStrand - 1) segments per real strand */
  private readonly ribbonVertexCount: number;

  /** false when the device would not grant a vertex-stage storage buffer */
  readonly ribbonsAvailable: boolean;

  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    sim: HairSim,
    cfg: EngineConfig,
    canPullInVertexStage: boolean,
    onShaderError?: (m: string) => void
  ) {
    this.device = device;
    this.format = format;
    this.sim = sim;
    this.cfg = cfg;
    this.ribbonsAvailable = canPullInVertexStage;
    this.ribbonVertexCount = 6 * (cfg.particlesPerStrand - 1) * sim.numStrands;

    this.mirror = mat4Identity();
    this.mirror[5] = -1; // scale(1,-1,1)

    const src = buildShaders(cfg);
    const moduleCache = new Map<string, GPUShaderModule>();
    const mod = (name: keyof typeof src): GPUShaderModule => {
      let m = moduleCache.get(name);
      if (!m) {
        m = createShaderModule(device, name, src[name], onShaderError);
        moduleCache.set(name, m);
      }
      return m;
    };

    const uniformLayout = device.createBindGroupLayout({
      label: 'renderUniformLayout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
      ],
    });
    const uniformPlusSimLayout = device.createBindGroupLayout({
      label: 'renderUniformPlusSimLayout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });
    // ribbons pull the particle buffer straight out of the vertex stage
    const uniformPlusParticlesLayout = device.createBindGroupLayout({
      label: 'renderRibbonLayout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    const makeUniform = (label: string, size: number): GPUBuffer =>
      device.createBuffer({
        label,
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

    const simpleGroup = (label: string, uniform: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        label,
        layout: uniformLayout,
        entries: [{ binding: 0, resource: { buffer: uniform } }],
      });

    const withSimGroup = (label: string, uniform: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        label,
        layout: uniformPlusSimLayout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: sim.simUniformBuffer } },
        ],
      });

    const withParticlesGroup = (label: string, uniform: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        label,
        layout: uniformPlusParticlesLayout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: sim.particles } },
        ],
      });

    // position vec4f @0 + skinned normal vec4f @16 (the floor quad pads a flat normal)
    const meshVertexLayout: GPUVertexBufferLayout = {
      arrayStride: 32,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x4' },
      ],
    };
    const hairVertexLayout: GPUVertexBufferLayout = {
      arrayStride: PARTICLE_STRIDE,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos
        { shaderLocation: 1, offset: 32, format: 'float32x4' }, // color (w = fix flag)
        { shaderLocation: 2, offset: PARTICLE_SHADE_OFFSET, format: 'float32' }, // prevPos.w
      ],
    };

    const depthOn = (write: boolean): GPUDepthStencilState => ({
      format: DEPTH_FORMAT,
      depthWriteEnabled: write,
      depthCompare: 'less',
    });
    /** the mirrored passes are stencil-masked by the floor quad */
    const reflectionStencil: GPUDepthStencilState = {
      format: DEPTH_FORMAT,
      depthWriteEnabled: true,
      depthCompare: 'less',
      stencilFront: { compare: 'equal', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
      stencilBack: { compare: 'equal', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
      stencilReadMask: 0xff,
      stencilWriteMask: 0x00, // glStencilMask(0x00)
    };

    // ------------------------------------------------------------ background
    {
      const uniform = makeUniform('bgUniforms', BG_U.SIZE);
      const module = mod('background');
      this.background = {
        uniform,
        bindGroup: simpleGroup('bgBindGroup', uniform),
        pipeline: device.createRenderPipeline({
          label: 'background',
          layout: device.createPipelineLayout({ bindGroupLayouts: [uniformLayout] }),
          vertex: { module, entryPoint: 'vs' },
          fragment: { module, entryPoint: 'fs', targets: [{ format }] },
          primitive: { topology: 'triangle-list' },
          depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
          multisample: { count: SAMPLE_COUNT },
        }),
      };
    }

    // ------------------------------------------------------------ skinned body mesh
    {
      const uniform = makeUniform('meshUniforms', RENDER_U.SIZE);
      const module = mod('mesh');
      this.mesh = {
        uniform,
        bindGroup: simpleGroup('meshBindGroup', uniform),
        pipeline: device.createRenderPipeline({
          label: 'mesh',
          layout: device.createPipelineLayout({ bindGroupLayouts: [uniformLayout] }),
          vertex: { module, entryPoint: 'vs', buffers: [meshVertexLayout] },
          fragment: { module, entryPoint: 'fs', targets: [{ format }] },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil: depthOn(true),
          multisample: { count: SAMPLE_COUNT },
        }),
      };
    }

    // ------------------------------------------------------------ hair, line strips
    {
      const module = mod('hair');
      const makeLineStage = (label: string, depthStencil: GPUDepthStencilState): Stage => {
        const uniform = makeUniform(`${label}Uniforms`, RENDER_U.SIZE);
        return {
          uniform,
          bindGroup: simpleGroup(`${label}BindGroup`, uniform),
          pipeline: device.createRenderPipeline({
            label,
            layout: device.createPipelineLayout({ bindGroupLayouts: [uniformLayout] }),
            vertex: { module, entryPoint: 'vs', buffers: [hairVertexLayout] },
            fragment: { module, entryPoint: 'fs', targets: [{ format, blend: ALPHA_BLEND }] },
            primitive: { topology: 'line-strip', stripIndexFormat: 'uint32' },
            depthStencil,
            multisample: { count: SAMPLE_COUNT },
          }),
        };
      };
      this.hairLines = makeLineStage('hairLines', depthOn(true));
      this.reflectionLines = makeLineStage('reflectionLines', reflectionStencil);
    }

    // ------------------------------------------------------------ hair, ribbons
    if (this.ribbonsAvailable) {
      const module = mod('ribbon');
      const makeRibbonStage = (label: string, depthStencil: GPUDepthStencilState): Stage => {
        const uniform = makeUniform(`${label}Uniforms`, RENDER_U.SIZE);
        return {
          uniform,
          bindGroup: withParticlesGroup(`${label}BindGroup`, uniform),
          pipeline: device.createRenderPipeline({
            label,
            layout: device.createPipelineLayout({
              bindGroupLayouts: [uniformPlusParticlesLayout],
            }),
            vertex: { module, entryPoint: 'vs' },
            fragment: { module, entryPoint: 'fs', targets: [{ format, blend: ALPHA_BLEND }] },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            depthStencil,
            multisample: { count: SAMPLE_COUNT },
          }),
        };
      };
      this.hairRibbons = makeRibbonStage('hairRibbons', depthOn(true));
      this.reflectionRibbons = makeRibbonStage('reflectionRibbons', reflectionStencil);
    } else {
      this.hairRibbons = null;
      this.reflectionRibbons = null;
    }

    // ------------------------------------------------------------ floor stencil mask
    {
      const uniform = makeUniform('floorUniforms', RENDER_U.SIZE);
      const module = mod('mesh'); // position-only, colour is masked out anyway
      this.floor = {
        uniform,
        bindGroup: simpleGroup('floorBindGroup', uniform),
        pipeline: device.createRenderPipeline({
          label: 'floorStencil',
          layout: device.createPipelineLayout({ bindGroupLayouts: [uniformLayout] }),
          vertex: { module, entryPoint: 'vs', buffers: [meshVertexLayout] },
          fragment: {
            module,
            entryPoint: 'fs',
            targets: [{ format, writeMask: 0 }], // glDepthMask(FALSE) + colour masked
          },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil: {
            format: DEPTH_FORMAT,
            depthWriteEnabled: false,
            depthCompare: 'less',
            stencilFront: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'replace' },
            stencilBack: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'replace' },
            stencilReadMask: 0xff,
            stencilWriteMask: 0xff,
          },
          multisample: { count: SAMPLE_COUNT },
        }),
      };
    }

    // ------------------------------------------------------------ voxel grid points
    {
      const uniform = makeUniform('voxelUniforms', RENDER_U.SIZE);
      const module = mod('voxelDebug');
      this.voxel = {
        uniform,
        bindGroup: withSimGroup('voxelBindGroup', uniform),
        pipeline: device.createRenderPipeline({
          label: 'voxelDebug',
          layout: device.createPipelineLayout({ bindGroupLayouts: [uniformPlusSimLayout] }),
          vertex: {
            module,
            entryPoint: 'vs',
            buffers: [
              { arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] },
              { arrayStride: 16, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }] },
              { arrayStride: 4, attributes: [{ shaderLocation: 2, offset: 0, format: 'sint32' }] },
            ],
          },
          fragment: { module, entryPoint: 'fs', targets: [{ format, blend: ALPHA_BLEND }] },
          primitive: { topology: 'point-list' },
          depthStencil: depthOn(true),
          multisample: { count: SAMPLE_COUNT },
        }),
      };
    }

    // ------------------------------------------------------------ bounding box
    {
      const uniform = makeUniform('bboxUniforms', RENDER_U.SIZE);
      const module = mod('bbox');
      this.bbox = {
        uniform,
        bindGroup: withSimGroup('bboxBindGroup', uniform),
        pipeline: device.createRenderPipeline({
          label: 'bbox',
          layout: device.createPipelineLayout({ bindGroupLayouts: [uniformPlusSimLayout] }),
          vertex: { module, entryPoint: 'vs' },
          fragment: { module, entryPoint: 'fs', targets: [{ format }] },
          primitive: { topology: 'line-list' },
          depthStencil: depthOn(true),
          multisample: { count: SAMPLE_COUNT },
        }),
      };
    }

    // ------------------------------------------------------------ collider wireframe
    {
      const uniform = makeUniform('colliderUniforms', RENDER_U.SIZE);
      const module = mod('colliders');
      this.colliders = {
        uniform,
        bindGroup: withSimGroup('colliderBindGroup', uniform),
        pipeline: device.createRenderPipeline({
          label: 'colliders',
          layout: device.createPipelineLayout({ bindGroupLayouts: [uniformPlusSimLayout] }),
          vertex: { module, entryPoint: 'vs' },
          fragment: { module, entryPoint: 'fs', targets: [{ format }] },
          primitive: { topology: 'line-list' },
          // an overlay: the point of the view is seeing where a capsule sits even when
          // it is buried in fur, so it ignores depth entirely
          depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
          multisample: { count: SAMPLE_COUNT },
        }),
      };
    }

    // ------------------------------------------------------------ floor geometry
    // ofMesh::plane(60,60) laid into the XZ plane and rotated 45 degrees about Y.
    // rotating (+-h, +-h) by 45 degrees turns the square into a diamond of radius h*sqrt(2)
    const r = FLOOR_HALF_SIZE * Math.SQRT2;
    const corners: Array<[number, number]> = [
      [r, 0],
      [0, -r],
      [-r, 0],
      [0, r],
    ];
    const quad = new Float32Array(6 * 8); // stride 32: position + a flat normal
    const order = [0, 1, 2, 0, 2, 3];
    order.forEach((corner, i) => {
      quad[i * 8 + 0] = corners[corner][0];
      quad[i * 8 + 1] = 0;
      quad[i * 8 + 2] = corners[corner][1];
      quad[i * 8 + 3] = 1;
      quad[i * 8 + 5] = 1; // normal (0,1,0)
    });
    this.floorVertices = device.createBuffer({
      label: 'floorQuad',
      size: quad.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.floorVertices, 0, quad);
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height && this.msaaView && this.depthView) return;
    this.width = width;
    this.height = height;
    this.msaaTexture?.destroy();
    this.depthTexture?.destroy();

    this.msaaTexture = this.device.createTexture({
      label: 'msaaColor',
      size: [width, height],
      format: this.format,
      sampleCount: SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthTexture = this.device.createTexture({
      label: 'depthStencil',
      size: [width, height],
      format: DEPTH_FORMAT,
      sampleCount: SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.msaaView = this.msaaTexture.createView();
    this.depthView = this.depthTexture.createView();
  }

  private writeRenderUniforms(stage: Stage, v: UniformValues): void {
    const s = this.scratch;
    s.setMat4(RENDER_U.viewProj, v.viewProj);
    s.setMat4(RENDER_U.model, v.model);
    s.setVec4(RENDER_U.overrideColor, v.color[0], v.color[1], v.color[2], v.alpha ?? 1);
    s.setVec4(RENDER_U.cameraPos, v.eye[0], v.eye[1], v.eye[2], 1);
    s.setF32(RENDER_U.canvasWidth, this.width);
    s.setF32(RENDER_U.canvasHeight, this.height);
    s.setF32(RENDER_U.strandWidth, v.strandWidth ?? 1);
    s.setF32(RENDER_U.shading, v.shading ? 1 : 0);
    s.setF32(RENDER_U.fade, v.fade ? 1 : 0);
    this.device.queue.writeBuffer(stage.uniform, 0, s.bytes);
  }

  /** the hair path actually in use, after the vertex-storage capability check */
  private useRibbons(params: SimParams): boolean {
    return params.renderMode === 'ribbons' && this.hairRibbons !== null;
  }

  render(
    encoder: GPUCommandEncoder,
    canvasView: GPUTextureView,
    camera: OrbitCamera,
    params: SimParams,
    timestampWrites?: GPURenderPassTimestampWrites
  ): void {
    if (!this.msaaView || !this.depthView) throw new Error('Renderer.resize() must run before render()');

    const aspect = this.height > 0 ? this.width / this.height : 1;
    const viewProj = camera.viewProjectionMatrix(aspect) as Float32Array;
    const eye = camera.position;
    const ribbons = this.useRibbons(params);
    const hair = ribbons ? this.hairRibbons! : this.hairLines;
    const reflection = ribbons ? this.reflectionRibbons! : this.reflectionLines;

    // --- per frame uniform writes (no buffer or bind group creation here)
    const bg = this.bgScratch;
    bg.setVec4(BG_U.centerColor, BG_CENTER[0], BG_CENTER[1], BG_CENTER[2], 1);
    bg.setVec4(BG_U.edgeColor, BG_EDGE[0], BG_EDGE[1], BG_EDGE[2], 1);
    bg.setF32(BG_U.resolution, this.width);
    bg.setF32(BG_U.resolution + 4, this.height);
    this.device.queue.writeBuffer(this.background.uniform, 0, bg.bytes);

    const common = { viewProj, eye, shading: params.shading, strandWidth: params.strandWidth };
    if (params.drawFur) {
      // the mesh's shaded path uses its own base colour; this is the original's flat
      // black, which it falls back to when shading is off
      this.writeRenderUniforms(this.mesh, {
        ...common,
        model: this.sim.model.modelMatrix,
        color: [0, 0, 0],
      });
      this.writeRenderUniforms(hair, { ...common, model: this.identity, color: [1, 1, 1] });
      this.writeRenderUniforms(this.floor, {
        ...common,
        model: this.identity,
        color: [1, 1, 1],
      });
      this.writeRenderUniforms(reflection, {
        ...common,
        model: this.mirror,
        color: WHITE_SMOKE,
        fade: true,
      });
    }
    if (params.drawVoxelGrid) {
      this.writeRenderUniforms(this.voxel, { ...common, model: this.identity, color: [1, 1, 1] });
    }
    if (params.drawBoundingBox) {
      this.writeRenderUniforms(this.bbox, { ...common, model: this.identity, color: [1, 0, 0] });
    }
    if (params.drawColliders) {
      this.writeRenderUniforms(this.colliders, {
        ...common,
        model: this.identity,
        color: [0, 0.55, 1],
      });
    }

    const pass = encoder.beginRenderPass({
      label: 'render',
      colorAttachments: [
        {
          view: this.msaaView,
          resolveTarget: canvasView,
          clearValue: { r: 1, g: 1, b: 1, a: 1 },
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
        stencilClearValue: 0,
        stencilLoadOp: 'clear',
        stencilStoreOp: 'discard',
      },
      ...(timestampWrites ? { timestampWrites } : {}),
    });
    pass.setStencilReference(STENCIL_REF);

    // 1. circular background gradient
    pass.setPipeline(this.background.pipeline);
    pass.setBindGroup(0, this.background.bindGroup);
    pass.draw(3);

    if (params.drawFur) {
      // 2. skinned body
      pass.setPipeline(this.mesh.pipeline);
      pass.setBindGroup(0, this.mesh.bindGroup);
      pass.setVertexBuffer(0, this.sim.skinnedVerts);
      pass.setIndexBuffer(this.sim.meshIndexBuffer, 'uint32');
      pass.drawIndexed(this.sim.meshIndexCount);

      // 3. hair straight out of the simulation buffer
      this.drawHair(pass, hair, ribbons);

      // 4. floor quad -> stencil = 1 (no colour, no depth write)
      pass.setPipeline(this.floor.pipeline);
      pass.setBindGroup(0, this.floor.bindGroup);
      pass.setVertexBuffer(0, this.floorVertices);
      pass.draw(6);

      // 5. same hair mirrored through y = 0, masked by the stencil and faded out
      this.drawHair(pass, reflection, ribbons);
    }

    // 6. voxel grid debug points
    if (params.drawVoxelGrid) {
      const ping = this.sim.readIndex;
      pass.setPipeline(this.voxel.pipeline);
      pass.setBindGroup(0, this.voxel.bindGroup);
      pass.setVertexBuffer(0, this.sim.velocityGrid[ping]);
      pass.setVertexBuffer(1, this.sim.gradientGrid[ping]);
      pass.setVertexBuffer(2, this.sim.densityAtomic);
      pass.draw(this.cfg.gridSize ** 3);
    }

    // 7. bounding box wireframe
    if (params.drawBoundingBox) {
      pass.setPipeline(this.bbox.pipeline);
      pass.setBindGroup(0, this.bbox.bindGroup);
      pass.draw(24);
    }

    // 8. capsule collider wireframes
    if (params.drawColliders && this.sim.activeColliders > 0) {
      pass.setPipeline(this.colliders.pipeline);
      pass.setBindGroup(0, this.colliders.bindGroup);
      pass.draw(COLLIDER_VERTS * this.sim.activeColliders);
    }

    pass.end();
  }

  private drawHair(pass: GPURenderPassEncoder, stage: Stage, ribbons: boolean): void {
    pass.setPipeline(stage.pipeline);
    pass.setBindGroup(0, stage.bindGroup);
    if (ribbons) {
      pass.draw(this.ribbonVertexCount);
      return;
    }
    pass.setVertexBuffer(0, this.sim.particles);
    pass.setIndexBuffer(this.sim.hairIndexBuffer, 'uint32');
    pass.drawIndexed(this.sim.hairIndexCount);
  }

  destroy(): void {
    this.msaaTexture?.destroy();
    this.depthTexture?.destroy();
    this.msaaTexture = null;
    this.depthTexture = null;
    this.msaaView = null;
    this.depthView = null;
  }
}
