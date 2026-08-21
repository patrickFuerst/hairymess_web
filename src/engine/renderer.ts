// Everything is drawn in a single 4x MSAA render pass, in the original's order:
// background gradient -> black skinned mesh -> hair -> floor stencil mask ->
// mirrored hair reflection -> optional voxel grid points and bounding box.

import type { SimParams } from '../params';
import type { OrbitCamera } from './camera';
import type { EngineConfig } from './config';
import { createShaderModule } from './device';
import type { HairSim } from './hairSim';
import { BG_U, PARTICLE_STRIDE, RENDER_U, UniformScratch } from './layout';
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
/** 30 x 30 floor plane, rotated 45 degrees about Y */
const FLOOR_HALF_SIZE = 15;

const ALPHA_BLEND: GPUBlendState = {
  color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
};

interface Stage {
  pipeline: GPURenderPipeline;
  bindGroup: GPUBindGroup;
  uniform: GPUBuffer;
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
  private readonly hair: Stage;
  private readonly floor: Stage;
  private readonly reflection: Stage;
  private readonly voxel: Stage;
  private readonly bbox: Stage;

  private readonly floorVertices: GPUBuffer;
  private readonly scratch = new UniformScratch(RENDER_U.SIZE);
  private readonly bgScratch = new UniformScratch(BG_U.SIZE);
  private readonly identity = mat4Identity();
  private readonly mirror: Float32Array;

  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    sim: HairSim,
    cfg: EngineConfig,
    onShaderError?: (m: string) => void
  ) {
    this.device = device;
    this.format = format;
    this.sim = sim;
    this.cfg = cfg;

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

    const meshVertexLayout: GPUVertexBufferLayout = {
      arrayStride: 16,
      attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }],
    };
    const hairVertexLayout: GPUVertexBufferLayout = {
      arrayStride: PARTICLE_STRIDE,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos
        { shaderLocation: 1, offset: 32, format: 'float32x4' }, // color (w = fix flag)
      ],
    };

    const depthOn = (write: boolean): GPUDepthStencilState => ({
      format: DEPTH_FORMAT,
      depthWriteEnabled: write,
      depthCompare: 'less',
    });

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

    // ------------------------------------------------------------ skinned mesh
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

    // ------------------------------------------------------------ hair
    {
      const uniform = makeUniform('hairUniforms', RENDER_U.SIZE);
      const module = mod('hair');
      this.hair = {
        uniform,
        bindGroup: simpleGroup('hairBindGroup', uniform),
        pipeline: device.createRenderPipeline({
          label: 'hair',
          layout: device.createPipelineLayout({ bindGroupLayouts: [uniformLayout] }),
          vertex: { module, entryPoint: 'vs', buffers: [hairVertexLayout] },
          fragment: { module, entryPoint: 'fs', targets: [{ format, blend: ALPHA_BLEND }] },
          primitive: { topology: 'line-strip', stripIndexFormat: 'uint32' },
          depthStencil: depthOn(true),
          multisample: { count: SAMPLE_COUNT },
        }),
      };
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

    // ------------------------------------------------------------ mirrored hair
    {
      const uniform = makeUniform('reflectionUniforms', RENDER_U.SIZE);
      const module = mod('floorFade');
      this.reflection = {
        uniform,
        bindGroup: simpleGroup('reflectionBindGroup', uniform),
        pipeline: device.createRenderPipeline({
          label: 'reflection',
          layout: device.createPipelineLayout({ bindGroupLayouts: [uniformLayout] }),
          vertex: { module, entryPoint: 'vs', buffers: [hairVertexLayout] },
          fragment: { module, entryPoint: 'fs', targets: [{ format, blend: ALPHA_BLEND }] },
          primitive: { topology: 'line-strip', stripIndexFormat: 'uint32' },
          depthStencil: {
            format: DEPTH_FORMAT,
            depthWriteEnabled: true,
            depthCompare: 'less',
            stencilFront: { compare: 'equal', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
            stencilBack: { compare: 'equal', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
            stencilReadMask: 0xff,
            stencilWriteMask: 0x00, // glStencilMask(0x00)
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

    // ------------------------------------------------------------ floor geometry
    // ofMesh::plane(30,30) laid into the XZ plane and rotated 45 degrees about Y.
    // rotating (+-h, +-h) by 45 degrees turns the square into a diamond of radius h*sqrt(2)
    const r = FLOOR_HALF_SIZE * Math.SQRT2;
    const corners: Array<[number, number]> = [
      [r, 0],
      [0, -r],
      [-r, 0],
      [0, r],
    ];
    const quad = new Float32Array(6 * 4);
    const order = [0, 1, 2, 0, 2, 3];
    order.forEach((corner, i) => {
      quad[i * 4 + 0] = corners[corner][0];
      quad[i * 4 + 1] = 0;
      quad[i * 4 + 2] = corners[corner][1];
      quad[i * 4 + 3] = 1;
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

  private writeRenderUniforms(
    stage: Stage,
    viewProj: Float32Array,
    model: Float32Array,
    color: readonly [number, number, number],
    alpha = 1
  ): void {
    const s = this.scratch;
    s.setMat4(RENDER_U.viewProj, viewProj);
    s.setMat4(RENDER_U.model, model);
    s.setVec4(RENDER_U.overrideColor, color[0], color[1], color[2], alpha);
    s.setF32(RENDER_U.canvasHeight, this.height);
    this.device.queue.writeBuffer(stage.uniform, 0, s.bytes);
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

    // --- per frame uniform writes (no buffer or bind group creation here)
    const bg = this.bgScratch;
    bg.setVec4(BG_U.centerColor, BG_CENTER[0], BG_CENTER[1], BG_CENTER[2], 1);
    bg.setVec4(BG_U.edgeColor, BG_EDGE[0], BG_EDGE[1], BG_EDGE[2], 1);
    bg.setF32(BG_U.resolution, this.width);
    bg.setF32(BG_U.resolution + 4, this.height);
    this.device.queue.writeBuffer(this.background.uniform, 0, bg.bytes);

    if (params.drawFur) {
      this.writeRenderUniforms(this.mesh, viewProj, this.sim.model.modelMatrix, [0, 0, 0]);
      this.writeRenderUniforms(this.hair, viewProj, this.identity, [1, 1, 1]);
      this.writeRenderUniforms(this.floor, viewProj, this.identity, [1, 1, 1]);
      this.writeRenderUniforms(this.reflection, viewProj, this.mirror, WHITE_SMOKE);
    }
    if (params.drawVoxelGrid) {
      this.writeRenderUniforms(this.voxel, viewProj, this.identity, [1, 1, 1]);
    }
    if (params.drawBoundingBox) {
      this.writeRenderUniforms(this.bbox, viewProj, this.identity, [1, 0, 0]);
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
      // 2. skinned mesh, solid black
      pass.setPipeline(this.mesh.pipeline);
      pass.setBindGroup(0, this.mesh.bindGroup);
      pass.setVertexBuffer(0, this.sim.skinnedPositions);
      pass.setIndexBuffer(this.sim.meshIndexBuffer, 'uint32');
      pass.drawIndexed(this.sim.meshIndexCount);

      // 3. hair line strips straight out of the simulation buffer
      pass.setPipeline(this.hair.pipeline);
      pass.setBindGroup(0, this.hair.bindGroup);
      pass.setVertexBuffer(0, this.sim.particles);
      pass.setIndexBuffer(this.sim.hairIndexBuffer, 'uint32');
      pass.drawIndexed(this.sim.hairIndexCount);

      // 4. floor quad -> stencil = 1 (no colour, no depth write)
      pass.setPipeline(this.floor.pipeline);
      pass.setBindGroup(0, this.floor.bindGroup);
      pass.setVertexBuffer(0, this.floorVertices);
      pass.draw(6);

      // 5. same hair mirrored through y = 0, masked by the stencil and faded out
      pass.setPipeline(this.reflection.pipeline);
      pass.setBindGroup(0, this.reflection.bindGroup);
      pass.setVertexBuffer(0, this.sim.particles);
      pass.setIndexBuffer(this.sim.hairIndexBuffer, 'uint32');
      pass.drawIndexed(this.sim.hairIndexCount);
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

    pass.end();
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
