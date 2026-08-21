// Byte-for-byte mirrors of the WGSL uniform structs in src/shaders/common.wgsl and
// friends. WGSL rules used here: mat4x4f align 16 / size 64, vec4f align 16 / size 16,
// vec2f align 8, f32-i32-u32 align 4, struct size rounded up to its alignment (16).
// Every SIZE below is a multiple of 16.

export const PARTICLE_STRIDE = 48; // pos vec4f @0, prevPos vec4f @16, color vec4f @32

// struct SimUniforms — 240 bytes
export const SIM_U = {
  modelMatrix: 0, // mat4x4f
  modelMatrixPrevInverted: 64, // mat4x4f
  modelTranslation: 128, // vec4f
  gravity: 144, // vec4f
  minBB: 160, // vec4f
  maxBB: 176, // vec4f
  velocityDamping: 192, // f32
  numIterationsPBD: 196, // i32
  stiffness: 200, // f32
  friction: 204, // f32
  repulsion: 208, // f32
  ftlDamping: 212, // f32
  deltaTime: 216, // f32
  gridSize: 220, // i32
  numVerticesPerStrand: 224, // u32
  numStrandsPerThreadGroup: 228, // u32
  numStrands: 232, // u32
  pad0: 236, // u32
  SIZE: 240,
} as const;

// struct RenderUniforms — 160 bytes
export const RENDER_U = {
  viewProj: 0, // mat4x4f
  model: 64, // mat4x4f
  overrideColor: 128, // vec4f
  canvasHeight: 144, // f32
  SIZE: 160,
} as const;

// struct BgUniforms — 48 bytes
export const BG_U = {
  centerColor: 0, // vec4f
  edgeColor: 16, // vec4f
  resolution: 32, // vec2f
  SIZE: 48,
} as const;

// struct SkinUniforms — 16 bytes
export const SKIN_U = {
  numVerts: 0, // u32
  numJoints: 4, // u32
  SIZE: 16,
} as const;

// struct FilterUniforms — 16 bytes
export const FILTER_U = {
  axis: 0, // i32
  SIZE: 16,
} as const;

/**
 * TypeScript 5.7+ types typed arrays as `Float32Array<ArrayBufferLike>` while
 * @webgpu/types wants an `ArrayBuffer`-backed view. Narrow once here instead of at
 * every upload site.
 */
export function gpuWrite(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: ArrayBufferView | ArrayBuffer,
  bufferOffset = 0
): void {
  device.queue.writeBuffer(buffer, bufferOffset, data as GPUAllowSharedBufferSource);
}

// Small typed scratch buffer with all three views over the same bytes.
export class UniformScratch {
  readonly bytes: ArrayBuffer;
  readonly f32: Float32Array;
  readonly i32: Int32Array;
  readonly u32: Uint32Array;

  constructor(byteLength: number) {
    if (byteLength % 16 !== 0) {
      throw new Error(`uniform size ${byteLength} must be a multiple of 16`);
    }
    this.bytes = new ArrayBuffer(byteLength);
    this.f32 = new Float32Array(this.bytes);
    this.i32 = new Int32Array(this.bytes);
    this.u32 = new Uint32Array(this.bytes);
  }

  setMat4(offset: number, m: ArrayLike<number>): void {
    this.f32.set(m as Float32Array, offset / 4);
  }

  setVec4(offset: number, x: number, y: number, z: number, w: number): void {
    const i = offset / 4;
    this.f32[i] = x;
    this.f32[i + 1] = y;
    this.f32[i + 2] = z;
    this.f32[i + 3] = w;
  }

  setF32(offset: number, v: number): void {
    this.f32[offset / 4] = v;
  }

  setI32(offset: number, v: number): void {
    this.i32[offset / 4] = v;
  }

  setU32(offset: number, v: number): void {
    this.u32[offset / 4] = v;
  }
}
