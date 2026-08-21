// Model loading (public/models/beast.json + beast.bin, baked by tools/bake_dae.py)
// plus the procedural UV-sphere fallback ("Furry Ball"). Both feed exactly the same
// GPU code path: bind pose + 4 joints/weights per vertex + per-frame joint palettes.

import { MAX_COLLIDERS } from '../params';

/**
 * SPEC v2 `"colliders"`: an auto-fitted capsule in bind/model space, skinned with
 * weight 1 to its two joints at runtime. A sphere is a degenerate capsule (a == b).
 */
export interface ModelCollider {
  jointA: number;
  headA: [number, number, number];
  jointB: number;
  headB: [number, number, number];
  radius: number;
}

export interface HairModel {
  name: string;
  vertexCount: number;
  jointCount: number;
  frameCount: number;
  duration: number; // seconds for one animation loop
  /** model space -> sim world space, column-major */
  modelMatrix: Float32Array;
  /**
   * Where the model is *placed* in the world — the origin of the (model-local) voxel
   * grid, i.e. the original's `g_modelTranslation`. This is NOT the translation column
   * of `modelMatrix`: for the baked beast that column is part of the Z-up/scale/centering
   * normalisation (the skinned result already straddles the world origin with the feet at
   * y = 0), so using it would shift the voxel grid off the model. A future animated mode
   * (the bouncing FurryBall) drives this per frame.
   */
  simTranslation: [number, number, number];
  /** vec4f per vertex (w = 1), model space bind pose */
  bindPositions: Float32Array;
  /** vec4u per vertex */
  skinJoints: Uint32Array;
  /** vec4f per vertex, normalised */
  skinWeights: Float32Array;
  /** triangle list */
  indices: Uint32Array;
  /** column-major mat4 * jointCount * frameCount (final skinning matrices) */
  palettes: Float32Array;
  /** vec3 per vertex, model space bind pose normals */
  bindNormals: Float32Array;
  /**
   * Body collision capsules in bind/model space. Empty when the manifest carries no
   * `"colliders"` section — the engine then falls back to ground-plane-only collision.
   */
  colliders: ModelCollider[];
  /**
   * Optional per-frame rigid placement. Rewrites `modelMatrix` and `simTranslation`
   * for the given animation phase (seconds). The beast's transform is static, so it
   * has none; the FurryBall bounces and rolls.
   */
  updateTransform?: (phase: number) => void;
}

interface BufferRange {
  offset: number;
  length?: number;
}

interface ModelManifest {
  version: number;
  vertexCount: number;
  indexCount: number;
  jointCount: number;
  frameCount: number;
  duration: number;
  modelMatrix: number[];
  buffers: {
    positions: BufferRange;
    normals: BufferRange;
    indices: BufferRange;
    joints: BufferRange;
    weights: BufferRange;
    palettes: BufferRange;
  };
  /** optional SPEC v2 section */
  colliders?: unknown;
}

function vec3(value: unknown): [number, number, number] | null {
  if (!Array.isArray(value) || value.length < 3) return null;
  const [x, y, z] = value.map(Number);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
  return [x, y, z];
}

/**
 * Reads the optional SPEC v2 `"colliders"` array. Anything malformed is dropped with a
 * warning rather than failing the load: a manifest without colliders (the shipped bake
 * until the section lands) is a valid manifest and simply means floor-only collision.
 * One slot is reserved for the pointer brush, hence `limit`.
 */
function parseColliders(raw: unknown, jointCount: number, limit: number): ModelCollider[] {
  if (raw === undefined || raw === null) return [];
  if (!Array.isArray(raw)) {
    console.warn('model manifest: "colliders" is not an array — ignoring it');
    return [];
  }
  const out: ModelCollider[] = [];
  for (const entry of raw) {
    if (out.length >= limit) {
      console.warn(`model manifest: more than ${limit} colliders, ignoring the rest`);
      break;
    }
    const c = entry as Record<string, unknown>;
    const headA = vec3(c.headA);
    const headB = vec3(c.headB);
    const jointA = Number(c.jointA);
    const jointB = Number(c.jointB);
    const radius = Number(c.radius);
    const jointsOk =
      Number.isInteger(jointA) && jointA >= 0 && jointA < jointCount &&
      Number.isInteger(jointB) && jointB >= 0 && jointB < jointCount;
    if (!headA || !headB || !jointsOk || !(radius > 0)) {
      console.warn('model manifest: skipping malformed collider', entry);
      continue;
    }
    out.push({ jointA, headA, jointB, headB, radius });
  }
  return out;
}

// ---------------------------------------------------------------- small mat4 helpers
// All matrices are column-major: m[col * 4 + row].

export function mat4Identity(): Float32Array {
  const m = new Float32Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  return m;
}

export function transformPoint(m: ArrayLike<number>, x: number, y: number, z: number, out: Float32Array, o: number): void {
  out[o] = m[0] * x + m[4] * y + m[8] * z + m[12];
  out[o + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
  out[o + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
}

export function transformDirection(m: ArrayLike<number>, x: number, y: number, z: number, out: Float32Array, o: number): void {
  out[o] = m[0] * x + m[4] * y + m[8] * z;
  out[o + 1] = m[1] * x + m[5] * y + m[9] * z;
  out[o + 2] = m[2] * x + m[6] * y + m[10] * z;
}

// ---------------------------------------------------------------- loading

function viewF32(buffer: ArrayBuffer, byteOffset: number, count: number, what: string): Float32Array {
  const need = byteOffset + count * 4;
  if (need > buffer.byteLength) {
    throw new Error(`${what}: need ${need} bytes, model binary is ${buffer.byteLength}`);
  }
  if (byteOffset % 4 !== 0) return new Float32Array(buffer.slice(byteOffset, need));
  return new Float32Array(buffer, byteOffset, count);
}

function viewU32(buffer: ArrayBuffer, byteOffset: number, count: number, what: string): Uint32Array {
  const need = byteOffset + count * 4;
  if (need > buffer.byteLength) {
    throw new Error(`${what}: need ${need} bytes, model binary is ${buffer.byteLength}`);
  }
  if (byteOffset % 4 !== 0) return new Uint32Array(buffer.slice(byteOffset, need));
  return new Uint32Array(buffer, byteOffset, count);
}

function viewU16(buffer: ArrayBuffer, byteOffset: number, count: number, what: string): Uint16Array {
  const need = byteOffset + count * 2;
  if (need > buffer.byteLength) {
    throw new Error(`${what}: need ${need} bytes, model binary is ${buffer.byteLength}`);
  }
  if (byteOffset % 2 !== 0) return new Uint16Array(buffer.slice(byteOffset, need));
  return new Uint16Array(buffer, byteOffset, count);
}

/**
 * Loads the baked model. `length` fields in the manifest are ignored on purpose —
 * element counts are derived from vertexCount / indexCount / jointCount / frameCount,
 * which sidesteps any bytes-vs-elements ambiguity in the manifest.
 */
export async function loadModel(manifestUrl: string): Promise<HairModel> {
  const manifestResponse = await fetch(manifestUrl);
  if (!manifestResponse.ok) {
    throw new Error(`${manifestUrl}: HTTP ${manifestResponse.status}`);
  }
  const manifest = (await manifestResponse.json()) as ModelManifest;

  const binUrl = manifestUrl.replace(/\.json(\?.*)?$/, '.bin');
  const binResponse = await fetch(binUrl);
  if (!binResponse.ok) {
    throw new Error(`${binUrl}: HTTP ${binResponse.status}`);
  }
  const bin = await binResponse.arrayBuffer();

  const { vertexCount, indexCount, jointCount, frameCount } = manifest;
  if (!vertexCount || !indexCount || !jointCount || !frameCount) {
    throw new Error('model manifest is missing vertexCount/indexCount/jointCount/frameCount');
  }
  if (!Array.isArray(manifest.modelMatrix) || manifest.modelMatrix.length !== 16) {
    throw new Error('model manifest modelMatrix must be 16 floats (column-major)');
  }

  const b = manifest.buffers;
  const positions = viewF32(bin, b.positions.offset, vertexCount * 3, 'positions');
  const normals = viewF32(bin, b.normals.offset, vertexCount * 3, 'normals');
  const indices = viewU32(bin, b.indices.offset, indexCount, 'indices');
  const joints16 = viewU16(bin, b.joints.offset, vertexCount * 4, 'joints');
  const weights = viewF32(bin, b.weights.offset, vertexCount * 4, 'weights');
  const palettes = viewF32(bin, b.palettes.offset, jointCount * frameCount * 16, 'palettes');

  // positions vec3 -> vec4 (storage buffers need the vec4 stride)
  const bindPositions = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i++) {
    bindPositions[i * 4 + 0] = positions[i * 3 + 0];
    bindPositions[i * 4 + 1] = positions[i * 3 + 1];
    bindPositions[i * 4 + 2] = positions[i * 3 + 2];
    bindPositions[i * 4 + 3] = 1;
  }

  const skinJoints = new Uint32Array(vertexCount * 4);
  for (let i = 0; i < skinJoints.length; i++) {
    const j = joints16[i];
    skinJoints[i] = j < jointCount ? j : 0;
  }

  // guard against unnormalised weights so a bad bake cannot collapse the mesh
  const skinWeights = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i++) {
    const o = i * 4;
    let sum = weights[o] + weights[o + 1] + weights[o + 2] + weights[o + 3];
    if (!(sum > 1e-6)) {
      skinWeights[o] = 1;
      continue;
    }
    sum = 1 / sum;
    skinWeights[o] = weights[o] * sum;
    skinWeights[o + 1] = weights[o + 1] * sum;
    skinWeights[o + 2] = weights[o + 2] * sum;
    skinWeights[o + 3] = weights[o + 3] * sum;
  }

  // one slot is kept free for the pointer brush capsule
  const colliders = parseColliders(manifest.colliders, jointCount, MAX_COLLIDERS - 1);
  console.log(
    colliders.length > 0
      ? `model colliders: ${colliders.length} capsule(s) from the manifest, world radii ` +
          `${Math.min(...colliders.map((c) => c.radius)).toFixed(3)} .. ` +
          `${Math.max(...colliders.map((c) => c.radius)).toFixed(3)}`
      : 'model colliders: none in the manifest — ground plane only'
  );

  return {
    name: 'beast',
    vertexCount,
    jointCount,
    frameCount,
    duration: manifest.duration > 0 ? manifest.duration : 1,
    modelMatrix: new Float32Array(manifest.modelMatrix),
    simTranslation: [0, 0, 0],
    bindPositions,
    bindNormals: new Float32Array(normals),
    skinJoints,
    skinWeights,
    indices: new Uint32Array(indices),
    palettes: new Float32Array(palettes),
    colliders,
  };
}

// ---------------------------------------------------------------- sphere fallback

/**
 * Video reference: "Num Hairstrands: 80601" — (200+1) * (400+1), the exact vertex count
 * `ofMesh::sphere(4, 200)` produces. `createSphereModel` welds each pole's `sectors+1`
 * duplicated vertices down to one (see its doc comment), so the actual strand count is
 * `2 * sectors` lower — 79,801 at this resolution. This constant stays the resolution
 * *target* fed to `sphereRingsForStrands`, so `?strands=` and adaptive quality keep
 * tracking the video's density rather than its now-superseded exact count.
 */
export const SPHERE_DEFAULT_STRANDS = 80601;
export const SPHERE_MIN_STRANDS = 1000;
export const SPHERE_MAX_STRANDS = 150000;
/** ofMesh::sphere(4, ...) and the commented-out sphere collision in the original */
const SPHERE_RADIUS = 4;
/** bounce: translation = (0, 4 + 5*|sin t|, 0); roll: slerp factor sin(0.2 t) */
const BOUNCE_BASE = 4;
const BOUNCE_HEIGHT = 5;
const ROLL_RATE = 0.2;
/** |sin t| has period PI, sin(0.2 t) has period 10*PI — 10*PI covers both exactly */
const SPHERE_ANIMATION_PERIOD = 10 * Math.PI;

/**
 * Rings for a UV sphere whose (pre-weld) vertex count `(rings+1) * (2*rings+1)` is
 * closest to `targetStrands`; solving 2r^2 + 3r + 1 = N gives r = (sqrt(8N+1) - 3) / 4.
 * `createSphereModel` welds the two pole rings afterwards, a small fixed adjustment
 * (`2 * sectors` vertices) that does not need to feed back into this choice of rings.
 */
export function sphereRingsForStrands(targetStrands: number): number {
  const n = Math.min(SPHERE_MAX_STRANDS, Math.max(SPHERE_MIN_STRANDS, Math.round(targetStrands)));
  return Math.max(4, Math.round((Math.sqrt(8 * n + 1) - 3) / 4));
}

/**
 * UV sphere of radius 4 centred on the model-space origin — the original's
 * "Furry Ball". The world placement (including the bounce) comes from the model
 * matrix, so at phase 0 the ball rests on the floor with its centre at y = 4.
 *
 * **Poles are welded.** `ofMesh::sphere()`'s tessellation emits `sectors+1` distinct
 * vertices at each pole that all land on the exact same point (`sin(phi) = 0` there, so
 * `theta` no longer affects the position). Treated as independent hair roots, those
 * coincident vertices packed hundreds of strands into a single voxel-grid cell, and the
 * (non-normalised) repulsion gradient blew a crater in the fur at each pole — see
 * IMPLEMENTATION.md's known-artefact note (now resolved). Emitting one vertex per pole
 * instead reproduces every other ring exactly and drops the strand count by
 * `2 * sectors` (800 at the default 200x400 tessellation: 80,601 -> 79,801).
 */
export function createSphereModel(targetStrands = SPHERE_DEFAULT_STRANDS): HairModel {
  const rings = sphereRingsForStrands(targetStrands);
  const sectors = rings * 2;
  const radius = SPHERE_RADIUS;
  const cx = 0;
  const cy = 0;
  const cz = 0;

  // Middle rings (r = 1 .. rings-1) keep the full sectors+1 vertices each, including the
  // seam duplicate at theta = 0 / 2*PI — the tessellation is untouched except at the
  // poles (r = 0 and r = rings), which each collapse to a single vertex.
  const ringVerts = sectors + 1;
  const middleRings = rings - 1;
  const northPole = 0;
  const firstMiddle = 1;
  const ringStart = (r: number): number => firstMiddle + (r - 1) * ringVerts; // r in 1 .. rings-1
  const southPole = firstMiddle + middleRings * ringVerts;
  const vertexCount = southPole + 1;

  const bindPositions = new Float32Array(vertexCount * 4);
  const bindNormals = new Float32Array(vertexCount * 3);
  const skinJoints = new Uint32Array(vertexCount * 4); // all zero -> joint 0
  const skinWeights = new Float32Array(vertexCount * 4);

  const writeVertex = (v: number, phi: number, theta: number): void => {
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    const nx = sinPhi * Math.cos(theta);
    const ny = cosPhi;
    const nz = sinPhi * Math.sin(theta);
    bindPositions[v * 4 + 0] = cx + nx * radius;
    bindPositions[v * 4 + 1] = cy + ny * radius;
    bindPositions[v * 4 + 2] = cz + nz * radius;
    bindPositions[v * 4 + 3] = 1;
    bindNormals[v * 3 + 0] = nx;
    bindNormals[v * 3 + 1] = ny;
    bindNormals[v * 3 + 2] = nz;
    skinWeights[v * 4 + 0] = 1;
  };

  writeVertex(northPole, 0, 0); // phi = 0 -> sinPhi = 0, theta cannot affect the position
  for (let r = 1; r < rings; r++) {
    const phi = (r / rings) * Math.PI; // 0 at +Y
    const row = ringStart(r);
    for (let s = 0; s <= sectors; s++) {
      const theta = (s / sectors) * Math.PI * 2;
      writeVertex(row + s, phi, theta);
    }
  }
  writeVertex(southPole, Math.PI, 0);

  // Triangles: a fan from each pole to its neighbouring ring, plus the same quad strips
  // as an unwelded UV sphere for every ring pair strictly in between. This is exactly the
  // original tessellation with the (previously degenerate, zero-area) pole-to-pole
  // triangles removed — welding does not change any other triangle's winding.
  const indices = new Uint32Array(6 * sectors * (rings - 1));
  let i = 0;

  {
    const row = ringStart(1);
    for (let s = 0; s < sectors; s++) {
      indices[i++] = northPole;
      indices[i++] = row + s;
      indices[i++] = row + s + 1;
    }
  }

  for (let r = 1; r < rings - 1; r++) {
    const a0 = ringStart(r);
    const b0 = ringStart(r + 1);
    for (let s = 0; s < sectors; s++) {
      const a = a0 + s;
      const b = b0 + s;
      indices[i++] = a;
      indices[i++] = b;
      indices[i++] = a + 1;
      indices[i++] = a + 1;
      indices[i++] = b;
      indices[i++] = b + 1;
    }
  }

  {
    const row = ringStart(rings - 1);
    for (let s = 0; s < sectors; s++) {
      indices[i++] = row + s;
      indices[i++] = southPole;
      indices[i++] = row + s + 1;
    }
  }

  const model: HairModel = {
    name: 'sphere',
    vertexCount,
    jointCount: 1,
    frameCount: 1,
    duration: SPHERE_ANIMATION_PERIOD,
    modelMatrix: mat4Identity(),
    simTranslation: [0, 0, 0],
    bindPositions,
    bindNormals,
    skinJoints,
    skinWeights,
    indices,
    palettes: mat4Identity(), // single identity frame
    // The original's commented-out sphere collision, switched on: a degenerate capsule
    // at the model origin with the ball's own radius. The generic per-frame skinning
    // path (identity palette x the bounce matrix) carries it along with the animation.
    colliders: [
      { jointA: 0, headA: [0, 0, 0], jointB: 0, headB: [0, 0, 0], radius: SPHERE_RADIUS },
    ],
  };

  // ofApp.cpp, the two lines that are commented out in the checked-in build:
  //   mModelOrientation.slerp(sin(0.2f * t), identity, rotate(180, 1, 1, 0));
  //   mModelAnimation.setTranslation(0, 4 + 5 * abs(sin(t)), 0);
  // slerp from identity to a 180 degree rotation is just that rotation scaled by the
  // factor, so the roll angle is sin(0.2 t) * PI about the normalised (1,1,0) axis.
  const axis: [number, number, number] = [Math.SQRT1_2, Math.SQRT1_2, 0];
  model.updateTransform = (phase: number): void => {
    const height = BOUNCE_BASE + BOUNCE_HEIGHT * Math.abs(Math.sin(phase));
    composeTranslationRotation(model.modelMatrix, 0, height, 0, axis, Math.sin(ROLL_RATE * phase) * Math.PI);
    // the voxel grid is model-local, so it has to follow the ball
    model.simTranslation[0] = 0;
    model.simTranslation[1] = height;
    model.simTranslation[2] = 0;
  };
  model.updateTransform(0);
  return model;
}

/** out = translate(t) * rotate(angle, axis), column-major. */
function composeTranslationRotation(
  out: Float32Array,
  tx: number,
  ty: number,
  tz: number,
  axis: readonly [number, number, number],
  angle: number
): void {
  const [x, y, z] = axis;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;

  out[0] = t * x * x + c;
  out[1] = t * x * y + s * z;
  out[2] = t * x * z - s * y;
  out[3] = 0;
  out[4] = t * x * y - s * z;
  out[5] = t * y * y + c;
  out[6] = t * y * z + s * x;
  out[7] = 0;
  out[8] = t * x * z + s * y;
  out[9] = t * y * z - s * x;
  out[10] = t * z * z + c;
  out[11] = 0;
  out[12] = tx;
  out[13] = ty;
  out[14] = tz;
  out[15] = 1;
}

// ---------------------------------------------------------------- animation

/**
 * Linearly blends the two palettes bracketing `phase` (seconds, already wrapped into
 * [0, duration)) into `dst` (jointCount * 16 floats).
 */
export function lerpPalette(model: HairModel, phase: number, dst: Float32Array): void {
  const stride = model.jointCount * 16;
  if (model.frameCount <= 1) {
    dst.set(model.palettes.subarray(0, stride));
    return;
  }
  const t = (phase / model.duration) * model.frameCount;
  const f = Math.floor(t);
  const a = t - f;
  const f0 = ((f % model.frameCount) + model.frameCount) % model.frameCount;
  const f1 = (f0 + 1) % model.frameCount;
  const o0 = f0 * stride;
  const o1 = f1 * stride;
  const p = model.palettes;
  if (a <= 0) {
    dst.set(p.subarray(o0, o0 + stride));
    return;
  }
  const inv = 1 - a;
  for (let i = 0; i < stride; i++) {
    dst[i] = p[o0 + i] * inv + p[o1 + i] * a;
  }
}

/** floats written per collider by `skinColliders`: ax ay az bx by bz radius */
export const COLLIDER_FLOATS = 7;

/**
 * Skins the collider capsules into world space with the frame's joint palette:
 * `modelMatrix x (palette[joint] x head)` per endpoint, exactly as SPEC v2 describes.
 *
 * The **radius is taken as a sim-world length and passed through unscaled**. SPEC calls
 * the section "bind/model space", which is true of the endpoints — the shipped bake's
 * heads are raw DAE coordinates (y up to 182) that the palette's 0.01 and the model
 * matrix's 2.62 bring down to world — but its radii are 0.10 .. 0.44 against a 4.5-tall
 * beast, i.e. already fitted in world units. Scaling them by the same 0.0262 would
 * produce centimetre-thin capsules that nothing could ever touch (verified visually:
 * the debug wireframe collapsed to bare lines). See IMPLEMENTATION.md.
 *
 * Writes `COLLIDER_FLOATS` floats per collider and returns the count.
 */
export function skinColliders(model: HairModel, palette: Float32Array, out: Float32Array): number {
  const m = model.modelMatrix;
  const scratch = new Float32Array(3);
  const count = Math.min(model.colliders.length, Math.floor(out.length / COLLIDER_FLOATS));

  for (let i = 0; i < count; i++) {
    const c = model.colliders[i];
    const o = i * COLLIDER_FLOATS;
    const ja = c.jointA * 16;
    const jb = c.jointB * 16;

    transformPoint(palette.subarray(ja, ja + 16), c.headA[0], c.headA[1], c.headA[2], scratch, 0);
    transformPoint(m, scratch[0], scratch[1], scratch[2], out, o);
    transformPoint(palette.subarray(jb, jb + 16), c.headB[0], c.headB[1], c.headB[2], scratch, 0);
    transformPoint(m, scratch[0], scratch[1], scratch[2], out, o + 3);

    out[o + 6] = c.radius;
  }
  return count;
}

export interface SkinnedSurface {
  positions: Float32Array;
  normals: Float32Array;
}

/**
 * CPU linear-blend skinning, used to place the hair particles (once at start-up, and
 * again whenever the colour scheme is switched). Returns world-space positions
 * (`modelMatrix` applied) and unit normals rotated by the same matrix — the original
 * grew hair along the untransformed model-space normal, which only worked because its
 * model matrix had no reorientation.
 */
export function skinWorld(model: HairModel, palette: Float32Array): SkinnedSurface {
  const n = model.vertexCount;
  const positions = new Float32Array(n * 3);
  const normals = new Float32Array(n * 3);
  const m = model.modelMatrix;

  const tmpP = new Float32Array(3);
  const tmpN = new Float32Array(3);

  for (let i = 0; i < n; i++) {
    const px = model.bindPositions[i * 4 + 0];
    const py = model.bindPositions[i * 4 + 1];
    const pz = model.bindPositions[i * 4 + 2];
    const nx = model.bindNormals[i * 3 + 0];
    const ny = model.bindNormals[i * 3 + 1];
    const nz = model.bindNormals[i * 3 + 2];

    let sx = 0;
    let sy = 0;
    let sz = 0;
    let snx = 0;
    let sny = 0;
    let snz = 0;
    for (let k = 0; k < 4; k++) {
      const w = model.skinWeights[i * 4 + k];
      if (w === 0) continue;
      const j = model.skinJoints[i * 4 + k] * 16;
      transformPoint(palette.subarray(j, j + 16), px, py, pz, tmpP, 0);
      transformDirection(palette.subarray(j, j + 16), nx, ny, nz, tmpN, 0);
      sx += w * tmpP[0];
      sy += w * tmpP[1];
      sz += w * tmpP[2];
      snx += w * tmpN[0];
      sny += w * tmpN[1];
      snz += w * tmpN[2];
    }

    transformPoint(m, sx, sy, sz, positions, i * 3);
    transformDirection(m, snx, sny, snz, normals, i * 3);
    const len = Math.hypot(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
    if (len > 1e-6) {
      normals[i * 3] /= len;
      normals[i * 3 + 1] /= len;
      normals[i * 3 + 2] /= len;
    } else {
      normals[i * 3] = 0;
      normals[i * 3 + 1] = 1;
      normals[i * 3 + 2] = 0;
    }
  }
  return { positions, normals };
}
