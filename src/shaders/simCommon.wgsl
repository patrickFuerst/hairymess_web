// hairSimulation.glsl — bindings, grid sampling and the shared kernel prologue/epilogue.
// Composed in front of simDFTL.wgsl / simPBD.wgsl (which only contain their main()).

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> strandLengths: array<f32>;
@group(0) @binding(3) var<storage, read> roots: array<vec4f>;   // model space (skinning output)
@group(0) @binding(4) var<storage, read> velocityGrid: array<vec4f>;
@group(0) @binding(5) var<storage, read> gradientGrid: array<vec4f>;

var<workgroup> sharedPos: array<vec4f, WORK_GROUP_SIZE>;
var<workgroup> sharedFix: array<u32, WORK_GROUP_SIZE>;
var<workgroup> sharedLen: array<f32, STRANDS_PER_GROUP>;

// ---------------------------------------------------------------- grid sampling

fn mapToGrid(pos: vec3f) -> vec3f {
  return ((pos - u.minBB.xyz) / (u.maxBB.xyz - u.minBB.xyz)) * f32(GRID_SIZE);
}

fn getVelocity(c: vec3i) -> vec3f {
  if (!inGrid(c)) { return vec3f(0.0); }
  return velocityGrid[voxelIndex(c.x, c.y, c.z)].xyz;
}

fn getGradient(c: vec3i) -> vec3f {
  if (!inGrid(c)) { return vec3f(0.0); }
  return gradientGrid[voxelIndex(c.x, c.y, c.z)].xyz;
}

fn trilinearWeight(delta: vec3f, corner: u32) -> f32 {
  let wx = select(1.0 - delta.x, delta.x, (corner & 1u) == 1u);
  let wy = select(1.0 - delta.y, delta.y, ((corner >> 1u) & 1u) == 1u);
  let wz = select(1.0 - delta.z, delta.z, ((corner >> 2u) & 1u) == 1u);
  return wx * wy * wz;
}

fn cornerOffset(corner: u32) -> vec3i {
  return vec3i(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
}

fn trilinearVelocity(gridPos: vec3f) -> vec3f {
  let s = mapToGrid(gridPos);
  let cellF = floor(s);
  let delta = s - cellF;
  let base = vec3i(cellF);
  var acc = vec3f(0.0);
  for (var c = 0u; c < 8u; c = c + 1u) {
    acc += getVelocity(base + cornerOffset(c)) * trilinearWeight(delta, c);
  }
  return acc;
}

fn trilinearGradient(gridPos: vec3f) -> vec3f {
  let s = mapToGrid(gridPos);
  let cellF = floor(s);
  let delta = s - cellF;
  let base = vec3i(cellF);
  var acc = vec3f(0.0);
  for (var c = 0u; c < 8u; c = c + 1u) {
    acc += getGradient(base + cornerOffset(c)) * trilinearWeight(delta, c);
  }
  return acc;
}

// calculateFrictionAndRepulsionVelocityCorrection(); gridPos is voxel-grid space
// (world position minus g_modelTranslation).
fn frictionAndRepulsion(vel: vec3f, gridPos: vec3f) -> vec3f {
  let interpolated = trilinearVelocity(gridPos);
  var v = (1.0 - u.friction) * vel + u.friction * interpolated;
  // non-normalised gradient variant (the one the original ships with)
  v = v + u.repulsion * trilinearGradient(gridPos) * u.deltaTime;
  return v;
}

// computeHelper.glsl positionIntegration()
fn positionIntegration(pos: vec3f, vel: vec3f) -> vec3f {
  return pos
       + vel * u.velocityDamping * u.deltaTime
       + u.gravity.xyz * u.deltaTime * u.deltaTime;
}

// ---------------------------------------------------------------- prologue / epilogue

struct ThreadState {
  localVertexIndex: u32,
  localStrandIndex: u32,
  globalStrandIndex: u32,
  vertexIndexInStrand: u32,
  gi: u32,
  oldPosition: vec4f,
  prevPosition: vec4f,
  color: vec4f,
}

// calculateIndices() + the shared-memory load of hairSimulation.glsl main().
// Writes workgroup memory but contains no barrier: the caller issues the barrier
// from uniform control flow.
fn loadState(gid: vec3u, lid: vec3u, wid: vec3u) -> ThreadState {
  var s: ThreadState;
  s.localVertexIndex = lid.x;
  s.localStrandIndex = lid.x / NUM_HAIR_PARTICLES;
  s.globalStrandIndex = wid.x * STRANDS_PER_GROUP + s.localStrandIndex;
  s.vertexIndexInStrand = lid.x % NUM_HAIR_PARTICLES;
  s.gi = gid.x;

  if (s.vertexIndexInStrand > 0u) {
    let p = particles[s.gi];
    s.oldPosition = vec4f(p.pos.xyz, 1.0);
    s.prevPosition = vec4f(p.prevPos.xyz, 1.0);
    s.color = p.color;
    sharedFix[s.localVertexIndex] = select(0u, 1u, p.color.w > 0.5);
  } else {
    // roots are re-derived from the skinned mesh every step
    let r = roots[s.globalStrandIndex];
    s.oldPosition = u.modelMatrix * vec4f(r.xyz, 1.0);
    s.prevPosition = vec4f(0.0);   // original: vec4(0)
    s.color = vec4f(1.0);          // original: vec4(1) -> white root, w = 1 = fixed
    sharedFix[s.localVertexIndex] = 1u;
  }
  sharedPos[s.localVertexIndex] = s.oldPosition;

  // one writer per strand; the caller's barrier publishes it
  if (s.vertexIndexInStrand == 0u) {
    sharedLen[s.localStrandIndex] = strandLengths[s.globalStrandIndex];
  }
  return s;
}

// updateParticle(): prevPos becomes the position this step started from, the colour
// (and with it the fix flag in .w) is written back unchanged.
fn writeBack(s: ThreadState, pos: vec3f) {
  particles[s.gi].pos = vec4f(pos, 1.0);
  particles[s.gi].prevPos = vec4f(s.oldPosition.xyz, 1.0);
  particles[s.gi].color = s.color;
}
