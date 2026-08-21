// hairSimulation.glsl — bindings, grid sampling and the shared kernel prologue/epilogue.
// Composed in front of simDFTL.wgsl / simPBD.wgsl (which only contain their main()).

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
// roots: model space (skinning output). w carries the strand's rest length, packed in
// at build time and preserved by the skinning pass — it used to be its own binding.
@group(0) @binding(2) var<storage, read> roots: array<vec4f>;
@group(0) @binding(3) var<storage, read> velocityGrid: array<vec4f>;
@group(0) @binding(4) var<storage, read> gradientGrid: array<vec4f>;
// the fill pass's atomic density buffer, viewed as plain i32 for the self-shadow taps
@group(0) @binding(5) var<storage, read> densityI: array<i32>;

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

// ---------------------------------------------------------------- body collision

/**
 * Runs every world-space capsule against the particle, in the same place the ground
 * plane collision runs. `pos`/`vel` are world space; fixed particles never call this.
 */
fn resolveColliders(pos: vec3f, vel: vec3f) -> CollisionResult {
  var r: CollisionResult;
  r.pos = pos;
  r.vel = vel;
  r.dragged = 0.0;
  let n = min(u.colliderCount, MAX_COLLIDERS);
  for (var i = 0u; i < n; i = i + 1u) {
    var c: Capsule;
    c.a = u.colliderA[i].xyz;
    c.radius = u.colliderA[i].w;
    c.b = u.colliderB[i].xyz;
    c.hasVelocity = u.colliderB[i].w;
    c.vel = u.colliderVel[i].xyz;
    let hit = resolveCapsule(c, r.pos, r.vel);
    r.pos = hit.pos;
    r.vel = hit.vel;
    r.dragged = max(r.dragged, hit.dragged);
  }
  return r;
}

// ---------------------------------------------------------------- fur self-shadowing

/** Nearest-cell density lookup in grid space; cells outside the box contribute nothing. */
fn densityAt(gridPos: vec3f) -> f32 {
  let c = vec3i(floor(mapToGrid(gridPos)));
  if (!inGrid(c)) { return 0.0; }
  return f32(densityI[voxelIndex(c.x, c.y, c.z)]) / DENSITY_SCALE;
}

/**
 * Cheap deep-fur shadowing straight off the density grid the sim already builds:
 * three taps marching towards the light, attenuated by Beer-Lambert. The occlusion is
 * measured *relative to the density at the particle itself*, which makes the term
 * self-calibrating — the beast (~140k particles in a 10^3 box) and the FurryBall
 * (~1.3M in a 14^3 box) have wildly different absolute densities but the same ratio
 * deep inside the coat. Result lands in the particle's prevPos.w for the renderer.
 */
fn selfShadow(pos: vec3f) -> f32 {
  let gridPos = pos - u.modelTranslation.xyz;
  let extent = u.maxBB.xyz - u.minBB.xyz;
  let cell = min(extent.x, min(extent.y, extent.z)) / f32(GRID_SIZE);
  let step = LIGHT_DIR * cell * 1.5;

  var sum = 0.0;
  for (var i = 1; i <= 3; i = i + 1) {
    sum = sum + densityAt(gridPos + step * f32(i));
  }
  // 1.0 keeps sparse tip cells (density < 1 particle) from exploding the ratio
  let local = max(densityAt(gridPos), 1.0);
  // exp(-0.35 * 3) = 0.35: fully enclosed fur (all three taps as dense as here)
  return clamp(exp(-0.35 * sum / local), 0.35, 1.0);
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
    sharedLen[s.localStrandIndex] = roots[s.globalStrandIndex].w;
  }
  return s;
}

// updateParticle(): prevPos becomes the position this step started from, the colour
// (and with it the fix flag in .w) is written back unchanged. prevPos.w carries the
// self-shadow transmittance, which the renderer reads back per vertex.
fn writeBackPrev(s: ThreadState, pos: vec3f, prevPos: vec3f) {
  particles[s.gi].pos = vec4f(pos, 1.0);
  particles[s.gi].prevPos = vec4f(prevPos, selfShadow(pos));
  particles[s.gi].color = s.color;
}

fn writeBack(s: ThreadState, pos: vec3f) {
  writeBackPrev(s, pos, s.oldPosition.xyz);
}
