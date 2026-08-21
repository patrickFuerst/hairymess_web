// voxelGridFill.glsl — trilinear scatter of density + velocity into the voxel grid.
// The original used GL_NV_shader_atomic_float; WGSL has no float atomics, so this
// scatters fixed-point i32 (DENSITY_SCALE / VELOCITY_SCALE), decoded in voxelPost.

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> densityAtomic: array<atomic<i32>>;
// 4 i32 per cell (x,y,z,unused) so cell indexing stays identical to the vec4f grids.
@group(0) @binding(3) var<storage, read_write> velocityAtomic: array<atomic<i32>>;

// dispatch is exactly paddedParticleCount / WORK_GROUP_SIZE -> no bounds check needed
@compute @workgroup_size(WORK_GROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = particles[gid.x];
  let velocity = (p.pos.xyz - p.prevPos.xyz) / u.deltaTime;

  // mapPositionToGridIndex(position - g_modelTranslation)
  let gridPos = p.pos.xyz - u.modelTranslation.xyz;
  let scaled = ((gridPos - u.minBB.xyz) / (u.maxBB.xyz - u.minBB.xyz)) * f32(GRID_SIZE);
  let cellF = floor(scaled);
  let delta = scaled - cellF;
  let base = vec3i(cellF);

  // The original only guarded the +1 corners. Guarding the base cell as well is the
  // corrected equivalent and is what parks the padding strands harmlessly outside.
  if (!inGrid(base)) { return; }

  for (var c = 0u; c < 8u; c = c + 1u) {
    let ox = i32(c & 1u);
    let oy = i32((c >> 1u) & 1u);
    let oz = i32((c >> 2u) & 1u);
    let cell = base + vec3i(ox, oy, oz);
    if (cell.x >= GRID_SIZE || cell.y >= GRID_SIZE || cell.z >= GRID_SIZE) { continue; }

    let wx = select(1.0 - delta.x, delta.x, ox == 1);
    let wy = select(1.0 - delta.y, delta.y, oy == 1);
    let wz = select(1.0 - delta.z, delta.z, oz == 1);
    let weight = wx * wy * wz;

    let ci = voxelIndex(cell.x, cell.y, cell.z);
    // density value per particle is 1.0 (original: trilinearInsertDensity(pos, 1.0))
    atomicAdd(&densityAtomic[ci], i32(weight * DENSITY_SCALE));
    let v = velocity * weight * VELOCITY_SCALE;
    atomicAdd(&velocityAtomic[ci * 4 + 0], i32(v.x));
    atomicAdd(&velocityAtomic[ci * 4 + 1], i32(v.y));
    atomicAdd(&velocityAtomic[ci * 4 + 2], i32(v.z));
  }
}
