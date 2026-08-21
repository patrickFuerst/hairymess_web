// voxelGrid_vs.glsl / voxelGrid_fs.glsl — one point per voxel, coloured by the
// density gradient (the variant the original ships uncommented).
//
// Like the original's mVoxelVBO, the three grids arrive as per-vertex attributes
// instead of storage buffers: WebGPU's maxStorageBuffersInVertexStage defaults to 0,
// while vertex buffers are always available. Attribute i is cell i, and decoding the
// vertex index below is exactly the inverse of voxelIndex(), so the two agree.

@group(0) @binding(0) var<uniform> ru: RenderUniforms;
@group(0) @binding(1) var<uniform> u: SimUniforms;

@vertex
fn vs(@builtin(vertex_index) vi: u32,
      @location(0) velocity: vec4f,
      @location(1) gradient: vec4f,
      @location(2) densityFixedPoint: i32) -> VSOut {
  let id = i32(vi);
  let ix = id % GRID_SIZE;
  let iz = (id / GRID_SIZE) % GRID_SIZE;
  let iy = id / (GRID_SIZE * GRID_SIZE);

  let cellSize = (u.maxBB.xyz - u.minBB.xyz) / f32(GRID_SIZE);
  // grid space -> world space (the sim samples the grid at world - modelTranslation)
  let p = u.minBB.xyz
        + vec3f(f32(ix), f32(iy), f32(iz)) * cellSize
        + cellSize * 0.5
        + u.modelTranslation.xyz;

  // the original also offers density and velocity colouring, both commented out:
  //   density:  f32(densityFixedPoint) / DENSITY_SCALE
  //   velocity: vec4f(velocity.xyz, step(0.0, length(velocity.xyz)))
  let alpha = select(0.0, 1.0, length(gradient.xyz) > 0.0);

  var o: VSOut;
  o.color = vec4f(gradient.xyz, alpha);
  if (alpha == 0.0) {
    o.position = vec4f(2.0, 2.0, 2.0, 1.0); // outside the clip volume -> discarded
  } else {
    o.position = ru.viewProj * vec4f(p, 1.0);
  }
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  // voxelGrid_fs.glsl passes the colour straight through (no gamma).
  return in.color;
}
