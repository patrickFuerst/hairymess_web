// voxelGridPost.glsl — decode the fixed-point atomics, normalise velocity by density
// and build the (non-normalised) density gradient with central differences.
// The atomic buffers are viewed as plain array<i32> here: a different pipeline may
// legally view the same buffer with a non-atomic type.

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read> densityI: array<i32>;
@group(0) @binding(2) var<storage, read> velocityI: array<i32>;
@group(0) @binding(3) var<storage, read_write> velocityW: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> gradientW: array<vec4f>;

fn density(x: i32, y: i32, z: i32) -> f32 {
  return f32(densityI[voxelIndex(x, y, z)]) / DENSITY_SCALE;
}

@compute @workgroup_size(VOXEL_LOCAL_SIZE, VOXEL_LOCAL_SIZE, VOXEL_LOCAL_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  let z = i32(gid.z);
  let ci = voxelIndex(x, y, z);

  let d = f32(densityI[ci]) / DENSITY_SCALE;
  var v = vec3f(
    f32(velocityI[ci * 4 + 0]),
    f32(velocityI[ci * 4 + 1]),
    f32(velocityI[ci * 4 + 2])
  ) / VELOCITY_SCALE;
  if (d > 0.0) { v = v / d; }
  velocityW[ci] = vec4f(v, 0.0);

  // calculateDensityGradient(): interior cells only, otherwise zero.
  // Note the original's sign: (d(i-1) - d(i+1)) / 2, i.e. it points *down* the
  // density field, which is what the repulsion term wants.
  var grad = vec3f(0.0);
  if (x >= 1 && x < GRID_SIZE - 1 &&
      y >= 1 && y < GRID_SIZE - 1 &&
      z >= 1 && z < GRID_SIZE - 1) {
    grad = vec3f(
      (density(x - 1, y, z) - density(x + 1, y, z)) * 0.5,
      (density(x, y - 1, z) - density(x, y + 1, z)) * 0.5,
      (density(x, y, z - 1) - density(x, y, z + 1)) * 0.5
    );
  }
  gradientW[ci] = vec4f(grad, 0.0);
}
