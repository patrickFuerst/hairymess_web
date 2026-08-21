// voxelGridFilter.glsl — separable 3-tap box filter, one dispatch per axis.
// Group 0 flips with the ping-pong parity, group 1 selects the axis (3 static
// uniform buffers, no per-frame writes).

@group(0) @binding(0) var<storage, read> velocityR: array<vec4f>;
@group(0) @binding(1) var<storage, read> gradientR: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> velocityW: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> gradientW: array<vec4f>;

struct FilterUniforms {
  axis: i32, //  0   0 = x, 1 = y, 2 = z
  pad0: i32, //  4
  pad1: i32, //  8
  pad2: i32, // 12
}
@group(1) @binding(0) var<uniform> f: FilterUniforms;

@compute @workgroup_size(VOXEL_LOCAL_SIZE, VOXEL_LOCAL_SIZE, VOXEL_LOCAL_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  let z = i32(gid.z);

  let kernel = 1.0 / 3.0;
  var velocityValue = vec4f(0.0);
  var gradientValue = vec4f(0.0);

  for (var i = 0; i < 3; i = i + 1) {
    var cx = x;
    var cy = y;
    var cz = z;
    // The original computed the tap index in unsigned arithmetic, so an
    // underflow at index 0 wrapped and hit the `>= gridSize` branch: an
    // out-of-range tap falls back to the centre sample.
    if (f.axis == 0) {
      let t = x - 1 + i;
      cx = select(x, t, t >= 0 && t < GRID_SIZE);
    } else if (f.axis == 1) {
      let t = y - 1 + i;
      cy = select(y, t, t >= 0 && t < GRID_SIZE);
    } else {
      let t = z - 1 + i;
      cz = select(z, t, t >= 0 && t < GRID_SIZE);
    }
    let idx = voxelIndex(cx, cy, cz);
    velocityValue += kernel * velocityR[idx];
    gradientValue += kernel * gradientR[idx];
  }

  let dst = voxelIndex(x, y, z);
  velocityW[dst] = velocityValue;
  gradientW[dst] = gradientValue;
}
