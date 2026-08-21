// Linear blend skinning of the bind pose with the CPU-lerped joint palette.
// Replaces the original's CPU (Assimp) skinning: writes both the render mesh
// positions and the hair root positions (model space; the sim applies modelMatrix).

struct SkinUniforms {
  numVerts: u32,  //  0
  numJoints: u32, //  4
  pad0: u32,      //  8
  pad1: u32,      // 12
}

@group(0) @binding(0) var<uniform> su: SkinUniforms;
@group(0) @binding(1) var<storage, read> bindPositions: array<vec4f>;
@group(0) @binding(2) var<storage, read> skinJoints: array<vec4u>;
@group(0) @binding(3) var<storage, read> skinWeights: array<vec4f>;
@group(0) @binding(4) var<storage, read> palette: array<mat4x4f>;
@group(0) @binding(5) var<storage, read_write> roots: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> skinnedPositions: array<vec4f>;

// No barriers in this kernel, so the early-out is safe.
@compute @workgroup_size(WORK_GROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= su.numVerts) { return; }

  let bp = vec4f(bindPositions[i].xyz, 1.0);
  let j = skinJoints[i];
  let w = skinWeights[i];

  var p = w.x * (palette[j.x] * bp);
  p += w.y * (palette[j.y] * bp);
  p += w.z * (palette[j.z] * bp);
  p += w.w * (palette[j.w] * bp);

  let skinned = vec4f(p.xyz, 1.0);
  skinnedPositions[i] = skinned;
  roots[i] = skinned;
}
