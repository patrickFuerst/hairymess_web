// Linear blend skinning of the bind pose with the CPU-lerped joint palette.
// Replaces the original's CPU (Assimp) skinning: writes both the render mesh
// vertices (position + normal, interleaved) and the hair root positions (model
// space; the sim applies modelMatrix).

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
// two vec4f per vertex: [2i] = position (w = 1), [2i+1] = normal (w = 0).
// One buffer instead of two keeps the group at seven storage bindings and gives the
// mesh pass a single interleaved vertex buffer (stride 32).
@group(0) @binding(6) var<storage, read_write> skinnedVerts: array<vec4f>;
@group(0) @binding(7) var<storage, read> bindNormals: array<vec4f>;

// No barriers in this kernel, so the early-out is safe.
@compute @workgroup_size(WORK_GROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= su.numVerts) { return; }

  let bp = vec4f(bindPositions[i].xyz, 1.0);
  let bn = vec4f(bindNormals[i].xyz, 0.0);
  let j = skinJoints[i];
  let w = skinWeights[i];

  var p = w.x * (palette[j.x] * bp);
  p += w.y * (palette[j.y] * bp);
  p += w.z * (palette[j.z] * bp);
  p += w.w * (palette[j.w] * bp);

  // normals only need the palette's rotation part, so skin them as directions
  var n = w.x * (palette[j.x] * bn);
  n += w.y * (palette[j.y] * bn);
  n += w.z * (palette[j.z] * bn);
  n += w.w * (palette[j.w] * bn);
  var nn = vec3f(0.0, 1.0, 0.0);
  let nl = length(n.xyz);
  if (nl > 1e-6) { nn = n.xyz / nl; }

  let skinned = vec4f(p.xyz, 1.0);
  skinnedVerts[i * 2u] = skinned;
  skinnedVerts[i * 2u + 1u] = vec4f(nn, 0.0);
  // roots.w is the strand's rest length, written once at build time — keep it.
  roots[i] = vec4f(p.xyz, roots[i].w);
}
