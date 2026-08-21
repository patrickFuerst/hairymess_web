// Red wireframe of the simulation bounding box (ofDrawBox in the original).
// 12 edges as a line list, generated from the uniform — no vertex buffer.

@group(0) @binding(0) var<uniform> ru: RenderUniforms;
@group(0) @binding(1) var<uniform> u: SimUniforms;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  // corner bit 0 = x, bit 1 = y, bit 2 = z; each edge joins corners differing in one bit
  var edges = array<u32, 24>(
    0u, 1u,  2u, 3u,  4u, 5u,  6u, 7u,   // along x
    0u, 2u,  1u, 3u,  4u, 6u,  5u, 7u,   // along y
    0u, 4u,  1u, 5u,  2u, 6u,  3u, 7u    // along z
  );
  let corner = edges[vi];
  let minB = u.minBB.xyz + u.modelTranslation.xyz;
  let maxB = u.maxBB.xyz + u.modelTranslation.xyz;
  let p = vec3f(
    select(minB.x, maxB.x, (corner & 1u) != 0u),
    select(minB.y, maxB.y, (corner & 2u) != 0u),
    select(minB.z, maxB.z, (corner & 4u) != 0u)
  );

  var o: VSOut;
  o.position = ru.viewProj * vec4f(p, 1.0);
  o.color = ru.overrideColor;
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  return in.color;
}
