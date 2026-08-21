// Skinned body mesh (ofApp::drawAnimatedMesh drew it solid black) and — with a
// different pipeline state — the stencil-only floor quad.
// Vertex buffer: skinnedVerts / floor corners, stride 32, position @0 + normal @16.

@group(0) @binding(0) var<uniform> ru: RenderUniforms;

/**
 * The body's unlit base. Kept here rather than in `overrideColor` so that switching
 * shading off leaves the original's solid-black silhouette exactly as it was.
 */
const BODY_BASE: vec3f = vec3f(0.04, 0.04, 0.045);

struct MeshOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) world: vec3f,
}

@vertex
fn vs(@location(0) position: vec4f, @location(1) normal: vec4f) -> MeshOut {
  var o: MeshOut;
  let world = (ru.model * vec4f(position.xyz, 1.0)).xyz;
  o.position = ru.viewProj * vec4f(world, 1.0);
  // the model matrix is a rotation + uniform scale, so this needs no inverse transpose
  o.normal = (ru.model * vec4f(normal.xyz, 0.0)).xyz;
  o.world = world;
  return o;
}

@fragment
fn fs(in: MeshOut) -> @location(0) vec4f {
  // shading off -> exactly the original's flat silhouette
  if (ru.shading < 0.5) {
    return vec4f(gammaCorrect(ru.overrideColor.rgb), ru.overrideColor.a);
  }

  var n = vec3f(0.0, 1.0, 0.0);
  let nl = length(in.normal);
  if (nl > 1e-6) { n = in.normal / nl; }
  let v = normalize(ru.cameraPos.xyz - in.world);

  let hemi = 0.5 + 0.5 * n.y;                       // sky above, ground bounce below
  let ndl = max(dot(n, LIGHT_DIR), 0.0);
  let rim = pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 3.0) * 0.15;

  // Deliberately worked in display space: the numbers below are the final pixel
  // values, so the body stays a near-black silhouette (measured 0.08 unshaded side ..
  // 0.21 lit) instead of being lifted into mid-grey by the pow(1/2.2) the other passes
  // apply. That gamma is folded into these constants.
  let c = BODY_BASE * (0.6 + 0.8 * hemi) + vec3f(0.11 * ndl) + vec3f(rim);
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), ru.overrideColor.a);
}
