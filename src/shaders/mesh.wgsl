// Skinned mesh (drawn solid black, like ofApp::drawAnimatedMesh) and — with a
// different pipeline state — the stencil-only floor quad.
// Vertex buffer: skinnedPositions / floor corners, vec4f, stride 16.

@group(0) @binding(0) var<uniform> ru: RenderUniforms;

@vertex
fn vs(@location(0) position: vec4f) -> @builtin(position) vec4f {
  return ru.viewProj * (ru.model * vec4f(position.xyz, 1.0));
}

@fragment
fn fs() -> @location(0) vec4f {
  return vec4f(gammaCorrect(ru.overrideColor.rgb), ru.overrideColor.a);
}
