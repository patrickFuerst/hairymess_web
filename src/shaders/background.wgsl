// ofBackgroundGradient(ofColor::lightGray, ofColor::white, OF_GRADIENT_CIRCULAR):
// a triangle fan centred on the viewport whose rim (at half the screen diagonal)
// carries the edge colour. Reproduced here analytically on a full screen triangle.

struct BgUniforms {
  centerColor: vec4f, //  0
  edgeColor: vec4f,   // 16
  resolution: vec2f,  // 32  framebuffer size in device pixels
  pad0: f32,          // 40
  pad1: f32,          // 44
}
@group(0) @binding(0) var<uniform> bg: BgUniforms;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  let center = bg.resolution * 0.5;
  let halfDiagonal = max(length(center), 1.0);
  let t = clamp(distance(fragPos.xy, center) / halfDiagonal, 0.0, 1.0);
  // oF draws this with its default (non gamma-correcting) shader, so no gammaCorrect here.
  return vec4f(mix(bg.centerColor.rgb, bg.edgeColor.rgb, t), 1.0);
}
