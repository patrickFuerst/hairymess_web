// basic_VS.glsl + basic_FS.glsl. The particle storage buffer is bound directly as a
// vertex buffer (stride 48): pos @0 -> location 0, color @32 -> location 1.

@group(0) @binding(0) var<uniform> ru: RenderUniforms;

@vertex
fn vs(@location(0) position: vec4f, @location(1) color: vec4f) -> VSOut {
  var o: VSOut;
  o.position = ru.viewProj * (ru.model * vec4f(position.xyz, 1.0));
  // color.w carries the fix flag, never an opacity -> force it to 1
  o.color = vec4f(color.rgb, 1.0);
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let c = ru.overrideColor * in.color;
  return vec4f(gammaCorrect(c.rgb), c.a);
}
