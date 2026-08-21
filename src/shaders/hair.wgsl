// basic_VS.glsl + basic_FS.glsl — the original 1px line-strip hair path
// ('render: lines', and the automatic fallback when the device cannot bind a storage
// buffer in the vertex stage). The particle storage buffer is bound directly as a
// vertex buffer (stride 48): pos @0 -> location 0, color @32 -> location 1,
// prevPos.w (the solver's self-shadow term) @28 -> location 2.
//
// `ru.fade` switches this same pipeline between the upright hair and its mirrored
// reflection under the floor (floor_fs.glsl).

@group(0) @binding(0) var<uniform> ru: RenderUniforms;

struct LineOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
  @location(1) shade: f32,
}

@vertex
fn vs(@location(0) position: vec4f,
      @location(1) color: vec4f,
      @location(2) shade: f32) -> LineOut {
  var o: LineOut;
  o.position = ru.viewProj * (ru.model * vec4f(position.xyz, 1.0));
  // color.w carries the fix flag, never an opacity -> force it to 1
  o.color = vec4f(color.rgb, 1.0);
  o.shade = shade;
  return o;
}

@fragment
fn fs(in: LineOut) -> @location(0) vec4f {
  var c = (ru.overrideColor * in.color).rgb;
  // A line strip has no tangent to hand the fragment shader without pulling the
  // neighbouring particle, which this path deliberately cannot do — so 'shading' here
  // means the volumetric self-shadow only. Ribbons get the full Kajiya-Kay model.
  if (ru.shading > 0.5) { c = c * in.shade; }

  var alpha = ru.overrideColor.a * in.color.a;
  if (ru.fade > 0.5) { alpha = reflectionFade(in.position.y, ru.canvasHeight); }
  return vec4f(gammaCorrect(c), alpha);
}
