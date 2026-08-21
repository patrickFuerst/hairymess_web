// floor_fs.glsl — the mirrored hair below the floor, faded out with distance from
// the horizon. Same geometry as the hair pass; ru.model carries scale(1,-1,1).

@group(0) @binding(0) var<uniform> ru: RenderUniforms;

@vertex
fn vs(@location(0) position: vec4f, @location(1) color: vec4f) -> VSOut {
  var o: VSOut;
  o.position = ru.viewProj * (ru.model * vec4f(position.xyz, 1.0));
  o.color = vec4f(color.rgb, 1.0);
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  // GL's gl_FragCoord is y-up, WebGPU's @builtin(position) is y-down.
  let y = ru.canvasHeight - in.position.y;
  let delta = y / max(ru.canvasHeight - ru.canvasHeight / 2.0, 1.0);
  let alpha = clamp(0.01 + pow(delta, 4.0), 0.0, 1.0);
  let c = ru.overrideColor * in.color;
  return vec4f(gammaCorrect(c.rgb), alpha);
}
