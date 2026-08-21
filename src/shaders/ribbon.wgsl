// Camera-facing hair ribbons — the default render path.
//
// Non-indexed draw of 6 vertices per strand segment, with the particle buffer pulled
// through a vertex-stage storage binding (see IMPLEMENTATION.md: the device is asked
// for maxStorageBuffersInVertexStage >= 1 and falls back to the line path when the
// adapter will not give it). Only real strands are drawn, so the parked padding
// strands can never produce geometry.
//
//   vertex_index -> segment  = vi / 6,  corner = vi % 6
//   segment      -> strand   = segment / (N - 1),  k = segment % (N - 1)
//   particles      strand * N + k  and  + 1
//
// `ru.fade` switches the same pipeline between the upright hair and its mirrored
// reflection under the floor.

@group(0) @binding(0) var<uniform> ru: RenderUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct RibbonOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
  @location(1) tangent: vec3f,
  @location(2) world: vec3f,
  @location(3) shade: f32,
}

/** width taper along the strand: full at the root, ~0.3 at the tip */
fn taper(k: f32) -> f32 {
  return mix(1.0, 0.3, k / f32(max(NUM_HAIR_PARTICLES - 1u, 1u)));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> RibbonOut {
  let segment = vi / 6u;
  let corner = vi % 6u;
  let segmentsPerStrand = max(NUM_HAIR_PARTICLES - 1u, 1u);
  let strand = segment / segmentsPerStrand;
  let k = segment % segmentsPerStrand;
  let i0 = strand * NUM_HAIR_PARTICLES + k;
  let i1 = i0 + 1u;

  let p0 = particles[i0];
  let p1 = particles[i1];
  let w0 = (ru.model * vec4f(p0.pos.xyz, 1.0)).xyz;
  let w1 = (ru.model * vec4f(p1.pos.xyz, 1.0)).xyz;
  let c0 = ru.viewProj * vec4f(w0, 1.0);
  let c1 = ru.viewProj * vec4f(w1, 1.0);

  // two triangles: (a-, a+, b-) and (b-, a+, b+)
  var sides = array<f32, 6>(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
  var ends = array<f32, 6>(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
  let side = sides[corner];
  let atEnd = ends[corner] > 0.5;

  let clip = select(c0, c1, atEnd);
  let world = select(w0, w1, atEnd);

  // screen-space expansion, so the ribbon keeps a constant pixel width with distance
  let res = vec2f(max(ru.canvasWidth, 1.0), max(ru.canvasHeight, 1.0));
  let ndc0 = c0.xy / max(c0.w, 1e-6);
  let ndc1 = c1.xy / max(c1.w, 1e-6);
  var dir = (ndc1 - ndc0) * res;
  let dirLength = length(dir);
  // a degenerate (zero-length) segment has no direction — any fixed one will do, the
  // quad collapses to a dot either way
  dir = select(vec2f(1.0, 0.0), dir / dirLength, dirLength > 1e-8);
  let normal = vec2f(-dir.y, dir.x);

  let halfWidth = 0.5 * ru.strandWidth * taper(f32(k) + ends[corner]);
  let offset = normal * (2.0 * halfWidth / res) * side;

  var o: RibbonOut;
  o.position = vec4f(clip.xy + offset * clip.w, clip.z, clip.w);
  o.color = vec4f(select(p0.color.rgb, p1.color.rgb, atEnd), 1.0);
  o.world = world;
  o.shade = select(p0.prevPos.w, p1.prevPos.w, atEnd);

  var t = w1 - w0;
  let tl = length(t);
  o.tangent = select(vec3f(0.0, 1.0, 0.0), t / tl, tl > 1e-8);
  return o;
}

@fragment
fn fs(in: RibbonOut) -> @location(0) vec4f {
  var c = (ru.overrideColor * in.color).rgb;
  if (ru.shading > 0.5) {
    let t = normalize(in.tangent);
    let v = normalize(ru.cameraPos.xyz - in.world);
    let lit = kajiyaKay(t, v);
    c = c * lit.x * in.shade + vec3f(lit.y * in.shade);
  }

  var alpha = ru.overrideColor.a * in.color.a;
  if (ru.fade > 0.5) { alpha = reflectionFade(in.position.y, ru.canvasHeight); }
  return vec4f(gammaCorrect(c), alpha);
}
