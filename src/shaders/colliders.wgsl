// Debug wireframe for the world-space capsule colliders, generated entirely in the
// vertex shader from SimUniforms — no vertex buffer, exactly like bbox.wgsl.
//
// Per capsule: 3 rings of RING_SEGMENTS line segments plus 4 axial lines, so
// VERTS_PER_CAPSULE = 3 * RING_SEGMENTS * 2 + 8 vertices of a line list.
// For a proper capsule the rings sit at a, the midpoint and b, perpendicular to the
// axis. For a degenerate one (a == b, i.e. a sphere — the FurryBall's body collider,
// and every pointer brush) the three rings become three orthogonal great circles,
// which is what actually reads as a sphere.

@group(0) @binding(0) var<uniform> ru: RenderUniforms;
@group(0) @binding(1) var<uniform> u: SimUniforms;

const RING_SEGMENTS: u32 = 24u;
const RING_VERTS: u32 = RING_SEGMENTS * 2u;
const VERTS_PER_CAPSULE: u32 = RING_VERTS * 3u + 8u;
const TAU: f32 = 6.2831853;

/** Any unit vector perpendicular to `n`. */
fn perpendicular(n: vec3f) -> vec3f {
  var reference = vec3f(0.0, 0.0, 1.0);
  if (abs(n.z) > 0.9) { reference = vec3f(1.0, 0.0, 0.0); }
  return normalize(cross(n, reference));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let index = min(vi / VERTS_PER_CAPSULE, MAX_COLLIDERS - 1u);
  let k = vi % VERTS_PER_CAPSULE;

  let a = u.colliderA[index].xyz;
  let radius = u.colliderA[index].w;
  let b = u.colliderB[index].xyz;

  let axisVector = b - a;
  let axisLength = length(axisVector);
  let degenerate = axisLength < 1e-5;
  var axis = vec3f(0.0, 1.0, 0.0);
  if (!degenerate) { axis = axisVector / axisLength; }
  let e1 = perpendicular(axis);
  let e2 = cross(axis, e1);

  var p = a;
  if (k < RING_VERTS * 3u) {
    let ring = k / RING_VERTS;
    let local = k % RING_VERTS;
    let step = local / 2u + local % 2u; // line list: segment i joins angle i and i+1
    let angle = f32(step) / f32(RING_SEGMENTS) * TAU;
    let c = cos(angle);
    let s = sin(angle);

    if (degenerate) {
      // three orthogonal great circles through the centre
      var uAxis = e1;
      var vAxis = e2;
      if (ring == 1u) { uAxis = e2; vAxis = axis; }
      if (ring == 2u) { uAxis = axis; vAxis = e1; }
      p = a + (uAxis * c + vAxis * s) * radius;
    } else {
      let centre = mix(a, b, f32(ring) * 0.5);
      p = centre + (e1 * c + e2 * s) * radius;
    }
  } else {
    // four lines running along the capsule, offset to its surface
    let axial = k - RING_VERTS * 3u;
    let line = axial / 2u;
    var dir = e1;
    if (line == 1u) { dir = -e1; }
    if (line == 2u) { dir = e2; }
    if (line == 3u) { dir = -e2; }
    let endpoint = select(a, b, (axial % 2u) == 1u);
    p = endpoint + dir * radius;
  }

  var o: VSOut;
  o.position = ru.viewProj * vec4f(p, 1.0);
  o.color = ru.overrideColor;
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  return in.color;
}
