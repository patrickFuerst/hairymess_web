// Shared structs + binding-free helpers.
// This file is string-composed in front of EVERY other shader (see src/engine/shaders.ts),
// together with a generated constants prelude that defines:
//   GRID_SIZE, NUM_HAIR_PARTICLES, STRANDS_PER_GROUP, WORK_GROUP_SIZE,
//   DENSITY_SCALE, VELOCITY_SCALE, MAX_COLLIDERS
// Unused structs / functions are legal in WGSL, so every shader can share this header.

// 48 byte stride. Must match PARTICLE_STRIDE in src/engine/layout.ts and the
// hair vertex buffer layout (pos @0 -> @location(0), prevPos.w @28 -> @location(2),
// color @32 -> @location(1)).
struct Particle {
  pos: vec4f,     // offset  0   xyz world position, w = 1
  prevPos: vec4f, // offset 16   xyz position at the start of the previous sim step,
                  //             w = fur self-shadow transmittance written by the solver
  color: vec4f,   // offset 32   rgb strand color, w = fix flag (0.0 / 1.0)
}

// Must match SIM_U in src/engine/layout.ts byte for byte.
struct SimUniforms {
  modelMatrix: mat4x4f,             //   0
  modelMatrixPrevInverted: mat4x4f, //  64  (parity with the original; verlet path unused)
  modelTranslation: vec4f,          // 128
  gravity: vec4f,                   // 144
  minBB: vec4f,                     // 160
  maxBB: vec4f,                     // 176
  velocityDamping: f32,             // 192
  numIterationsPBD: i32,            // 196
  stiffness: f32,                   // 200
  friction: f32,                    // 204
  repulsion: f32,                   // 208
  ftlDamping: f32,                  // 212
  deltaTime: f32,                   // 216
  gridSize: i32,                    // 220  (mirrors GRID_SIZE)
  numVerticesPerStrand: u32,        // 224
  numStrandsPerThreadGroup: u32,    // 228
  numStrands: u32,                  // 232
  colliderCount: u32,               // 236
  // World-space capsules, re-skinned on the CPU every frame (SPEC v2 "colliders").
  colliderA: array<vec4f, MAX_COLLIDERS>,   // 240        xyz = a, w = radius
  colliderB: array<vec4f, MAX_COLLIDERS>,   // +16*MAX     xyz = b, w = 1 -> carries velocity
  colliderVel: array<vec4f, MAX_COLLIDERS>, // +32*MAX     xyz = world velocity
}

// 192 bytes. Must match RENDER_U in src/engine/layout.ts.
struct RenderUniforms {
  viewProj: mat4x4f,   //   0
  model: mat4x4f,      //  64
  overrideColor: vec4f,// 128
  cameraPos: vec4f,    // 144  world-space eye, for the view vector
  canvasWidth: f32,    // 160  framebuffer size in device pixels
  canvasHeight: f32,   // 164
  strandWidth: f32,    // 168  ribbon width in device pixels
  shading: f32,        // 172  1 = Kajiya-Kay + self-shadow, 0 = flat colours
  fade: f32,           // 176  1 = mirrored reflection, apply the distance fade
  pad0: f32,           // 180
  pad1: f32,           // 184
  pad2: f32,           // 188
}

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
}

/** Fixed key light shared by the fur shading, the body shading and the self-shadowing. */
const LIGHT_DIR: vec3f = vec3f(0.3617697, 0.9044243, 0.2261061); // normalize(0.4, 1.0, 0.25)

// Faithful port of computeHelper.glsl voxelIndex(): x fastest, then z, then y.
fn voxelIndex(x: i32, y: i32, z: i32) -> i32 {
  return x + y * GRID_SIZE * GRID_SIZE + z * GRID_SIZE;
}

fn inGrid(c: vec3i) -> bool {
  return c.x >= 0 && c.x < GRID_SIZE &&
         c.y >= 0 && c.y < GRID_SIZE &&
         c.z >= 0 && c.z < GRID_SIZE;
}

// basic_FS.glsl / floor_fs.glsl:  Color.rgb = pow(Color.rgb, vec3(1.0/2.2))
// max() guards pow() against negative inputs (undefined for negative bases).
fn gammaCorrect(c: vec3f) -> vec3f {
  return pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2));
}

/**
 * floor_fs.glsl's distance fade for the mirrored hair, in screen space.
 * The original divided by h - h/2 and used a 4th power, which reached full opacity at
 * the vertical centre of the screen. The floor diamond is now 60x60 rather than 30x30,
 * so far more distant reflection passes the stencil; spreading the ramp over three
 * quarters of the screen with a cubic keeps that extra ground fading smoothly instead
 * of snapping to opaque halfway up.
 */
fn reflectionFade(fragY: f32, canvasHeight: f32) -> f32 {
  // GL's gl_FragCoord is y-up, WebGPU's @builtin(position) is y-down.
  let y = canvasHeight - fragY;
  let delta = y / max(canvasHeight * 0.75, 1.0);
  return clamp(0.01 + pow(delta, 3.0), 0.0, 1.0);
}

/**
 * Kajiya-Kay strand shading. `t` is the (normalised) strand tangent, `v` the view
 * vector. Diffuse is sin(T,L), the specular lobe is sin(T,H)^e; the ambient floor
 * keeps the original's flat-colour character rather than dropping shadowed fur to
 * black.
 */
fn kajiyaKay(t: vec3f, v: vec3f) -> vec2f {
  let h = normalize(LIGHT_DIR + v);
  let tl = dot(t, LIGHT_DIR);
  let th = dot(t, h);
  let diffuse = sqrt(max(1.0 - tl * tl, 0.0));
  let spec = pow(sqrt(max(1.0 - th * th, 0.0)), 32.0) * 0.25;
  return vec2f(0.45 + 0.55 * diffuse, spec);
}

struct CollisionResult {
  pos: vec3f,
  vel: vec3f,
  /**
   * 1.0 when a *moving* collider dragged the particle along with it. Static geometry
   * (the ground plane, the skinned body capsules) leaves this at 0: their velocity
   * response is the original's, which PBD discards because it collides after
   * integrating. Only the pointer brush has to reach the next step, and it does so by
   * rewriting prevPos.
   */
  dragged: f32,
}

// computeHelper.glsl calculatePlaneCollision() + hairSimulation.glsl checkCollision(),
// specialised to the ground plane (position (0,0,0), normal (0,1,0)).
fn checkCollision(prevPos: vec3f, pos: vec3f, vel: vec3f) -> CollisionResult {
  var r: CollisionResult;
  r.pos = pos;
  r.vel = vel;
  r.dragged = 0.0;

  let d = pos - prevPos;
  let dist = length(d);
  if (dist < 1e-7) { return r; }   // "solves some trouble" (original)

  let n = vec3f(0.0, 1.0, 0.0);
  let planePos = vec3f(0.0, 0.0, 0.0);

  // behind the plane?
  if (dot(pos - planePos, n) < 0.0) {
    let ray = d / dist;
    let denom = dot(ray, n);
    // Fall back to a straight projection onto the plane. The original always used the
    // line/plane intersection, which is only meaningful when the *segment* crosses the
    // plane: for a particle that was already below the plane and is travelling nearly
    // parallel to it, -prevPos.y / ray.y blows up and flings the particle across the
    // scene (or to infinity when ray.y is 0).
    var hit = vec3f(pos.x, planePos.y, pos.z);
    if (abs(denom) > 1e-6) {
      let delta = -dot(prevPos - planePos, n) / denom;
      if (delta >= 0.0 && delta <= dist) {
        hit = prevPos + delta * ray;   // identical to the original in this case
      }
    }
    r.pos = hit;

    let normalPart = dot(vel, n) * n;
    let tangentPart = vel - normalPart;
    r.vel = tangentPart - normalPart;
  }
  return r;
}

// ---------------------------------------------------------------- capsule colliders

/** Closest point to `p` on the segment a..b. */
fn closestPointOnSegment(a: vec3f, b: vec3f, p: vec3f) -> vec3f {
  let ab = b - a;
  let denom = dot(ab, ab);
  if (denom < 1e-12) { return a; }
  return a + ab * clamp(dot(p - a, ab) / denom, 0.0, 1.0);
}

// One capsule of the SPEC v2 collider array, in world space.
struct Capsule {
  a: vec3f,
  b: vec3f,
  radius: f32,
  /** > 0.5 when `vel` is meaningful (the pointer brush); skinned body capsules are static */
  hasVelocity: f32,
  vel: vec3f,
}

/**
 * Body collision. Projects the particle out to the capsule surface and reflects the
 * normal component of its velocity exactly like the ground plane response does
 * (`v = w - u`). A capsule that carries a velocity (the pointer brush) also drags the
 * particle towards its own motion.
 */
fn resolveCapsule(c: Capsule, pos: vec3f, vel: vec3f) -> CollisionResult {
  var r: CollisionResult;
  r.pos = pos;
  r.vel = vel;
  r.dragged = 0.0;

  let onAxis = closestPointOnSegment(c.a, c.b, pos);
  let d = pos - onAxis;
  let dist = length(d);
  if (dist >= c.radius) { return r; }

  // dead centre: no meaningful normal, push straight up like the plane response would
  var n = vec3f(0.0, 1.0, 0.0);
  if (dist > 1e-6) { n = d / dist; }

  r.pos = onAxis + n * c.radius;
  let normalPart = dot(vel, n) * n;
  r.vel = (vel - normalPart) - normalPart;
  if (c.hasVelocity > 0.5) {
    r.vel = mix(r.vel, c.vel, 0.7);
    r.dragged = 1.0;
  }
  return r;
}

// computeHelper.glsl constrainMultiplier(), fix flags as u32 (WGSL workgroup vars
// may not be bool arrays in practice; 0 = free, 1 = fixed).
fn constrainMultiplier(fixed0: u32, fixed1: u32) -> vec2f {
  if (fixed0 == 1u) {
    if (fixed1 == 1u) { return vec2f(0.0, 0.0); }
    return vec2f(0.0, 1.0);
  }
  if (fixed1 == 1u) { return vec2f(1.0, 0.0); }
  return vec2f(0.5, 0.5);
}
