// Shared structs + binding-free helpers.
// This file is string-composed in front of EVERY other shader (see src/engine/shaders.ts),
// together with a generated constants prelude that defines:
//   GRID_SIZE, NUM_HAIR_PARTICLES, STRANDS_PER_GROUP, WORK_GROUP_SIZE,
//   DENSITY_SCALE, VELOCITY_SCALE
// Unused structs / functions are legal in WGSL, so every shader can share this header.

// 48 byte stride. Must match PARTICLE_STRIDE in src/engine/layout.ts and the
// hair vertex buffer layout (pos @0 -> @location(0), color @32 -> @location(1)).
struct Particle {
  pos: vec4f,     // offset  0   xyz world position, w = 1
  prevPos: vec4f, // offset 16   xyz position at the start of the previous sim step
  color: vec4f,   // offset 32   rgb strand color, w = fix flag (0.0 / 1.0)
}

// 240 bytes. Must match SIM_U in src/engine/layout.ts byte for byte.
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
  pad0: u32,                        // 236
}

// 160 bytes. Must match RENDER_U in src/engine/layout.ts.
struct RenderUniforms {
  viewProj: mat4x4f,   //   0
  model: mat4x4f,      //  64
  overrideColor: vec4f,// 128
  canvasHeight: f32,   // 144  framebuffer height in device pixels
  pad0: f32,           // 148
  pad1: f32,           // 152
  pad2: f32,           // 156
}

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
}

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

struct CollisionResult {
  pos: vec3f,
  vel: vec3f,
}

// computeHelper.glsl calculatePlaneCollision() + hairSimulation.glsl checkCollision(),
// specialised to the ground plane (position (0,0,0), normal (0,1,0)).
fn checkCollision(prevPos: vec3f, pos: vec3f, vel: vec3f) -> CollisionResult {
  var r: CollisionResult;
  r.pos = pos;
  r.vel = vel;

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
