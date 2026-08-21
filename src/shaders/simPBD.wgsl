// PBDApproach.glsl — Position Based Dynamics.
// The iteration count comes from a uniform, so the loop and every barrier inside it
// stay in uniform control flow; the guarded work happens inside the `if`, the barrier
// after it does not.

fn applyLengthConstraintPair(a: u32, b: u32, targetLength: f32, stiffness: f32) {
  let p0 = sharedPos[a].xyz;
  let p1 = sharedPos[b].xyz;
  var d = p1 - p0;
  let dist = max(length(d), 1e-7);
  d = (1.0 - targetLength / dist) * d;
  let m = constrainMultiplier(sharedFix[a], sharedFix[b]);
  sharedPos[a] = vec4f(p0 + m.x * d * stiffness, 1.0);
  sharedPos[b] = vec4f(p1 - m.y * d * stiffness, 1.0);
}

@compute @workgroup_size(WORK_GROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let s = loadState(gid, lid, wid);
  workgroupBarrier();

  var velocity = vec3f(0.0);
  if (sharedFix[s.localVertexIndex] == 0u) {
    let pos = s.oldPosition.xyz;
    velocity = (pos - s.prevPosition.xyz) / u.deltaTime;
    velocity = frictionAndRepulsion(velocity, pos - u.modelTranslation.xyz);
    sharedPos[s.localVertexIndex] = vec4f(positionIntegration(pos, velocity), 1.0);
  }
  workgroupBarrier();

  let iterations = max(1, u.numIterationsPBD);
  // linearise the stiffness in the number of iterations
  let stiffness = 1.0 - pow(1.0 - u.stiffness, 1.0 / f32(iterations));
  let targetLength = sharedLen[s.localStrandIndex] / f32(NUM_HAIR_PARTICLES);

  let index0 = s.localVertexIndex * 2u;
  let index1 = index0 + 1u;
  let index2 = index0 + 2u;

  for (var i = 0; i < iterations; i = i + 1) {
    // even pairs (0,1) (2,3) (4,5) (6,7)
    if (s.localVertexIndex < WORK_GROUP_SIZE / 2u &&
        (index0 % NUM_HAIR_PARTICLES) < NUM_HAIR_PARTICLES - 1u) {
      applyLengthConstraintPair(index0, index1, targetLength, stiffness);
    }
    workgroupBarrier();
    // odd pairs (1,2) (3,4) (5,6); (7,8) is skipped, it would cross strands
    if (s.localVertexIndex < (WORK_GROUP_SIZE - 1u) / 2u &&
        (index1 % NUM_HAIR_PARTICLES) < NUM_HAIR_PARTICLES - 1u) {
      applyLengthConstraintPair(index1, index2, targetLength, stiffness);
    }
    workgroupBarrier();
  }

  // "somehow PBD doesn't like collision detection before constraint" (original)
  var finalPos = sharedPos[s.localVertexIndex].xyz;
  var prevOut = s.oldPosition.xyz;
  if (sharedFix[s.localVertexIndex] == 0u) {
    let plane = checkCollision(s.prevPosition.xyz, finalPos, velocity);
    finalPos = plane.pos;
    let body = resolveColliders(finalPos, plane.vel);
    finalPos = body.pos;
    // PBD integrates before it collides, so the capsule response's velocity would be
    // thrown away: it only survives through prevPos, which is what the next step
    // differentiates. Untouched particles keep the original's prevPos = oldPosition.
    if (body.dragged > 0.5) {
      prevOut = finalPos - body.vel * u.deltaTime;
    }
  }
  writeBackPrev(s, finalPos, prevOut);
}
