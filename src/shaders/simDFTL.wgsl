// DFTLApproach.glsl — Dynamic Follow The Leader.
// Every barrier below sits in uniform control flow: the dispatch covers exactly
// paddedParticleCount threads, so there is no early-out anywhere in this kernel.

@compute @workgroup_size(WORK_GROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let s = loadState(gid, lid, wid);
  workgroupBarrier();

  // The original read sharedPos[i+1] *after* neighbours had already written their
  // collision-corrected position (a race). Snapshot it before anybody writes, then
  // barrier: deterministic, and the barrier stays uniform.
  var posNextBefore = vec3f(0.0);
  if (s.vertexIndexInStrand < NUM_HAIR_PARTICLES - 1u) {
    posNextBefore = sharedPos[s.localVertexIndex + 1u].xyz;
  }
  workgroupBarrier();

  if (sharedFix[s.localVertexIndex] == 0u) {
    var pos = s.oldPosition.xyz;
    var velocity = (pos - s.prevPosition.xyz) / u.deltaTime;

    let hit = checkCollision(s.prevPosition.xyz, pos, velocity);
    pos = hit.pos;
    velocity = hit.vel;

    // follow-the-leader damping towards the next vertex down the strand
    var distanceToNext = vec3f(0.0);
    if (s.vertexIndexInStrand < NUM_HAIR_PARTICLES - 1u) {
      distanceToNext = pos - posNextBefore;
    }
    velocity = velocity - u.ftlDamping * distanceToNext / u.deltaTime;

    velocity = frictionAndRepulsion(velocity, pos - u.modelTranslation.xyz);
    sharedPos[s.localVertexIndex] = vec4f(positionIntegration(pos, velocity), 1.0);
  }
  workgroupBarrier();

  // length constraint, solved serially per strand by its root thread
  if (s.vertexIndexInStrand == 0u) {
    let targetLength = sharedLen[s.localStrandIndex] / f32(NUM_HAIR_PARTICLES);
    let base = s.localVertexIndex;
    for (var i = 0u; i < NUM_HAIR_PARTICLES - 1u; i = i + 1u) {
      let p0 = sharedPos[base + i].xyz;
      let p1 = sharedPos[base + i + 1u].xyz;
      var d = p1 - p0;
      let dist = max(length(d), 1e-7);
      d = (1.0 - targetLength / dist) * d;
      // applyLengthConstraintDFTL: the fix multiplier is computed but unused in the
      // original -- the follower is always the one that moves.
      sharedPos[base + i + 1u] = vec4f(p1 - d * u.stiffness, 1.0);
    }
  }
  workgroupBarrier();

  writeBack(s, sharedPos[s.localVertexIndex].xyz);
}
