* Added Newton/MJWarp as the OpenArm default, retaining an explicit historical PhysX replay profile.
  Stage-local preflight repairs camera joint direction, reflected geometry, fixed zero-mass markers
  and collision schemas without modifying the original robot USD. SDF contacts retain cavity/bore
  surfaces, and contact/triangle-pair workspaces are sized for replicated environments.
* Added D435i nominal 75 g payloads and explicitly marked user-approved CAD/infill approximations
  for unmeasured COM, inertia and printed mounts, preserving optical frames.
* Added physical Peg grasp welds before Newton model finalization, with consistent compile-time
  free-root poses and prototype-local semantic frames for homogeneous native clones.
* Made OpenArm stage lookup and ball mass access independent of Kit and PhysX tensor-view APIs.
* Separated new experiment outputs from archived historical training and evaluation artifacts.
