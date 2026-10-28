* Added an OpenArm gripper and oracle pose-tracking diagnostic and an explicit
  premature-closure penalty coefficient for isolated Newton reward experiments.
  Historical rewards and success criteria were preserved by default.
* Added an optional control-tick gripper reference speed limit while preserving
  legacy step commands by default and holding references through substeps.
* Recorded the actual Peg success-hold counter in evaluation traces.
* Recorded pre-action robot and task-body states in evaluation traces to audit
  paired physical starts without changing resets or random-generator state.
* Restored the recorded gripper reference speed in the ROS Lift adapter and
  matched its control-tick trajectory and reset behavior to the simulator.
* Added optional named-joint motor-effort recording to the Newton control probe,
  using the solver's actuator-force buffer with units and sample timing instead
  of interpreting PD estimates as measured contact forces.
* Extended the full-start Lift diagnostic to retain its validated joint-space
  approach and optionally track a constant-orientation Cartesian lift through
  ordinary incremental actions, without modifying task success or object state.
* Added a duration-controlled passive ball diagnostic with unchanged arm targets
  and open grippers, stopping at episode termination to expose uninterrupted
  drift separately from grasp or lift motion.
