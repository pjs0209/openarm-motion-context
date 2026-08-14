# Lift Motion-Context Checkpoint

The bundled deployment artifact was trained with the `motion_context`
communication mode for `Isaac-OpenArm-Lift-v0`.

- `best_agent.pt`: left/right MAPPO policy weights and observation-normalizer state
- `best_agent_motion_context.pt`: frozen running scales used by the 3D motion context

The deployment loader discovers the sidecar automatically from the policy filename.
Keep both files in the same directory.

SHA-256:

```text
210be848553471adc50ce99d6f8d1be02cd26e11aa2a8905acbea8098fb4402c  best_agent.pt
428d2a1fca172376b224030e92729d528c36b37b3c6d256ab0ad061b66c394fd  best_agent_motion_context.pt
```
