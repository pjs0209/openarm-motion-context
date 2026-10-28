"""CPU-stage candidate: freeze actor input normalization during PPO updates.

Use the rollout's existing scaler for all policy likelihood reevaluations.
Only after the full update completes, fit that scaler once on stored raw
rollout states for the NEXT rollout. Critic/value normalization is unchanged.
Statistics still move between rollouts: this is not policy invariance, a proven
bug repair, a GPU-tested builder or a performance claim.
"""

import ast
import hashlib
import inspect
import textwrap
from types import MethodType

import torch

MARKER = "_rollout_frozen_actor_normalization_version"


class Freeze(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_preprocess_policy_states":
            for keyword in node.keywords:
                if keyword.arg == "train":
                    if ast.unparse(keyword.value) != "not epoch":
                        raise ValueError("Original first-epoch actor scaler update required")
                    keyword.value = ast.Constant(False)
                    self.changed += 1
        return node


def adapted_update(function):
    text = inspect.getsource(function)
    tree = ast.parse(textwrap.dedent(text))
    transform = Freeze()
    tree = transform.visit(tree)
    if transform.changed != 1:
        raise ValueError("Exactly one actual actor preprocessing call required")
    function_node = tree.body[0]
    if not isinstance(function_node, ast.FunctionDef) or function_node.name != "_update":
        raise ValueError("Actual project update function required")
    tail = ast.parse(
        "with torch.no_grad():\n"
        "    for uid in self.possible_agents:\n"
        "        self._preprocess_policy_states(uid, self.memories[uid].get_tensor_by_name('states'), train=True)\n"
    ).body
    function_node.body.extend(tail)
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, "<rollout_frozen_actor_normalization>", "exec"), namespace)
    result = namespace["_update"]
    result.__qualname__ = "MotionContextMAPPO._update"
    return result, hashlib.sha256(text.encode()).hexdigest()


def install(agent):
    function = getattr(getattr(agent, "_full6d_original_update", agent._update), "__func__", None)
    if (
        function is None
        or function.__qualname__ != "MotionContextMAPPO._update"
        or agent.cfg.get("clip_predicted_values") is not True
        or not isinstance(agent._clip_predicted_values, dict)
        or any(value is not True for value in agent._clip_predicted_values.values())
    ):
        raise ValueError("Original project clipping-on updater required")
    for modules in agent.models.values():
        critic = modules["value"]
        if any(
            hasattr(critic, name)
            for name in (
                MARKER,
                "_robust_clipped_value_regression_version",
                "_project_unclipped_value_regression_version",
            )
        ):
            raise ValueError("Do not silently combine updater candidates")
    update, digest = adapted_update(function)
    if hasattr(agent, "_full6d_original_update"):
        agent._full6d_original_update = MethodType(update, agent)
    else:
        agent._update = MethodType(update, agent)
    for modules in agent.models.values():
        value = modules["value"]
        value.register_buffer(MARKER, next(value.parameters()).new_tensor(1.0))

        def validate(module, state, prefix, *args):
            marker = state.get(prefix + MARKER)
            if marker is None or marker.numel() != 1 or float(marker) != 1.0 or not torch.isfinite(marker).all():
                raise ValueError("Rollout-frozen actor normalization identity mismatch")

        value.register_load_state_dict_pre_hook(validate)
    return {
        "kind": "rollout_frozen_actor_normalization_v1",
        "original_project_update_sha256": digest,
        "limits": "Actor scaler changes after updates, critic normalization unchanged. No GPU smoke, task outcome, cause or policy-invariance claim.",
    }


def build_rollout_frozen_actor_bounded_all_mappo_agent(env, experiment_cfg):
    from scripts.studies.bounded_all_control import build_bounded_all_control_mappo_agent

    agent = build_bounded_all_control_mappo_agent(env, experiment_cfg)
    install(agent)
    return agent


def build_fresh_camera_rollout_frozen_actor_bounded_all_mappo_agent(env, experiment_cfg):
    from scripts.studies.bounded_all_control import build_fresh_camera_bounded_all_control_mappo_agent

    agent = build_fresh_camera_bounded_all_control_mappo_agent(env, experiment_cfg)
    install(agent)
    return agent
