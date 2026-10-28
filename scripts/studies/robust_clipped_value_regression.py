"""STAGED optimizer candidate; not installed in any live experiment.

The actual project updater clips predicted values before a single MSE, which
can suppress correction when the UNCLIPPED prediction is worse. This candidate
uses the larger original/clipped per-sample squared error. It preserves loss
scale, balanced actor projection and per-agent flag dictionaries. CPU evidence
is not a simulator smoke, performance gain or explanation of previous NaNs.
"""

import ast
import hashlib
import inspect
import textwrap
from types import MethodType

import torch

MARKER = "_robust_clipped_value_regression_version"


def transform_update(source):
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != "_update":
        raise ValueError("Actual project update function required")
    changes = {"save_original": 0, "loss": 0}

    class Transform(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            if ast.unparse(node.test) == "self._clip_predicted_values[uid]":
                if changes["save_original"]:
                    raise ValueError("Ambiguous project value-clipping branch")
                changes["save_original"] += 1
                saved = ast.parse("original_predicted_values = predicted_values").body[0]
                return [ast.copy_location(saved, node), node]
            return node

        def visit_Assign(self, node):
            if [ast.unparse(target) for target in node.targets] == ["value_loss"]:
                expected = "self._value_loss_scale[uid] * F.mse_loss(sampled['returns'], predicted_values)"
                if ast.unparse(node.value) != expected:
                    raise ValueError("Project value loss changed; refuse guesswork")
                changes["loss"] += 1
                replacement = ast.parse(
                    "value_loss = self._value_loss_scale[uid] * "
                    "torch.maximum((sampled['returns'] - original_predicted_values).square(), "
                    "(sampled['returns'] - predicted_values).square()).mean()"
                ).body[0]
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)

    tree = Transform().visit(tree)
    if changes != {"save_original": 1, "loss": 1}:
        raise ValueError("Exactly one project clipping branch and loss required")
    function.name = "_robust_clipped_value_update"
    ast.fix_missing_locations(tree)
    return tree


def install(agent):
    flags = agent._clip_predicted_values
    if not isinstance(flags, dict) or not flags or any(value is not True for value in flags.values()):
        raise ValueError("Explicit per-agent enabled value clipping required")
    attribute = "_full6d_original_update" if hasattr(agent, "_full6d_original_update") else "_update"
    bound = getattr(agent, attribute)
    if bound.__func__.__qualname__ != "MotionContextMAPPO._update":
        raise ValueError("Actual project updater required, not native or another patch")
    source = inspect.getsource(bound.__func__)
    tree = transform_update(source)
    scope = dict(bound.__func__.__globals__)
    exec(compile(tree, "<staged_robust_clipped_value_regression>", "exec"), scope)
    replacement = scope["_robust_clipped_value_update"]
    setattr(agent, attribute, MethodType(replacement, agent))
    for modules in agent.models.values():
        critic = modules["value"]
        critic.register_buffer(MARKER, torch.tensor(1.0, device=critic.device))

        def validate(module, incompatible):
            if MARKER in incompatible.missing_keys or float(getattr(module, MARKER)) != 1.0:
                raise ValueError("Robust clipped-value critic identity missing or changed")

        critic.register_load_state_dict_post_hook(validate)
    return {
        "actual_project_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "transformed_ast_sha256": hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        "loss": "scale * mean(max(original_squared_error, clipped_squared_error))",
        "stage": "CPU prototype only; no GPU integration or training outcome",
    }


def build_robust_clipped_value_bounded_all_mappo_agent(env, experiment_cfg):
    from scripts.studies.bounded_all_control import build_bounded_all_control_mappo_agent

    if experiment_cfg["agent"].get("clip_predicted_values") is not True:
        raise ValueError("Saved configuration must explicitly enable value clipping")
    agent = build_bounded_all_control_mappo_agent(env, experiment_cfg)
    agent._robust_clipped_value_identity = install(agent)
    return agent


def build_fresh_camera_robust_clipped_value_bounded_all_mappo_agent(env, experiment_cfg):
    from scripts.studies.bounded_all_control import build_fresh_camera_bounded_all_control_mappo_agent

    if experiment_cfg["agent"].get("clip_predicted_values") is not True:
        raise ValueError("Saved configuration must explicitly enable value clipping")
    agent = build_fresh_camera_bounded_all_control_mappo_agent(env, experiment_cfg)
    agent._robust_clipped_value_identity = install(agent)
    return agent
