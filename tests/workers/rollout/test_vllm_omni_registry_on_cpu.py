# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.rollout.replica import ImageOutput, RolloutMode, RolloutReplicaRegistry, TokenOutput


# ---------------------------------------------------------------------------
# Tests for verl/workers/rollout/base.py – _ROLLOUT_REGISTRY
# ---------------------------------------------------------------------------


def test_vllm_omni_rollout_registry_entry_exists():
    key = ("vllm_omni", "async")
    assert key in _ROLLOUT_REGISTRY
    assert _ROLLOUT_REGISTRY[key] == "verl.workers.rollout.vllm_rollout.vLLMOmniServerAdapter"


def test_rollout_registry_contains_all_builtin_entries():
    """All built-in rollout backends must be present in the registry."""
    expected_keys = [
        ("vllm", "async"),
        ("vllm_omni", "async"),
        ("sglang", "async"),
        ("trtllm", "async"),
    ]
    for key in expected_keys:
        assert key in _ROLLOUT_REGISTRY, f"Missing key {key} in _ROLLOUT_REGISTRY"


def test_rollout_registry_vllm_omni_module_path():
    """The vllm_omni async entry must point to the correct fully-qualified class path."""
    path = _ROLLOUT_REGISTRY[("vllm_omni", "async")]
    assert path == "verl.workers.rollout.vllm_rollout.vLLMOmniServerAdapter"
    module_part, class_part = path.rsplit(".", 1)
    assert module_part == "verl.workers.rollout.vllm_rollout"
    assert class_part == "vLLMOmniServerAdapter"


def test_rollout_registry_values_are_strings():
    """Registry values should be fully-qualified dotted paths (strings)."""
    for key, value in _ROLLOUT_REGISTRY.items():
        assert isinstance(value, str), f"Registry value for {key} is not a string: {value!r}"
        assert "." in value, f"Registry value for {key} does not look like a dotted path: {value!r}"


# ---------------------------------------------------------------------------
# Tests for verl/workers/rollout/replica.py – RolloutReplicaRegistry
# ---------------------------------------------------------------------------


def test_vllm_omni_replica_registry_entry_exists():
    assert "vllm_omni" in RolloutReplicaRegistry._registry
    assert callable(RolloutReplicaRegistry._registry["vllm_omni"])


def test_replica_registry_contains_all_builtin_backends():
    """All built-in replica backends must be registered."""
    for name in ("vllm", "vllm_omni", "sglang", "trtllm"):
        assert name in RolloutReplicaRegistry._registry, f"Missing backend '{name}' in RolloutReplicaRegistry"
        assert callable(RolloutReplicaRegistry._registry[name])


def test_replica_registry_get_raises_for_unknown_backend():
    """get() must raise ValueError when an unknown backend name is requested."""
    with pytest.raises(ValueError, match="Unknown rollout mode"):
        RolloutReplicaRegistry.get("nonexistent_backend_xyz")


def test_replica_registry_register_and_callable():
    """register() should store the loader and make it retrievable via the registry dict."""
    dummy_class = object

    def dummy_loader():
        return dummy_class

    original_registry = dict(RolloutReplicaRegistry._registry)
    try:
        RolloutReplicaRegistry.register("_test_dummy_backend", dummy_loader)
        assert "_test_dummy_backend" in RolloutReplicaRegistry._registry
        assert callable(RolloutReplicaRegistry._registry["_test_dummy_backend"])
    finally:
        # Restore registry state to avoid side effects on other tests.
        RolloutReplicaRegistry._registry.clear()
        RolloutReplicaRegistry._registry.update(original_registry)


# ---------------------------------------------------------------------------
# Tests for verl/workers/rollout/replica.py – RolloutMode enum
# ---------------------------------------------------------------------------


def test_rollout_mode_hybrid_value():
    assert RolloutMode.HYBRID.value == "hybrid"


def test_rollout_mode_colocated_value():
    assert RolloutMode.COLOCATED.value == "colocated"


def test_rollout_mode_standalone_value():
    assert RolloutMode.STANDALONE.value == "standalone"


def test_rollout_mode_all_members():
    """Ensure all expected enum members exist."""
    member_names = {m.name for m in RolloutMode}
    assert {"HYBRID", "COLOCATED", "STANDALONE"}.issubset(member_names)


# ---------------------------------------------------------------------------
# Tests for verl/workers/rollout/replica.py – ImageOutput model
# ---------------------------------------------------------------------------


def test_image_output_defaults():
    output = ImageOutput(image=[[[0.0]]])
    assert output.image == [[[0.0]]]
    assert output.log_probs is None
    assert output.stop_reason is None
    assert output.num_preempted is None
    assert output.extra_info == {}


def test_image_output_with_all_fields():
    """ImageOutput should accept and store all optional fields."""
    image = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
    log_probs = [-0.1, -0.2, -0.3]
    extra = {"all_latents": None, "global_steps": 10}
    output = ImageOutput(
        image=image,
        log_probs=log_probs,
        stop_reason="completed",
        num_preempted=2,
        extra_info=extra,
    )
    assert output.image == image
    assert output.log_probs == log_probs
    assert output.stop_reason == "completed"
    assert output.num_preempted == 2
    assert output.extra_info == extra


def test_image_output_stop_reason_aborted():
    output = ImageOutput(image=[[[0.0]]], stop_reason="aborted")
    assert output.stop_reason == "aborted"


def test_image_output_stop_reason_none():
    output = ImageOutput(image=[[[0.0]]], stop_reason=None)
    assert output.stop_reason is None


# ---------------------------------------------------------------------------
# Tests for verl/workers/rollout/replica.py – TokenOutput model
# ---------------------------------------------------------------------------


def test_token_output_defaults():
    """TokenOutput requires only token_ids; all other fields default to None/empty."""
    output = TokenOutput(token_ids=[1, 2, 3])
    assert output.token_ids == [1, 2, 3]
    assert output.log_probs is None
    assert output.routed_experts is None
    assert output.stop_reason is None
    assert output.num_preempted is None
    assert output.extra_info == {}


def test_token_output_with_all_fields():
    """TokenOutput should store all optional fields correctly."""
    output = TokenOutput(
        token_ids=[10, 20, 30],
        log_probs=[-0.5, -1.0, -1.5],
        routed_experts=[[0, 1], [2, 3]],
        stop_reason="completed",
        num_preempted=1,
        extra_info={"key": "value"},
    )
    assert output.token_ids == [10, 20, 30]
    assert output.log_probs == [-0.5, -1.0, -1.5]
    assert output.routed_experts == [[0, 1], [2, 3]]
    assert output.stop_reason == "completed"
    assert output.num_preempted == 1
    assert output.extra_info == {"key": "value"}


def test_token_output_stop_reason_aborted():
    output = TokenOutput(token_ids=[], stop_reason="aborted")
    assert output.stop_reason == "aborted"


def test_token_output_empty_token_ids():
    """TokenOutput with an empty token_ids list is valid (e.g. aborted request)."""
    output = TokenOutput(token_ids=[])
    assert output.token_ids == []
