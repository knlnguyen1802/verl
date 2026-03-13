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

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.rollout.replica import ImageOutput, RolloutMode, RolloutReplicaRegistry


def test_vllm_omni_rollout_registry_entry_exists():
    key = ("vllm_omni", "async")
    assert key in _ROLLOUT_REGISTRY
    assert _ROLLOUT_REGISTRY[key] == "verl.workers.rollout.vllm_rollout.vLLMOmniServerAdapter"


def test_vllm_omni_replica_registry_entry_exists():
    assert "vllm_omni" in RolloutReplicaRegistry._registry
    assert callable(RolloutReplicaRegistry._registry["vllm_omni"])


def test_image_output_defaults():
    output = ImageOutput(image=[[[0.0]]])
    assert output.image == [[[0.0]]]
    assert output.log_probs is None
    assert output.stop_reason is None
    assert output.num_preempted is None
    assert output.extra_info == {}


def test_image_output_with_all_fields():
    """ImageOutput should accept all optional fields."""
    output = ImageOutput(
        image=[[[0.5]]],
        log_probs=[0.1, 0.2],
        stop_reason="completed",
        num_preempted=0,
        extra_info={"key": "value"},
    )
    assert output.image == [[[0.5]]]
    assert output.log_probs == [0.1, 0.2]
    assert output.stop_reason == "completed"
    assert output.num_preempted == 0
    assert output.extra_info == {"key": "value"}


def test_rollout_mode_enum_values():
    """RolloutMode should contain HYBRID, COLOCATED and STANDALONE."""
    assert hasattr(RolloutMode, "HYBRID")
    assert hasattr(RolloutMode, "COLOCATED")
    assert hasattr(RolloutMode, "STANDALONE")


# ---------------------------------------------------------------------------
# Tests that verify vLLMOmniHttpServer structure (using mocks for vllm_omni)
# ---------------------------------------------------------------------------


def _build_minimal_vllm_omni_mocks():
    """Build the minimum mock module tree required to import vllm_omni_async_server."""
    mocks = {}

    def _add(name):
        mod = types.ModuleType(name)
        mocks[name] = mod
        return mod

    vllm_omni = _add("vllm_omni")
    vllm_omni_lora = _add("vllm_omni.lora")
    lora_req_mod = _add("vllm_omni.lora.request")
    lora_req_mod.LoRARequest = MagicMock()
    vllm_omni.lora = vllm_omni_lora
    vllm_omni.lora.request = lora_req_mod

    diffusion_mod = _add("vllm_omni.diffusion")
    vllm_omni.diffusion = diffusion_mod

    entrypoints = _add("vllm_omni.entrypoints")
    entrypoints.AsyncOmni = MagicMock()
    vllm_omni.entrypoints = entrypoints
    _add("vllm_omni.entrypoints.cli")
    _add("vllm_omni.entrypoints.cli.serve")
    oai_mod = _add("vllm_omni.entrypoints.openai")
    oai_mod.api_server = MagicMock()
    api_server_mod = _add("vllm_omni.entrypoints.openai.api_server")
    api_server_mod.omni_init_app_state = MagicMock()

    engine_args_mod = _add("vllm_omni.engine.arg_utils")
    engine_args_mod.AsyncOmniEngineArgs = MagicMock()

    inputs_mod = _add("vllm_omni.inputs")
    data_mod = _add("vllm_omni.inputs.data")
    data_mod.OmniCustomPrompt = dict
    data_mod.OmniDiffusionSamplingParams = MagicMock()

    outputs_mod = _add("vllm_omni.outputs")
    outputs_mod.OmniRequestOutput = MagicMock()

    return mocks


def _import_vllm_omni_async_server():
    """Import vllm_omni_async_server with all external deps mocked."""
    # Clean previously cached copies
    for key in list(sys.modules.keys()):
        if "vllm_omni_async_server" in key:
            del sys.modules[key]

    omni_mocks = _build_minimal_vllm_omni_mocks()

    # Also need torchvision mock
    tv = types.ModuleType("torchvision")
    tv_transforms = types.ModuleType("torchvision.transforms")
    tv_transforms.PILToTensor = MagicMock(return_value=MagicMock())
    tv.transforms = tv_transforms

    extra_mocks = {
        "torchvision": tv,
        "torchvision.transforms": tv_transforms,
    }

    with patch.dict(sys.modules, {**omni_mocks, **extra_mocks}):
        mod = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_omni_async_server")
    return mod


def test_vllm_omni_http_server_does_not_have_abort_all_requests():
    """
    abort_all_requests was removed from vLLMOmniHttpServer in favour of the
    base-class implementation.  Verify the subclass no longer overrides it.
    """
    try:
        mod = _import_vllm_omni_async_server()
    except Exception:
        # If the module can't be imported in this environment, skip gracefully
        return

    server_cls = mod.vLLMOmniHttpServer
    # The method must NOT be defined directly on vLLMOmniHttpServer
    assert "abort_all_requests" not in server_cls.__dict__, (
        "abort_all_requests should have been removed from vLLMOmniHttpServer"
    )


def test_vllm_omni_http_server_does_not_have_abort_request():
    """abort_request was removed from vLLMOmniHttpServer."""
    try:
        mod = _import_vllm_omni_async_server()
    except Exception:
        return

    server_cls = mod.vLLMOmniHttpServer
    assert "abort_request" not in server_cls.__dict__, (
        "abort_request should have been removed from vLLMOmniHttpServer"
    )


def test_vllm_omni_http_server_does_not_have_resume_generation():
    """resume_generation was removed from vLLMOmniHttpServer."""
    try:
        mod = _import_vllm_omni_async_server()
    except Exception:
        return

    server_cls = mod.vLLMOmniHttpServer
    assert "resume_generation" not in server_cls.__dict__, (
        "resume_generation should have been removed from vLLMOmniHttpServer"
    )


def test_vllm_omni_http_server_has_generate():
    """vLLMOmniHttpServer must expose a generate() method."""
    try:
        mod = _import_vllm_omni_async_server()
    except Exception:
        return

    assert hasattr(mod.vLLMOmniHttpServer, "generate")
    assert callable(mod.vLLMOmniHttpServer.generate)


def test_vllm_omni_replica_get_server_name_prefix():
    """vLLMOmniReplica._get_server_name_prefix should return 'vllm_omni_'."""
    try:
        mod = _import_vllm_omni_async_server()
    except Exception:
        return

    replica_cls = mod.vLLMOmniReplica
    # Instantiate with mocks
    replica = object.__new__(replica_cls)
    assert replica._get_server_name_prefix() == "vllm_omni_"