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
"""
CPU-runnable unit tests for vLLMOmniHttpServer and vLLMOmniReplica.

vllm_omni must be installed; otherwise the whole module is skipped gracefully.
GPU/distributed initialisation is bypassed by creating instances via
object.__new__ and manually setting the required attributes.
"""

import pytest
from unittest.mock import MagicMock

# Skip the whole module when vllm_omni is not installed.
pytest.importorskip("vllm_omni", reason="vllm_omni package is not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def http_server_instance():
    """Return a bare vLLMOmniHttpServer instance with __init__ bypassed."""
    from verl.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

    server = object.__new__(vLLMOmniHttpServer)
    server.config = MagicMock()
    server.config.engine_kwargs = {}
    server.model_config = MagicMock()
    server.lora_as_adapter = False
    server.engine = None
    server.node_rank = 0
    server.global_steps = 0
    return server


@pytest.fixture(scope="module")
def replica_instance():
    """Return a bare vLLMOmniReplica instance with __init__ bypassed."""
    from verl.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

    replica = object.__new__(vLLMOmniReplica)
    replica.replica_rank = 0
    replica.config = MagicMock()
    replica.model_config = MagicMock()
    replica.gpus_per_node = 8
    replica.is_reward_model = False
    replica.servers = []
    return replica


# ---------------------------------------------------------------------------
# Tests for vLLMOmniHttpServer template-method hooks
# ---------------------------------------------------------------------------


class TestVLLMOmniHttpServerHooks:
    def test_get_engine_kwargs_key_returns_vllm_omni(self, http_server_instance):
        assert http_server_instance._get_engine_kwargs_key() == "vllm_omni"

    def test_get_wake_up_tags_returns_only_weights(self, http_server_instance):
        """vLLM-Omni does not restore KV cache on wake-up – only weights."""
        tags = http_server_instance._get_wake_up_tags()
        assert tags == ["weights"]

    def test_get_worker_extension_cls(self, http_server_instance):
        cls = http_server_instance._get_worker_extension_cls()
        assert cls == "verl.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"

    def test_get_cli_description(self, http_server_instance):
        assert http_server_instance._get_cli_description() == "vLLM-Omni CLI"

    def test_preprocess_engine_kwargs_removes_custom_pipeline(self, http_server_instance):
        """custom_pipeline key must be removed from engine_kwargs during preprocessing."""
        kwargs = {"custom_pipeline": "some.module.Pipeline", "other_key": "value"}
        http_server_instance._preprocess_engine_kwargs(kwargs)
        assert "custom_pipeline" not in kwargs
        assert "other_key" in kwargs

    def test_preprocess_engine_kwargs_no_op_when_key_absent(self, http_server_instance):
        kwargs = {"tensor_parallel_size": 2}
        http_server_instance._preprocess_engine_kwargs(kwargs)
        assert kwargs == {"tensor_parallel_size": 2}

    def test_get_cli_modules_returns_list(self, http_server_instance):
        modules = http_server_instance._get_cli_modules()
        assert isinstance(modules, list)
        assert len(modules) == 1
        assert getattr(modules[0], "__name__", "") == "vllm_omni.entrypoints.cli.serve"


# ---------------------------------------------------------------------------
# Tests for vLLMOmniReplica
# ---------------------------------------------------------------------------


class TestVLLMOmniReplica:
    def test_get_server_name_prefix(self, replica_instance):
        assert replica_instance._get_server_name_prefix() == "vllm_omni_"
