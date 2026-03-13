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
CPU-runnable unit tests for vLLMOmniServerAdapter.

vllm_omni must be installed; otherwise the whole module is skipped gracefully.
GPU/distributed initialisation is bypassed by creating instances via
object.__new__ and manually setting the required attributes.
"""

import pytest

# Skip the whole module when vllm_omni is not installed.
pytest.importorskip("vllm_omni", reason="vllm_omni package is not installed")


@pytest.fixture()
def adapter_instance():
    """Create a vLLMOmniServerAdapter with __init__ bypassed.

    Attributes are set manually to represent a typical rank-0 worker in a
    tp=2, dp=1, pp=1 configuration.
    """
    from unittest.mock import MagicMock

    from verl.workers.rollout.vllm_rollout.vllm_omni_rollout import vLLMOmniServerAdapter

    adapter = object.__new__(vLLMOmniServerAdapter)

    # Config stub – replicate what __init__ would set.
    adapter.config = type(
        "Cfg",
        (),
        {
            "tensor_model_parallel_size": 2,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
    )()
    adapter.model_config = MagicMock()
    adapter.device_mesh = MagicMock()
    adapter.server_handle = None
    adapter.replica_rank = 0
    adapter.rollout_rank = 0
    adapter.node_rank = 0
    adapter.sleep_level = 1
    adapter.device_uuid = "GPU-fake-uuid-0000"
    adapter.zmq_handle = "ipc:///tmp/rl-colocate-zmq-GPU-fake-uuid-0000.sock"
    adapter.use_shm = False

    return adapter


# ---------------------------------------------------------------------------
# Tests for vLLMOmniServerAdapter._execute_method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_method_returns_none_for_non_zero_rollout_rank(adapter_instance):
    """Non-rank-0 workers must short-circuit and return None immediately."""
    adapter_instance.rollout_rank = 1
    result = await adapter_instance._execute_method("some_method")
    assert result is None


@pytest.mark.asyncio
async def test_execute_method_returns_none_for_rollout_rank_2(adapter_instance):
    """Same short-circuit behaviour for any rank != 0."""
    adapter_instance.rollout_rank = 2
    result = await adapter_instance._execute_method("some_method")
    assert result is None


@pytest.mark.asyncio
async def test_execute_method_lazily_inits_server_handle(adapter_instance):
    """On rank-0, server_handle is fetched via ray.get_actor on the first call."""
    import ray
    from unittest.mock import AsyncMock, MagicMock

    mock_handle = MagicMock()
    mock_handle.collective_rpc.remote = MagicMock(return_value=AsyncMock(return_value="result")())
    ray.get_actor = MagicMock(return_value=mock_handle)

    adapter_instance.rollout_rank = 0
    adapter_instance.server_handle = None

    result = await adapter_instance._execute_method("some_method")

    ray.get_actor.assert_called_once_with("vllm_omni_server_0_0")
    assert adapter_instance.server_handle is mock_handle
    assert result == "result"


@pytest.mark.asyncio
async def test_execute_method_reuses_existing_server_handle(adapter_instance):
    """If server_handle is already set, ray.get_actor must NOT be called again."""
    import ray
    from unittest.mock import AsyncMock, MagicMock

    mock_handle = MagicMock()
    mock_handle.collective_rpc.remote = MagicMock(return_value=AsyncMock(return_value="cached_result")())
    ray.get_actor = MagicMock()

    adapter_instance.rollout_rank = 0
    adapter_instance.server_handle = mock_handle  # already initialised

    result = await adapter_instance._execute_method("another_method")

    ray.get_actor.assert_not_called()
    assert result == "cached_result"


@pytest.mark.asyncio
async def test_execute_method_non_block_returns_future(adapter_instance):
    """With non_block=True the raw future (not the awaited result) is returned."""
    import ray
    from unittest.mock import MagicMock

    mock_handle = MagicMock()
    future = MagicMock(name="future")
    mock_handle.collective_rpc.remote = MagicMock(return_value=future)
    ray.get_actor = MagicMock(return_value=mock_handle)

    adapter_instance.rollout_rank = 0
    adapter_instance.server_handle = None

    result = await adapter_instance._execute_method("method", non_block=True)

    # In non_block mode the future itself is returned without awaiting.
    assert result is future


# ---------------------------------------------------------------------------
# Rank / node computation sanity checks
# ---------------------------------------------------------------------------


def test_replica_rank_computed_from_rank_env(adapter_instance):
    """replica_rank = rank // rollout_world_size (world_size = tp*dp*pp = 2)."""
    adapter_instance.replica_rank = 4 // 2
    assert adapter_instance.replica_rank == 2


def test_rollout_rank_modulo(adapter_instance):
    """rollout_rank = rank % rollout_world_size."""
    adapter_instance.rollout_rank = 3 % 2
    assert adapter_instance.rollout_rank == 1


def test_node_rank_derived_from_rollout_rank(adapter_instance):
    """node_rank = rollout_rank // local_world_size."""
    adapter_instance.rollout_rank = 5
    adapter_instance.node_rank = 5 // 4  # local_world_size = 4
    assert adapter_instance.node_rank == 1


def test_zmq_handle_format(adapter_instance):
    adapter_instance.device_uuid = "GPU-test-abc"
    adapter_instance.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{adapter_instance.device_uuid}.sock"
    assert adapter_instance.zmq_handle == "ipc:///tmp/rl-colocate-zmq-GPU-test-abc.sock"


def test_sleep_level_default(adapter_instance):
    assert adapter_instance.sleep_level == 1


def test_use_shm_default_false(adapter_instance):
    assert adapter_instance.use_shm is False
