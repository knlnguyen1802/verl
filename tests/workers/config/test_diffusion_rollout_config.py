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

from verl.workers.config.rollout import DiffusionRolloutConfig, DiffusionSamplingConfig, RolloutConfig


class TestDiffusionSamplingConfig:
    def test_default_values(self):
        config = DiffusionSamplingConfig()
        assert config.do_sample is True
        assert config.n == 1
        assert config.noise_level == 0.0
        assert config.num_inference_steps == 40
        assert config.seed == 42


class TestDiffusionRolloutConfig:
    def test_default_values(self):
        config = DiffusionRolloutConfig(name="vllm_omni")
        assert config.name == "vllm_omni"
        assert config.mode == "async"
        assert config.val_kwargs == DiffusionSamplingConfig()
        assert config.sde_type == "sde"

    def test_sync_mode_raises(self):
        with pytest.raises(ValueError, match="Rollout mode 'sync' has been removed"):
            DiffusionRolloutConfig(name="vllm_omni", mode="sync")

    def test_pipeline_parallel_not_supported_for_vllm_omni(self):
        with pytest.raises(NotImplementedError, match="not implemented pipeline_model_parallel_size > 1"):
            DiffusionRolloutConfig(name="vllm_omni", pipeline_model_parallel_size=2)

    def test_inherits_from_rollout_config(self):
        """DiffusionRolloutConfig should inherit from RolloutConfig."""
        assert issubclass(DiffusionRolloutConfig, RolloutConfig)

    def test_inherits_rollout_config_fields(self):
        """DiffusionRolloutConfig should have all base RolloutConfig fields."""
        config = DiffusionRolloutConfig(name="vllm_omni")
        # Fields inherited from RolloutConfig
        assert config.prompt_length == 512
        assert config.response_length == 512
        assert config.dtype == "bfloat16"
        assert config.gpu_memory_utilization == 0.5
        assert config.tensor_model_parallel_size == 2
        assert config.data_parallel_size == 1
        assert config.pipeline_model_parallel_size == 1
        assert config.enable_chunked_prefill is True
        assert config.enable_prefix_caching is True
        assert config.load_format == "dummy"

    def test_diffusion_specific_fields(self):
        """DiffusionRolloutConfig should have its own diffusion-specific fields."""
        config = DiffusionRolloutConfig(name="vllm_omni")
        assert config.image_height == 512
        assert config.image_width == 512
        assert config.num_inference_steps == 10
        assert config.noise_level == 0.7
        assert config.guidance_scale == 4.5
        assert config.sde_type == "sde"
        assert config.sde_window_size is None
        assert config.sde_window_range is None

    def test_val_kwargs_is_diffusion_sampling_config(self):
        """val_kwargs should be DiffusionSamplingConfig, not base SamplingConfig."""
        config = DiffusionRolloutConfig(name="vllm_omni")
        assert isinstance(config.val_kwargs, DiffusionSamplingConfig)

    def test_pipeline_parallel_ok_for_non_vllm_omni(self):
        """Pipeline parallelism > 1 should only raise for vllm_omni."""
        # Should not raise for other rollout names
        config = DiffusionRolloutConfig(name="other_rollout", pipeline_model_parallel_size=2)
        assert config.pipeline_model_parallel_size == 2

    def test_super_post_init_called(self):
        """__post_init__ must invoke super().__post_init__(), so RolloutConfig validation applies."""
        # The sync-mode check lives in RolloutConfig.__post_init__; DiffusionRolloutConfig
        # must call super().__post_init__() for it to propagate.
        with pytest.raises(ValueError, match="Rollout mode 'sync' has been removed"):
            DiffusionRolloutConfig(name="vllm_omni", mode="sync")

    def test_mutable_fields(self):
        """DiffusionRolloutConfig._mutable_fields contains diffusion-specific overrides."""
        assert "max_model_len" in DiffusionRolloutConfig._mutable_fields
        assert "load_format" in DiffusionRolloutConfig._mutable_fields

    def test_custom_image_dimensions(self):
        """Image height and width can be overridden."""
        config = DiffusionRolloutConfig(name="vllm_omni", image_height=1024, image_width=768)
        assert config.image_height == 1024
        assert config.image_width == 768
