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
CPU-runnable unit tests for verl/workers/engine/veomni/utils.py.

Only torch and verl.utils.device are imported at module level by utils.py,
so these tests work in any CPU environment where torch is available.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from verl.workers.engine.veomni.utils import (  # noqa: E402
    MOE_PARAM_HANDERS,
    VL_TYPE2INDEX,
    _map_moe_params_qwen3_moe,
)


# ---------------------------------------------------------------------------
# Tests for VL_TYPE2INDEX
# ---------------------------------------------------------------------------


class TestVLType2Index:
    """Verify the shape and content of the VL_TYPE2INDEX look-up table."""

    def test_contains_qwen2_5_vl(self):
        assert "qwen2_5_vl" in VL_TYPE2INDEX

    def test_contains_qwen3_vl(self):
        assert "qwen3_vl" in VL_TYPE2INDEX

    def test_contains_qwen3_vl_moe(self):
        assert "qwen3_vl_moe" in VL_TYPE2INDEX

    def test_all_entries_have_image_input_index(self):
        for name, indices in VL_TYPE2INDEX.items():
            assert "IMAGE_INPUT_INDEX" in indices, f"Missing IMAGE_INPUT_INDEX for {name}"

    def test_all_entries_have_video_input_index(self):
        for name, indices in VL_TYPE2INDEX.items():
            assert "VIDEO_INPUT_INDEX" in indices, f"Missing VIDEO_INPUT_INDEX for {name}"

    def test_image_input_index_values(self):
        """All model types share the same special token ID for image input."""
        for name, indices in VL_TYPE2INDEX.items():
            assert indices["IMAGE_INPUT_INDEX"] == 151655, f"Unexpected IMAGE_INPUT_INDEX for {name}"

    def test_video_input_index_values(self):
        """All model types share the same special token ID for video input."""
        for name, indices in VL_TYPE2INDEX.items():
            assert indices["VIDEO_INPUT_INDEX"] == 151656, f"Unexpected VIDEO_INPUT_INDEX for {name}"

    def test_number_of_entries(self):
        """Exactly three model families are registered."""
        assert len(VL_TYPE2INDEX) == 3

    def test_indices_are_positive_integers(self):
        for name, indices in VL_TYPE2INDEX.items():
            for key, val in indices.items():
                assert isinstance(val, int), f"Index {key} for {name} is not an int: {val!r}"
                assert val > 0, f"Index {key} for {name} must be positive, got {val}"

    def test_image_and_video_indices_are_different(self):
        """Image and video token IDs must not collide."""
        for name, indices in VL_TYPE2INDEX.items():
            assert indices["IMAGE_INPUT_INDEX"] != indices["VIDEO_INPUT_INDEX"], (
                f"IMAGE_INPUT_INDEX and VIDEO_INPUT_INDEX are the same for {name}"
            )


# ---------------------------------------------------------------------------
# Tests for MOE_PARAM_HANDERS
# ---------------------------------------------------------------------------


class TestMOEParamHandlers:
    """Verify the structure of the MOE_PARAM_HANDERS dispatch table."""

    def test_contains_qwen3_moe(self):
        assert "qwen3_moe" in MOE_PARAM_HANDERS

    def test_all_handlers_are_callable(self):
        for name, handler in MOE_PARAM_HANDERS.items():
            assert callable(handler), f"Handler for {name} is not callable"

    def test_qwen3_moe_handler_is_the_private_function(self):
        """The qwen3_moe handler should be _map_moe_params_qwen3_moe."""
        assert MOE_PARAM_HANDERS["qwen3_moe"] is _map_moe_params_qwen3_moe

    def test_handler_dict_is_not_empty(self):
        assert len(MOE_PARAM_HANDERS) >= 1
