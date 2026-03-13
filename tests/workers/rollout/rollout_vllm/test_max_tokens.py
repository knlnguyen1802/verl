# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Unit tests for the max_tokens calculation in vLLMHttpServer.generate().

The formula changed from:
    max_tokens = config.response_length + config.prompt_length - len(prompt_ids)
to:
    max_tokens = min(
        config.response_length,
        config.prompt_length + config.response_length - len(prompt_ids),
    )

This ensures max_tokens never exceeds response_length (tensor alignment) while
still respecting the remaining context budget.
"""

import pytest


def compute_max_tokens(response_length: int, prompt_length: int, actual_prompt_len: int) -> int:
    """
    Replicate the max_tokens calculation from vLLMHttpServer.generate()
    so that the pure logic can be tested without importing heavy dependencies.
    """
    return min(
        response_length,
        prompt_length + response_length - actual_prompt_len,
    )


class TestMaxTokensCalculation:
    """Tests for the default max_tokens calculation in vLLMHttpServer.generate()."""

    def test_short_prompt_returns_response_length(self):
        """When prompt is short, max_tokens should be capped at response_length."""
        # prompt_length=512, response_length=512, actual=100
        # old formula: 512+512-100 = 924
        # new formula: min(512, 924) = 512
        result = compute_max_tokens(response_length=512, prompt_length=512, actual_prompt_len=100)
        assert result == 512

    def test_empty_prompt_returns_response_length(self):
        """Zero-length prompt: remaining budget > response_length, cap at response_length."""
        result = compute_max_tokens(response_length=512, prompt_length=512, actual_prompt_len=0)
        assert result == 512

    def test_prompt_equals_prompt_length_returns_response_length(self):
        """When actual prompt length equals configured prompt_length, returns exactly response_length."""
        result = compute_max_tokens(response_length=512, prompt_length=512, actual_prompt_len=512)
        assert result == 512

    def test_multi_turn_prompt_exceeds_prompt_length(self):
        """Multi-turn: prompt exceeds prompt_length, budget is reduced below response_length."""
        # prompt_length=512, response_length=512, actual=700
        # remaining budget: 512+512-700 = 324
        # new formula: min(512, 324) = 324
        result = compute_max_tokens(response_length=512, prompt_length=512, actual_prompt_len=700)
        assert result == 324

    def test_budget_equals_response_length(self):
        """Remaining budget exactly equals response_length — should return response_length."""
        # prompt_length=512, response_length=256, actual=512
        # remaining budget: 512+256-512 = 256 == response_length
        result = compute_max_tokens(response_length=256, prompt_length=512, actual_prompt_len=512)
        assert result == 256

    def test_very_long_prompt_returns_small_budget(self):
        """Very long prompt leaves only a small generation budget."""
        # prompt_length=512, response_length=512, actual=900
        # remaining: 512+512-900 = 124
        result = compute_max_tokens(response_length=512, prompt_length=512, actual_prompt_len=900)
        assert result == 124

    def test_asymmetric_lengths(self):
        """Asymmetric prompt/response lengths."""
        # prompt_length=256, response_length=128, actual=200
        # remaining: 256+128-200 = 184
        # new formula: min(128, 184) = 128
        result = compute_max_tokens(response_length=128, prompt_length=256, actual_prompt_len=200)
        assert result == 128

    def test_asymmetric_with_overflow(self):
        """When actual prompt overflows the budget, returns the leftover budget."""
        # prompt_length=256, response_length=128, actual=300
        # remaining: 256+128-300 = 84
        result = compute_max_tokens(response_length=128, prompt_length=256, actual_prompt_len=300)
        assert result == 84

    def test_result_is_always_le_response_length(self):
        """The result must never exceed response_length (key invariant)."""
        for actual in range(0, 1025, 50):
            result = compute_max_tokens(response_length=512, prompt_length=512, actual_prompt_len=actual)
            assert result <= 512, f"result {result} > response_length 512 for actual_prompt_len={actual}"

    def test_old_formula_would_exceed_response_length(self):
        """Demonstrate that the old formula could produce values > response_length."""
        response_length = 512
        prompt_length = 512
        actual_prompt_len = 100  # shorter than configured prompt_length
        old_result = response_length + prompt_length - actual_prompt_len  # = 924
        new_result = compute_max_tokens(response_length, prompt_length, actual_prompt_len)
        assert old_result > response_length, "Old formula would exceed response_length"
        assert new_result <= response_length, "New formula must cap at response_length"
        assert new_result == response_length
