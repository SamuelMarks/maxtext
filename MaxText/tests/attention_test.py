#  Copyright 2023–2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for Attentions."""

import itertools
import os.path
import random
import sys
import unittest
import json  # For debugging output
from typing import Any

import yaml  # For loading YAML

import pytest

from absl.testing import parameterized

import numpy as np

from jax.sharding import Mesh
import jax
import jax.numpy as jnp

from flax.core import freeze

from MaxText import maxtext_utils
# from MaxText import pyconfig # No longer used
from MaxText.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR  # MODEL_MODE enums are now in type_h
from MaxText.globals import PKG_DIR
from MaxText.layers import attentions
from MaxText.layers.attentions import Attention, MLA, ChunkedCausalMask

# Import Pydantic models and utilities
from MaxText.configs.type_h import (
    MaxTextConfig,
    IciParallelismConfig,
    ModelCallMode,
    AttentionType,
    RoPEType,
    AttentionKernel,
    DatasetType,
    HardwareType,
    OptimizerType,
    DecoderBlockType,
    DcnParallelismConfig,
)
from MaxText.configs.utils import merge_pydantic_models, config_to_flat_dict


class BidirectionalBlockMaskTest(unittest.TestCase):
  """Test for make_bidirectional_block_mask."""

  def test_one_block_mask(self):
    bidirectional_mask = np.asarray([[0, 1, 1, 1, 0, 0]])
    # pylint: disable=protected-access
    block_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.asarray(
        [
            [
                [False, False, False, False, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
            ]
        ]
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_two_blocks_mask(self):
    bidirectional_mask = np.asarray([[0, 1, 1, 0, 1, 1]])
    # pylint: disable=protected-access
    block_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.asarray(
        [
            [
                [False, False, False, False, False, False],
                [False, True, True, False, False, False],
                [False, True, True, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, True, True],
                [False, False, False, False, True, True],
            ]
        ]
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_batch_block_masks(self):
    bidirectional_mask = np.asarray([[0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1]])
    # pylint: disable=protected-access
    block_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.asarray(
        [
            [
                [False, False, False, False, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
            ],
            [
                [False, False, False, False, False, False],
                [False, True, True, False, False, False],
                [False, True, True, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, True, True],
                [False, False, False, False, True, True],
            ],
        ]
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_empty_block_mask(self):
    bidirectional_mask = np.asarray([[0, 0, 0, 0, 0, 0]])
    # pylint: disable=protected-access
    block_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.zeros(
        (bidirectional_mask.shape[0], bidirectional_mask.shape[1], bidirectional_mask.shape[1]), dtype=bool
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_full_block_mask(self):
    bidirectional_mask = np.asarray([[1, 1, 1, 1, 1, 1]])
    # pylint: disable=protected-access
    block_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.ones(
        (bidirectional_mask.shape[0], bidirectional_mask.shape[1], bidirectional_mask.shape[1]), dtype=bool
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_combine_with_causal_mask(self):
    seq_len = 6
    row_ids = np.arange(seq_len, dtype=np.int32)[:, None]
    col_ids = np.arange(seq_len, dtype=np.int32)[None, :]
    causal_mask = (col_ids <= row_ids)[None, None, None, :, :]
    bidirectional_mask = np.asarray([[0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1]])
    # pylint: disable=protected-access
    image_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
    combined_mask = causal_mask | image_mask[:, None, None, ...]
    expected_mask = np.asarray(
        [
            [
                [
                    [
                        [True, False, False, False, False, False],
                        [True, True, True, True, False, False],
                        [True, True, True, True, False, False],
                        [True, True, True, True, False, False],
                        [True, True, True, True, True, False],
                        [True, True, True, True, True, True],
                    ]
                ]
            ],
            [
                [
                    [
                        [True, False, False, False, False, False],
                        [True, True, True, False, False, False],
                        [True, True, True, False, False, False],
                        [True, True, True, True, False, False],
                        [True, True, True, True, True, True],
                        [True, True, True, True, True, True],
                    ]
                ]
            ],
        ]
    )
    np.testing.assert_array_equal(combined_mask, expected_mask)


class ChunkedCausalMaskTest(unittest.TestCase):
  """Test for the ChunkedCausalMask."""

  def test_basic_chunking(self):
    """Tests the mask with a simple chunk size."""
    seq_len = 8
    chunk_size = 4
    mask = ChunkedCausalMask(shape=(seq_len, seq_len), chunk_size=chunk_size)

    # Manually compute the expected mask
    # Causal within chunks (0-3, 4-7)
    expected_mask = np.zeros((seq_len, seq_len), dtype=np.bool_)
    for r in range(seq_len):
      for c in range(seq_len):
        q_chunk = r // chunk_size
        kv_chunk = c // chunk_size
        if q_chunk == kv_chunk and r >= c:
          expected_mask[r, c] = True

    # Get the actual mask by slicing
    actual_mask = mask[:, :]

    np.testing.assert_array_equal(actual_mask, expected_mask)
    # Make sure _generate_chunk_attention_mask also produces the same mask
    # pylint: disable=protected-access
    actual_mask = attentions._generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual_mask, expected_mask)

  def test_full_length_chunk(self):
    """Tests when chunk size equals sequence length (should be causal)."""
    seq_len = 6
    chunk_size = 6  # Same as seq_len
    mask = ChunkedCausalMask(shape=(seq_len, seq_len), chunk_size=chunk_size)

    # Expected mask is a standard lower triangular causal mask
    expected_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.bool_))

    actual_mask = mask[:, :]
    np.testing.assert_array_equal(actual_mask, expected_mask)
    # Make sure _generate_chunk_attention_mask also produces the same mask
    # pylint: disable=protected-access
    actual_mask = attentions._generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual_mask, expected_mask)

  def test_single_token_chunk(self):
    """Tests when chunk size is 1 (only attend to self)."""
    seq_len = 5
    chunk_size = 1
    mask = ChunkedCausalMask(shape=(seq_len, seq_len), chunk_size=chunk_size)

    # Expected mask is just the identity matrix
    expected_mask = np.eye(seq_len, dtype=np.bool_)

    actual_mask = mask[:, :]
    np.testing.assert_array_equal(actual_mask, expected_mask)
    # Make sure _generate_chunk_attention_mask also produces the same mask
    # pylint: disable=protected-access
    actual_mask = attentions._generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual_mask, expected_mask)

  def test_non_square_shape(self):
    """Tests with different query and key sequence lengths."""
    q_len = 6
    kv_len = 8
    chunk_size = 3
    mask = ChunkedCausalMask(shape=(q_len, kv_len), chunk_size=chunk_size)

    # Manually compute expected mask
    expected_mask = np.zeros((q_len, kv_len), dtype=np.bool_)
    for r in range(q_len):
      for c in range(kv_len):
        q_chunk = r // chunk_size
        kv_chunk = c // chunk_size
        if q_chunk == kv_chunk and r >= c:
          expected_mask[r, c] = True

    actual_mask = mask[:, :]
    np.testing.assert_array_equal(actual_mask, expected_mask)
    # Make sure _generate_chunk_attention_mask also produces the same mask
    # pylint: disable=protected-access
    actual_mask = attentions._generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual_mask, expected_mask)

  def test_value_error_on_zero_chunk_size(self):
    """Tests that a ValueError is raised for chunk_size <= 0."""
    with self.assertRaises(ValueError):
      ChunkedCausalMask(shape=(4, 4), chunk_size=0)
    with self.assertRaises(ValueError):
      ChunkedCausalMask(shape=(4, 4), chunk_size=-2)
    with self.assertRaises(ValueError):
      # pylint: disable=protected-access
      attentions._generate_chunk_attention_mask(mask_shape=(4, 4), chunk_size=0)


class AttentionTest(unittest.TestCase):
  """Test for the Attention"""

  # Note: if you are changing these configs, please make sure to change the configs in
  # context_parallelism.py as well, since we are using the same configs for both
  # tests to get the same mesh and other config
  # These are now structured as a dictionary to be passed to MaxTextConfig for overriding.
  # The keys should match the attribute names in MaxTextConfig and its sub-models.
  config_override_dict = {
      "dataset_sources": {"base": {"per_device_batch_size": 1.0}},  # Updated path
      "run_settings": {"run_name": "test"},  # Updated path
      "checkpoint_save_settings": {"enable_checkpointing": False},  # Updated path
      "training_settings": {"max_prefill_predict_length": 16, "max_target_length": 512},  # Updated path
      "splash_attention_config": {  # Updated path
          "sa_block_q": 128,
          "sa_block_kv": 128,
          "sa_block_kv_compute": 128,
          "sa_block_q_dkv": 128,
          "sa_block_kv_dkv": 128,
          "sa_block_kv_dkv_compute": 128,
          "sa_block_q_dq": 128,
          "sa_block_kv_dq": 128,
      },
  }

  def setUp(self):
    super().setUp()
    with open(os.path.join(PKG_DIR, "configs", "base.yml"), "rt", encoding="utf8") as f:
      base_config_dict = yaml.safe_load(f)

    base_pydantic_config = MaxTextConfig.model_validate(base_config_dict)
    override_pydantic_config = MaxTextConfig.model_validate(self.config_override_dict)
    self.cfg = merge_pydantic_models(base_pydantic_config, override_pydantic_config)
    # For debugging JSON output:
    # with open("/tmp/attention_test_cfg.json", "w") as f:
    #     json.dump(config_to_flat_dict(self.cfg), f, indent=2, default=str)

    # Configuration for context parallelism test (cfg_cp)
    # Start with a deep copy of the merged self.cfg, then apply specific CP overrides
    cp_override_dict = {
        "ici_parallelism_settings": {"ici_context_parallelism": 4},  # Path to ICIParallelismConfig instance
        "mesh_layout": {"context_parallel_load_balance": False},  # Path to MeshConfig instance
    }
    # We need to be careful when merging sub-objects that might already exist.
    # merge_pydantic_models handles deep merging of nested Pydantic models.
    cfg_cp_base = self.cfg.model_copy(deep=True)
    cp_override_pydantic_config = MaxTextConfig.model_validate(cp_override_dict)
    self.cfg_cp = merge_pydantic_models(cfg_cp_base, cp_override_pydantic_config)

    self.rng = jax.random.PRNGKey(0)

    devices_array = maxtext_utils.create_device_mesh_with_maxtextconfig(self.cfg)  # Pass the Pydantic config
    self.mesh = Mesh(devices_array, self.cfg.mesh_layout.mesh_axes)  # Access new structure
    devices_array_cp = maxtext_utils.create_device_mesh_with_maxtextconfig(self.cfg_cp)
    self.mesh_cp = Mesh(devices_array_cp, self.cfg_cp.mesh_layout.mesh_axes)

    self.global_batch_size = self.cfg.global_batch_info.global_batch_size_to_train_on  # Updated path
    self.num_kv_heads = self.cfg.model_architecture.num_kv_heads  # Updated path
    self.num_query_heads = self.cfg.model_architecture.num_query_heads  # Updated path
    self.max_target_length = self.cfg.training_settings.max_target_length  # Updated path
    self.max_prefill_predict_length = self.cfg.training_settings.max_prefill_predict_length  # Updated path
    self.head_dim = self.cfg.model_architecture.head_dim  # Updated path
    self.embed_dim = self.cfg.model_architecture.emb_dim  # Updated path (no more base_ prefix)
    self.dtype_str = self.cfg.model_quantization_config.dtype  # Updated path
    try:
      self.dtype = getattr(jnp, self.dtype_str)
    except AttributeError:
      self.dtype = jnp.float32  # Fallback

    self.attention_type_enum = self.cfg.attention_kernel_config.attention_type  # Updated path,  get enum value

    self._attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel=AttentionKernel.DOT_PRODUCT.value,  # Pass string value
        dtype=self.dtype,
        dropout_rate=self.cfg.activations_logits_config.dropout_rate,  # Updated path
        name="self_attention",
        attention_type=self.attention_type_enum,  # Pass enum member
    )

    self._attention_as_mha_generic_variable = self._attention_as_mha_generic.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

  def get_data(self, dtype):
    """get data"""
    # Use already resolved self.global_batch_size etc.
    lnx = jax.random.normal(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=dtype,
    )

    decoder_segment_ids = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(
        self.rng, (self.global_batch_size, self.max_target_length), 0, self.max_target_length
    )

    return lnx, decoder_segment_ids, decoder_positions

  def get_structured_data(self, dtype):
    """get structured data"""
    lnx = jax.random.normal(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=dtype,
    )

    decoder_positions = jnp.stack(
        [jnp.arange(self.max_target_length, dtype=np.int32) for _ in range(self.global_batch_size)]
    )

    decoder_segment_ids = (
        jax.numpy.zeros((self.global_batch_size, self.max_target_length)) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    )

    return lnx, decoder_segment_ids, decoder_positions

  @pytest.mark.tpu_only
  def test_autoregression(self):
    prefill_length = self.max_prefill_predict_length  # Use attribute from setUp
    decode_total_length = self.max_target_length  # Use attribute from setUp
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(self.dtype)

    mha_full = self._attention_as_mha_generic.apply(
        self._attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,  # Use Enum from type_h
        rngs={"aqt": self.rng},
    )

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    mha_prefill, output_cache = self._attention_as_mha_generic.apply(
        self._attention_as_mha_generic_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=ModelCallMode.PREFILL,  # Use Enum from type_h
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )

    self.assertTrue(
        jax.numpy.allclose(mha_prefill, mha_full[:, :prefill_length, :], rtol=1e-02, atol=1e-02, equal_nan=False)
    )

    attention_vars_for_loop = self._attention_as_mha_generic_variable.copy()  # Make a mutable copy for the loop
    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      if "cache" in output_cache:  # Check if cache needs to be updated
        attention_vars_for_loop.update(output_cache)

      mha_idx, output_cache = self._attention_as_mha_generic.apply(
          attention_vars_for_loop,  # Use the loop-local variables
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=ModelCallMode.AUTOREGRESSIVE,  # Use Enum from type_h
          rngs={"aqt": self.rng},
          mutable=["cache"],
      )

      mha_full_this_idx = mha_full[:, idx : idx + 1, :]
      self.assertTrue(mha_full_this_idx.shape == mha_idx.shape)
      self.assertTrue(jax.numpy.allclose(mha_full_this_idx, mha_idx, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu_only
  def test_model_mode_prefill_dtype_float32(self):
    self._test_model_mode_prefill_dtype(jnp.float32)

  @pytest.mark.tpu_only
  def test_model_mode_prefill_dtype_bfloat16(self):
    """test model mode prefill for dtype bfloat16"""
    self._test_model_mode_prefill_dtype(jnp.bfloat16)

  def _test_model_mode_prefill_dtype(self, dtype_jnp):
    """test model mode prefill for specified dtype"""
    lnx, decoder_segment_ids, decoder_positions = self.get_data(dtype_jnp)  # Pass jnp dtype
    prefill_length = self.max_prefill_predict_length
    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    # Create a config with the specified dtype for this test
    temp_override_dict = {
        "model_quantization_config": {"dtype": แม็กซ์เท็กซ์ยูทิลิตี้.ดีไทป์ทูสตริง(ดีไทป์เจเอ็นพี)}
    }  # Example for how to set specific dtype if needed temporarily

    # Create a new config instance for this specific test, or modify self.cfg if appropriate for the test's scope
    # For an isolated test, better to create a new config object or copy and modify
    cfg_for_test = self.cfg.model_copy(deep=True)
    cfg_for_test.model_quantization_config.dtype = maxtext_utils.dtype_to_string(dtype_jnp)

    attention_as_mha_generic = Attention(
        config=cfg_for_test,  # Use modified config for this test
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=cfg_for_test.training_settings.max_prefill_predict_length,  # Use from new config
        mesh=self.mesh,
        attention_kernel=AttentionKernel.DOT_PRODUCT.value,  # Pass string value
        dtype=dtype_jnp,  # Pass jnp dtype
        dropout_rate=cfg_for_test.activations_logits_config.dropout_rate,
        name="self_attention",
    )

    attention_as_mha_generic_variable = attention_as_mha_generic.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    mha_prefill, _ = attention_as_mha_generic.apply(
        attention_as_mha_generic_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=ModelCallMode.PREFILL,  # Use Enum
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )

    self.assertEqual(dtype_jnp, mha_prefill.dtype)

  @pytest.mark.tpu_only
  def test_tpu_kernel_attention_mha(self):
    self.tpu_kernel_attention_helper(self.num_kv_heads)

  @pytest.mark.tpu_only
  def test_tpu_kernel_attention_gqa(self):
    self.tpu_kernel_attention_helper(self.num_kv_heads // 2)

  @pytest.mark.tpu_only
  def test_tpu_kernel_attention_mqa(self):
    self.tpu_kernel_attention_helper(1)

  def tpu_kernel_attention_helper(self, num_kv_heads_override):
    """Test equivalence between dot_product and TPU accelerated"""

    lnx, decoder_segment_ids, decoder_positions = self.get_data(self.dtype)

    attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=num_kv_heads_override,  # Use override
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel=AttentionKernel.DOT_PRODUCT.value,
        dtype=self.dtype,
        dropout_rate=self.cfg.activations_logits_config.dropout_rate,
        name="self_attention",
    )

    attention_as_mha_generic_variable = attention_as_mha_generic.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    mha_generic_output = attention_as_mha_generic.apply(
        attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,  # Note: Original test swaps these for some reason
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_as_mha_flash = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=num_kv_heads_override,  # Use override
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel=AttentionKernel.FLASH.value,  # Use flash
        dtype=self.dtype,
        dropout_rate=self.cfg.activations_logits_config.dropout_rate,
        name="self_attention",  # Re-using name, params should be compatible if only kernel changes
    )
    # Re-use variables if kernels are parameter-compatible
    # Initializing again for safety if params structure differs by kernel, though ideally not.
    attention_as_mha_flash_variable = attention_as_mha_flash.init(
        {"params": self.rng, "aqt": self.rng},  # Re-init, or use attention_as_mha_generic_variable if compatible
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    mha_generic_flash_output = attention_as_mha_flash.apply(
        attention_as_mha_flash_variable,  # For flash
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,
        rngs={"aqt": self.rng},
    )

    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_output, rtol=1e-01, atol=1e-01, equal_nan=False)
    )

    # Test with Context Parallelism
    attention_as_mha_flash_cp = Attention(
        config=self.cfg_cp,
        num_query_heads=self.cfg_cp.model_architecture.num_query_heads,  # Use cfg_cp
        num_kv_heads=num_kv_heads_override,  # Use override
        head_dim=self.cfg_cp.model_architecture.head_dim,
        max_target_length=self.cfg_cp.training_settings.max_target_length,
        max_prefill_predict_length=self.cfg_cp.training_settings.max_prefill_predict_length,
        mesh=self.mesh_cp,
        attention_kernel=AttentionKernel.FLASH.value,  # Use flash
        dtype=self.dtype,  # Assuming dtype is same for this comparison
        dropout_rate=self.cfg_cp.activations_logits_config.dropout_rate,
        name="self_attention_cp",  # Different name for CP version
    )
    attention_as_mha_flash_cp_variable = attention_as_mha_flash_cp.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    mha_generic_flash_cp_output = attention_as_mha_flash_cp.apply(
        attention_as_mha_flash_cp_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,
        rngs={"aqt": self.rng},
    )

    self.assertTrue(
        jax.numpy.allclose(mha_generic_flash_output, mha_generic_flash_cp_output, rtol=1e-01, atol=1e-01, equal_nan=False),
        msg="Logits from generic flash and flash attention+context parallelism are not close.",
    )

    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_cp_output, rtol=1e-01, atol=1e-01, equal_nan=False),
        msg="Logits from generic dot product and flash attention+context parallelism are not close.",
    )

  @pytest.mark.tpu_only
  def test_dot_product_cache_axis_order(self):
    all_axis_orders = tuple(itertools.permutations(range(4)))
    for axis_order_tuple in random.choices(all_axis_orders, k=4):  # Renamed to avoid conflict
      axis_order_str = ",".join(map(str, axis_order_tuple))  # Convert tuple to string
      self.dot_product_attention_helper(prefill_cache_axis_order=axis_order_str, ar_cache_axis_order=axis_order_str)
      print(f"passed test for axis_order={axis_order_str}")

  def dot_product_attention_helper(self, prefill_cache_axis_order: str, ar_cache_axis_order: str):
    for compute_axis_order_tuple in [(0, 1, 2, 3), (0, 2, 1, 3)]:
      compute_axis_order_str = ",".join(map(str, compute_axis_order_tuple))
      self._dot_product_attention(
          prefill_cache_axis_order,
          ar_cache_axis_order,
          compute_axis_order=compute_axis_order_str,
      )
      print(f"passed subtest for compute_axis_order={compute_axis_order_str}")

  def _dot_product_attention(
      self,
      prefill_cache_axis_order_str: str,  # Expect string
      ar_cache_axis_order_str: str,  # Expect string
      compute_axis_order_str: str,  # Expect string
  ):
    """Test equalvant between different layout control in dot_product"""

    rtol, atol = 1e-02, 1e-02

    # Create a temporary config for this specific test scenario
    override_dict_for_dot_product = {
        "dataset_sources": {"base": {"per_device_batch_size": 1.0}},
        "run_settings": {"run_name": "test"},
        "checkpoint_save_settings": {"enable_checkpointing": False},
        "training_settings": {"max_target_length": 128, "max_prefill_predict_length": 16},
        "attention_kernel_config": {"attention": AttentionKernel.DOT_PRODUCT},
        "attention_layout_config": {  # Updated path for these from type_h.py
            "prefill_cache_axis_order": prefill_cache_axis_order_str,
            "ar_cache_axis_order": ar_cache_axis_order_str,
            "compute_axis_order": compute_axis_order_str,
        },
    }
    with open(os.path.join(PKG_DIR, "configs", "base.yml"), "rt", encoding="utf8") as f:
      base_config_dict = yaml.safe_load(f)
    base_pydantic_config = MaxTextConfig.model_validate(base_config_dict)
    override_pydantic_config = MaxTextConfig.model_validate(override_dict_for_dot_product)
    config_for_dot_prod = merge_pydantic_models(base_pydantic_config, override_pydantic_config)

    prefill_length = config_for_dot_prod.training_settings.max_prefill_predict_length
    decode_total_length = config_for_dot_prod.training_settings.max_target_length

    try:
      lnx_dtype = getattr(jnp, config_for_dot_prod.model_quantization_config.dtype)
    except AttributeError:
      lnx_dtype = jnp.float32  # fallback
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(lnx_dtype)  # Pass jnp dtype

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    attention_w_layout = Attention(
        mesh=self.mesh,  # Assuming self.mesh is initialized appropriately in setUp
        config=config_for_dot_prod,
        num_query_heads=config_for_dot_prod.model_architecture.num_query_heads,
        num_kv_heads=config_for_dot_prod.model_architecture.num_kv_heads,
        head_dim=config_for_dot_prod.model_architecture.head_dim,
        max_target_length=decode_total_length,
        max_prefill_predict_length=prefill_length,
        attention_kernel=config_for_dot_prod.attention_kernel_config.attention.value,  # string value
        dtype=lnx_dtype,  # jnp dtype
        # prefill_cache_axis_order, ar_cache_axis_order, compute_axis_order are now part of config.attention_layout_config
    )
    # Use a consistent embed_dim for init, matching what get_structured_data uses
    init_embed_dim = self.embed_dim  # From setUp, assumed to be consistent
    attention_w_layout_variable = attention_w_layout.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, decode_total_length, init_embed_dim)),
        jnp.ones((self.global_batch_size, decode_total_length, init_embed_dim)),
        jnp.ones((self.global_batch_size, decode_total_length)),
    )
    attention_w_layout_full = attention_w_layout.apply(
        attention_w_layout_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_w_layout_prefill, attention_w_layout_output_cache = attention_w_layout.apply(
        attention_w_layout_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=ModelCallMode.PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    self.assertTrue(
        jax.numpy.allclose(attention_w_layout_full[:, :prefill_length, :], attention_w_layout_prefill, equal_nan=False)
    )

    loop_vars = attention_w_layout_variable.copy()
    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      if "cache" in attention_w_layout_output_cache:
        loop_vars.update(attention_w_layout_output_cache)

      attention_w_layout_idx, attention_w_layout_output_cache = attention_w_layout.apply(
          loop_vars,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=ModelCallMode.AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
      )

      attention_w_layout_full_this_idx = attention_w_layout_full[:, idx : idx + 1, :]
      self.assertTrue(attention_w_layout_full_this_idx.shape == attention_w_layout_idx.shape)
      self.assertTrue(
          jax.numpy.allclose(attention_w_layout_full_this_idx, attention_w_layout_idx, rtol=rtol, atol=atol, equal_nan=False)
      )

  @pytest.mark.tpu_only
  def test_dot_product_reshape_q(self):
    for compute_axis_order_tuple in [(0, 1, 2, 3), (0, 2, 1, 3)]:
      compute_axis_order_str = ",".join(map(str, compute_axis_order_tuple))
      self._dot_product_attention_reshape_q(
          compute_axis_order=compute_axis_order_str,
      )
      print(f"test passed for compute_axis_order: {compute_axis_order_str}")

  def _dot_product_attention_reshape_q(self, compute_axis_order: str):
    """Test equalvant between q and reshape q in dot_product"""

    rtol, atol = 1e-02, 1e-02
    # Create a temporary config for this specific test scenario
    override_dict_for_reshape_q = {
        "dataset_sources": {"base": {"per_device_batch_size": 1.0}},
        "run_settings": {"run_name": "test"},
        "checkpoint_save_settings": {"enable_checkpointing": False},
        "training_settings": {"max_target_length": 128, "max_prefill_predict_length": 16},
        "attention_kernel_config": {"attention": AttentionKernel.DOT_PRODUCT},
        "attention_layout_config": {"compute_axis_order": compute_axis_order},  # This will be merged
    }
    with open(os.path.join(PKG_DIR, "configs", "base.yml"), "rt", encoding="utf8") as f:
      base_config_dict = yaml.safe_load(f)
    base_pydantic_config = MaxTextConfig.model_validate(base_config_dict)
    override_pydantic_config = MaxTextConfig.model_validate(override_dict_for_reshape_q)
    env_config = merge_pydantic_models(base_pydantic_config, override_pydantic_config)

    prefill_length = env_config.training_settings.max_prefill_predict_length
    decode_total_length = env_config.training_settings.max_target_length
    try:
      lnx_dtype = getattr(jnp, env_config.model_quantization_config.dtype)
    except AttributeError:
      lnx_dtype = jnp.float32  # fallback
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(lnx_dtype)

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    init_embed_dim = self.embed_dim  # From setUp, assumed to be consistent

    # Config for attention_wo_reshape_q
    cfg_wo_reshape = env_config.model_copy(deep=True)
    cfg_wo_reshape.attention_layout_config.reshape_q = False

    attention_wo_reshape_q = Attention(
        mesh=self.mesh,  # Assuming self.mesh is initialized appropriately in setUp
        config=cfg_wo_reshape,
        num_query_heads=cfg_wo_reshape.model_architecture.num_query_heads,
        num_kv_heads=cfg_wo_reshape.model_architecture.num_kv_heads,
        head_dim=cfg_wo_reshape.model_architecture.head_dim,
        max_target_length=decode_total_length,
        max_prefill_predict_length=prefill_length,
        attention_kernel=cfg_wo_reshape.attention_kernel_config.attention.value,
        dtype=lnx_dtype,
        # reshape_q is False due to cfg_wo_reshape
    )
    attention_wo_reshape_q_variable = attention_wo_reshape_q.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, decode_total_length, init_embed_dim)),
        jnp.ones((self.global_batch_size, decode_total_length, init_embed_dim)),
        jnp.ones((self.global_batch_size, decode_total_length)),
    )

    # Config for attention_w_reshape_q
    cfg_w_reshape = env_config.model_copy(deep=True)
    cfg_w_reshape.attention_layout_config.reshape_q = True

    attention_w_reshape_q = Attention(
        mesh=self.mesh,
        config=cfg_w_reshape,
        num_query_heads=cfg_w_reshape.model_architecture.num_query_heads,
        num_kv_heads=cfg_w_reshape.model_architecture.num_kv_heads,
        head_dim=cfg_w_reshape.model_architecture.head_dim,
        max_target_length=decode_total_length,
        max_prefill_predict_length=prefill_length,
        attention_kernel=cfg_w_reshape.attention_kernel_config.attention.value,
        dtype=lnx_dtype,
        # reshape_q is True due to cfg_w_reshape
    )
    attention_w_reshape_q_variable = attention_w_reshape_q.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, decode_total_length, init_embed_dim)),
        jnp.ones((self.global_batch_size, decode_total_length, init_embed_dim)),
        jnp.ones((self.global_batch_size, decode_total_length)),
    )
    # Ensure weights are the same for a fair comparison
    attention_w_reshape_q_variable = attention_w_reshape_q_variable.copy(
        {"params": attention_wo_reshape_q_variable["params"]}
    )

    attention_wo_reshape_q_full = attention_wo_reshape_q.apply(
        attention_wo_reshape_q_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,
        rngs={"aqt": self.rng},
    )
    attention_w_reshape_q_full = attention_w_reshape_q.apply(
        attention_w_reshape_q_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_wo_reshape_q_prefill, attention_wo_reshape_q_output_cache = attention_wo_reshape_q.apply(
        attention_wo_reshape_q_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=ModelCallMode.PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    self.assertTrue(
        jax.numpy.allclose(
            attention_wo_reshape_q_full[:, :prefill_length, :], attention_wo_reshape_q_prefill, equal_nan=False
        )
    )

    attention_w_reshape_q_prefill, attention_w_reshape_q_output_cache = attention_w_reshape_q.apply(
        attention_w_reshape_q_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=ModelCallMode.PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    self.assertTrue(
        jax.numpy.allclose(attention_w_reshape_q_full[:, :prefill_length, :], attention_w_reshape_q_prefill, equal_nan=False)
    )
    self.assertTrue(jax.numpy.allclose(attention_wo_reshape_q_prefill, attention_w_reshape_q_prefill, equal_nan=False))
    self.assertTrue(
        jax.numpy.allclose(
            attention_wo_reshape_q_full[:, :prefill_length, :],
            attention_w_reshape_q_full[:, :prefill_length, :],
            equal_nan=False,
        )
    )

    loop_vars_wo_reshape = attention_wo_reshape_q_variable.copy()
    loop_vars_w_reshape = attention_w_reshape_q_variable.copy()

    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]

      if "cache" in attention_wo_reshape_q_output_cache:
        loop_vars_wo_reshape.update(attention_wo_reshape_q_output_cache)
      attention_wo_reshape_q_idx, attention_wo_reshape_q_output_cache = attention_wo_reshape_q.apply(
          loop_vars_wo_reshape,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=ModelCallMode.AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
      )
      attention_wo_reshape_q_full_this_idx = attention_wo_reshape_q_full[:, idx : idx + 1, :]
      self.assertTrue(attention_wo_reshape_q_full_this_idx.shape == attention_wo_reshape_q_idx.shape)
      self.assertTrue(
          jax.numpy.allclose(
              attention_wo_reshape_q_full_this_idx, attention_wo_reshape_q_idx, rtol=rtol, atol=atol, equal_nan=False
          )
      )

      if "cache" in attention_w_reshape_q_output_cache:
        loop_vars_w_reshape.update(attention_w_reshape_q_output_cache)
      attention_w_reshape_q_idx, attention_w_reshape_q_output_cache = attention_w_reshape_q.apply(
          loop_vars_w_reshape,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=ModelCallMode.AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
      )
      attention_w_reshape_q_full_this_idx = attention_w_reshape_q_full[:, idx : idx + 1, :]
      self.assertTrue(attention_w_reshape_q_full_this_idx.shape == attention_w_reshape_q_idx.shape)
      self.assertTrue(
          jax.numpy.allclose(
              attention_w_reshape_q_full_this_idx, attention_w_reshape_q_idx, rtol=rtol, atol=atol, equal_nan=False
          )
      )
      self.assertTrue(
          jax.numpy.allclose(attention_w_reshape_q_idx, attention_wo_reshape_q_idx, rtol=rtol, atol=atol, equal_nan=False)
      )

  def test_sliding_window_attention(self):
    """Test sliding window attention"""
    try:
      dtype_jnp = getattr(jnp, self.cfg.model_quantization_config.dtype)
    except AttributeError:
      dtype_jnp = jnp.float32

    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(dtype_jnp)

    # Global Attention
    global_attn = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel=AttentionKernel.DOT_PRODUCT.value,
        dtype=self.dtype,
        dropout_rate=self.cfg.activations_logits_config.dropout_rate,
        name="global_attention",
        attention_type=AttentionType.GLOBAL,  # Pass Enum
    )

    # Attention with sliding window of size 8
    sliding_attn_cfg = self.cfg.model_copy(deep=True)
    sliding_attn_cfg.attention_kernel_config.attention_type = AttentionType.LOCAL_SLIDING
    sliding_attn_cfg.attention_kernel_config.sliding_window_size = 8

    sliding_attn = Attention(
        config=sliding_attn_cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel=AttentionKernel.DOT_PRODUCT.value,
        dtype=self.dtype,
        dropout_rate=sliding_attn_cfg.activations_logits_config.dropout_rate,
        name="sliding_window_attention",  # Different name scope for different params
        attention_type=sliding_attn_cfg.attention_kernel_config.attention_type,  # Pass Enum
        # sliding_window_size is part of the config now
    )

    attn_variable = freeze(  # Use a consistent variable set for comparison
        global_attn.init(
            {"params": self.rng, "aqt": self.rng},
            jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
            jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
            jnp.ones((self.global_batch_size, self.max_target_length)),
        )
    )
    # Make sure sliding_attn uses the same parameters if we want to compare outputs directly
    # For this test, it might be better to init sliding_attn separately if its params were different.
    # However, if only the mask changes, using the same weights is fine.
    sliding_attn_vars = sliding_attn.init(  # Init to get its own state if different
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )
    sliding_attn_vars = sliding_attn_vars.copy({"params": attn_variable["params"]})

    global_attn_output = global_attn.apply(
        attn_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,
        rngs={"aqt": self.rng},
    )

    sliding_window_output = sliding_attn.apply(
        sliding_attn_vars,
        lnx,
        lnx,
        # Use its own vars (with shared params if intended for direct output comparison)
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,
        rngs={"aqt": self.rng},
    )

    self.assertFalse(
        jax.numpy.allclose(
            sliding_window_output.astype(jnp.bfloat16), global_attn_output.astype(jnp.bfloat16), rtol=1e-04, atol=1e-04
        )
    )

    # Attention with sliding window of size max_target_length
    sliding_attn_full_cfg = self.cfg.model_copy(deep=True)
    sliding_attn_full_cfg.attention_kernel_config.attention_type = AttentionType.LOCAL_SLIDING
    sliding_attn_full_cfg.attention_kernel_config.sliding_window_size = self.max_target_length

    sliding_attn_full = Attention(
        config=sliding_attn_full_cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel=AttentionKernel.DOT_PRODUCT.value,
        dtype=self.dtype,
        dropout_rate=sliding_attn_full_cfg.activations_logits_config.dropout_rate,
        name="sliding_window_attention_full",  # Different name scope
        attention_type=sliding_attn_full_cfg.attention_kernel_config.attention_type,
        # sliding_window_size is part of config
    )
    sliding_attn_full_vars = sliding_attn_full.init(  # Init to get its own state
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )
    sliding_attn_full_vars = sliding_attn_full_vars.copy({"params": attn_variable["params"]})

    sliding_window_output_full = sliding_attn_full.apply(
        sliding_attn_full_vars,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,
        rngs={"aqt": self.rng},
    )

    self.assertTrue(
        jax.numpy.allclose(
            sliding_window_output_full.astype(jnp.bfloat16), global_attn_output.astype(jnp.bfloat16), rtol=1e-04, atol=1e-04
        )
    )


class MLATest(parameterized.TestCase):
  """Test for the Multi-Headed Latent Attention"""

  def init_mla(self, rope_type_str: str):  # rope_type_str is "yarn" or "default"
    """Helper function to initialize MLA with different model names."""
    with open(os.path.join(PKG_DIR, "configs", "base.yml"), "rt", encoding="utf8") as f:
      base_yaml_data = yaml.safe_load(f)
    base_cfg_from_yaml = MaxTextConfig.model_validate(base_yaml_data)

    selected_rope_type_enum = RoPEType.YARN if rope_type_str == "yarn" else RoPEType.DEFAULT

    # Compactly define overrides using dictionary for MaxTextConfig.model_validate
    override_dict = {
        "path_config": {"base_output_directory": ""},
        "general_run_setting": {"run_name": "test"},
        "checkpoint_save_settings": {"enable_checkpointing": False},
        "training_settings": {"max_target_length": 128, "max_prefill_predict_length": 16},
        "attention_kernel_config": {"attention_type": AttentionType.MLA.value},  # Pass string value
        "rope_config": {"rope_type": selected_rope_type_enum.value},  # Pass string value
        # For the JSON output generation, we need the exact fields as in "ExpectThis"
        # This means explicitly setting many defaults if base.yml differs from "ExpectThis"
        "model_architecture": {
            "emb_dim": 2048,
            "mlp_dim": 7168,
            "num_decoder_layers": 16,
            "num_query_heads": 16,
            "num_kv_heads": 16,
            "head_dim": 128,
            "base_moe_mlp_dim": 7168,
        },
        "model_operations": {"decoder_block": DecoderBlockType.LLAMA2.value, "normalization_layer_epsilon": 1e-05},
        "activations_logits_config": {
            "attn_logits_soft_cap": None,
            "final_logits_soft_cap": None,
            "dropout_rate": 0.0,
        },  # For null
        "mla_parameters": {  # Ensure mla_parameters is not None for the test
            "q_lora_rank": 0,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 192,  # Changed MLA params from original test
        },
        "yarn_rope_config": None,  # To ensure "yarn_rope_config": null
        "dataset_sources_config": {
            "base": {"per_device_batch_size": 1.0, "dataset_type": DatasetType.TFDS.value},
            "tfds": {"dataset_name": "c4/en:3.0.1", "eval_dataset_name": "c4/en:3.0.1", "dataset_path": ""},
        },
        "hardware_config": {"hardware": HardwareType.TPU.value, "num_slices": 1},
        "ici_parallelism_settings": {  # This is an object of IciParallelismConfig
            "ici_data_parallelism": 1,
            "ici_fsdp_parallelism": -1,  # ... and all other fields from "Expect This"
        },
        "dcn_parallelism_config": {  # This is an object of DcnParallelismConfig
            "dcn_data_parallelism": -1  # ... and all other fields from "Expect This"
        },
        # Add more specific overrides here to match "Expect This" json for MLA test
        "learning_rate_config": {"learning_rate_schedule_steps": 150001},
        "adamw_optimizer": {"mu_dtype": "float32"},
        "opt_type": OptimizerType.ADAMW.value,
        "model_quantization_config": {"dtype": "bfloat16", "quantization_local_shard_count": 1},
    }
    # Fill in ici_parallelism_settings and dcn_parallelism_config completely
    default_ici = IciParallelismConfig()
    override_dict["ici_parallelism_settings"] = {
        "ici_data_parallelism": 1,
        "ici_fsdp_parallelism": -1,
        "ici_fsdp_transpose_parallelism": default_ici.ici_fsdp_transpose_parallelism,
        "ici_sequence_parallelism": default_ici.ici_sequence_parallelism,
        "ici_context_parallelism": default_ici.ici_context_parallelism,
        "ici_context_autoregressive_parallelism": default_ici.ici_context_autoregressive_parallelism,
        "ici_tensor_parallelism": default_ici.ici_tensor_parallelism,
        "ici_tensor_transpose_parallelism": default_ici.ici_tensor_transpose_parallelism,
        "ici_tensor_sequence_parallelism": default_ici.ici_tensor_sequence_parallelism,
        "ici_autoregressive_parallelism": default_ici.ici_autoregressive_parallelism,
        "ici_pipeline_parallelism": default_ici.ici_pipeline_parallelism,
        "ici_expert_parallelism": default_ici.ici_expert_parallelism,
    }
    default_dcn = DcnParallelismConfig()
    override_dict["dcn_parallelism_config"] = {
        "dcn_data_parallelism": -1,
        "dcn_fsdp_parallelism": default_dcn.dcn_fsdp_parallelism,  # ... and so on
        "dcn_fsdp_transpose_parallelism": default_dcn.dcn_fsdp_transpose_parallelism,
        "dcn_sequence_parallelism": default_dcn.dcn_sequence_parallelism,
        "dcn_context_parallelism": default_dcn.dcn_context_parallelism,
        "dcn_context_autoregressive_parallelism": default_dcn.dcn_context_autoregressive_parallelism,
        "dcn_tensor_parallelism": default_dcn.dcn_tensor_parallelism,
        "dcn_tensor_transpose_parallelism": default_dcn.dcn_tensor_transpose_parallelism,
        "dcn_tensor_sequence_parallelism": default_dcn.dcn_tensor_sequence_parallelism,
        "dcn_pipeline_parallelism": default_dcn.dcn_pipeline_parallelism,
        "dcn_expert_parallelism": default_dcn.dcn_expert_parallelism,
        "dcn_autoregressive_parallelism": default_dcn.dcn_autoregressive_parallelism,
    }

    test_specific_overrides = MaxTextConfig.model_validate(override_dict)
    cfg = merge_pydantic_models(base_cfg_from_yaml, test_specific_overrides)
    if cfg.path_config:
      cfg.path_config._run_name_for_paths = cfg.general_run_setting.run_name

    with open(os.path.join("/tmp", f"new_mla_test_cfg_rope_{rope_type_str}.json"), "wt") as f:
      json.dump(config_to_flat_dict(cfg), f, indent=4, sort_keys=True, default=str)

    rng = jax.random.PRNGKey(0)
    devices_array = maxtext_utils.create_device_mesh_with_maxtextconfig(cfg)
    mesh = Mesh(devices_array, cfg.mesh_layout.mesh_axes)

    global_batch_size = cfg.global_batch_info.global_batch_size_to_train_on
    num_kv_heads = cfg.model_architecture.num_kv_heads
    num_query_heads = cfg.model_architecture.num_query_heads
    max_target_length = cfg.training_config.max_target_length
    max_prefill_predict_length = cfg.training_config.max_prefill_predict_length
    head_dim = cfg.model_architecture.head_dim
    embed_dim = cfg.model_architecture.emb_dim

    dtype_str = cfg.model_quantization_config.dtype
    try:
      selected_dtype = getattr(jnp, dtype_str)
    except AttributeError:
      selected_dtype = jnp.float32  # Fallback

    attention_type_enum_val = cfg.attention_kernel_config.attention_type

    mla_instance = MLA(
        config=cfg,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        max_prefill_predict_length=max_prefill_predict_length,
        mesh=mesh,
        attention_kernel=cfg.attention_kernel_config.attention.value,  # Pass string from enum
        dtype=selected_dtype,
        dropout_rate=cfg.activations_logits_config.dropout_rate,
        name="self_attention",
        attention_type=attention_type_enum_val,  # Pass enum member
        # Values from original test, not "Expect This" directly as those were mostly defaults
        q_lora_rank=cfg.mla_parameters.q_lora_rank if cfg.mla_parameters else 10,
        kv_lora_rank=cfg.mla_parameters.kv_lora_rank if cfg.mla_parameters else 20,
        qk_nope_head_dim=cfg.mla_parameters.qk_nope_head_dim if cfg.mla_parameters else 128,
        qk_rope_head_dim=cfg.mla_parameters.qk_rope_head_dim if cfg.mla_parameters else 64,
        v_head_dim=cfg.mla_parameters.v_head_dim if cfg.mla_parameters else 192,
    )

    mla_variable = mla_instance.init(
        {"params": rng, "aqt": rng},
        jnp.ones((global_batch_size, max_target_length, embed_dim)),
        jnp.ones((global_batch_size, max_target_length, embed_dim)),
        jnp.ones((global_batch_size, max_target_length)),
    )
    return cfg, mla_instance, mla_variable, rng

  # get_data and get_structured_data methods in MLATest need to be updated to use
  # the new config attribute paths, e.g., cfg.global_batch_sizes.global_batch_size_to_train_on,
  # cfg.training_process.max_target_length, cfg.model_structure.emb_dim

  def get_data(self, cfg: MaxTextConfig, rng_key: jax.random.PRNGKey, dtype: Any) -> Any:  # Added type hints
    """get data"""
    batch_size = cfg.global_batch_info.global_batch_size_to_train_on  # Updated path
    max_len = cfg.training_config.max_target_length  # Updated path
    emb_d = cfg.model_architecture.emb_dim  # Updated path

    lnx = jax.random.normal(rng_key, shape=(batch_size, max_len, emb_d), dtype=dtype)
    decoder_segment_ids = jax.random.randint(rng_key, (batch_size, max_len), 0, 4)
    decoder_positions = jax.random.randint(rng_key, (batch_size, max_len), 0, max_len)
    return lnx, decoder_segment_ids, decoder_positions

  def get_structured_data(self, cfg: MaxTextConfig, rng_key: jax.random.PRNGKey, dtype: Any) -> Any:  # Added type hints
    """get structured data"""
    batch_size = cfg.global_batch_info.global_batch_size_to_train_on  # Updated path
    max_len = cfg.training_config.max_target_length  # Updated path
    emb_d = cfg.model_architecture.emb_dim  # Updated path

    lnx = jax.random.normal(rng_key, shape=(batch_size, max_len, emb_d), dtype=dtype)
    decoder_positions = jnp.stack([jnp.arange(max_len, dtype=jnp.int32) for _ in range(batch_size)])
    decoder_segment_ids = jax.numpy.zeros((batch_size, max_len)) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    return lnx, decoder_segment_ids, decoder_positions

  @parameterized.named_parameters(
      {"testcase_name": "RoPE_Yarn_Autoregression", "rope_type_str": "yarn"},  # Use rope_type_str
      {"testcase_name": "Default_Autoregression", "rope_type_str": "default"},  # Use rope_type_str
  )
  @pytest.mark.tpu_only
  def test_autoregression(self, rope_type_str: str):  # Changed parameter name
    cfg, mla, mla_variable, rng = self.init_mla(rope_type_str)  # Pass string

    try:
      data_dtype = getattr(jnp, cfg.model_quantization_config.dtype)  # Get string from config
    except AttributeError:
      data_dtype = jnp.float32  # Fallback

    prefill_length = cfg.training_config.max_prefill_predict_length  # Updated path
    decode_total_length = cfg.training_config.max_target_length  # Updated path
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(cfg, rng, data_dtype)

    mla_full = mla.apply(
        mla_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=ModelCallMode.TRAIN,  # Use Enum
        rngs={"aqt": rng},
    )

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    mla_prefill, output_cache = mla.apply(
        mla_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=ModelCallMode.PREFILL,  # Use Enum
        rngs={"aqt": rng},
        mutable=["cache"],
    )

    self.assertTrue(
        jax.numpy.allclose(mla_prefill, mla_full[:, :prefill_length, :], rtol=1e-02, atol=1e-02, equal_nan=False)
    )

    # Create a mutable copy for the loop
    mla_variable_loop = mla_variable.copy()
    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      if "cache" in output_cache:  # Check if cache needs to be updated
        mla_variable_loop.update(output_cache)

      mla_idx, output_cache = mla.apply(
          mla_variable_loop,  # Use the loop-local, updated variables
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=ModelCallMode.AUTOREGRESSIVE,  # Use Enum
          rngs={"aqt": rng},
          mutable=["cache"],
      )

      mla_full_this_idx = mla_full[:, idx : idx + 1, :]
      self.assertEqual(mla_full_this_idx.shape, mla_idx.shape)
      # TODO (b/394626702) uncomment last check when decode and kv_cache are implemented for MLA
      # self.assertTrue(jax.numpy.allclose(mla_full_this_idx, mla_idx, rtol=1e-02, atol=1e-02, equal_nan=False))


if __name__ == "__main__":
  unittest.main()
