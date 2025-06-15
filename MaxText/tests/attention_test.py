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

import os
import sys
import unittest
from unittest.mock import patch

import pytest

from absl.testing import parameterized

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from MaxText import maxtext_utils, pyconfig
from MaxText.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, \
    DECODING_ACTIVE_SEQUENCE_INDICATOR
from MaxText.configs.types_g import MaxTextConfig, BaseDatasetConfig, DatasetNestingConfig, BasicTrainingConfig, DatasetType, ModelArchitectureConfig, QuantizationConfig, CheckpointConfig, CheckpointSavingConfig, AttentionMechanismConfig, RunConfig, AttentionKernel
from MaxText.globals import PKG_DIR
from MaxText.layers import attentions
from MaxText.layers.attentions import Attention, MLA, ChunkedCausalMask


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
  """Test for the attention layer."""

  def setUp(self):
    super().setUp()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    self.config = MaxTextConfig(
        dataset_nesting_config=DatasetNestingConfig(
            base=BaseDatasetConfig(
                per_device_batch_size=1,
                dataset_type=DatasetType.SYNTHETIC,
            )
        ),
        basic_training_config=BasicTrainingConfig(
            max_target_length=4096,
        ),
        model_architecture_config=ModelArchitectureConfig(
            base_num_query_heads=8,
            base_num_kv_heads=8,
            head_dim=256,
            base_emb_dim=2048,
            base_mlp_dim=8192,
            base_num_decoder_layers=2,
        ),
        quantization_config=QuantizationConfig(
            dtype="bfloat16",
        ),
        checkpoint_config=CheckpointConfig(
            saving=CheckpointSavingConfig(
                enable_checkpointing=False,
            ),
        ),
        attention_mechanism_config=AttentionMechanismConfig(
            fused_qkv=False,
            attention=AttentionKernel.DOT_PRODUCT,
        ),
        run_config=RunConfig(run_name="test", base_output_directory="attention_test_dir"),
    )
    self.batch_size = self.config.dataset_nesting_config.base.per_device_batch_size
    self.num_query_heads = self.config.model_architecture_config.base_num_query_heads
    self.num_kv_heads = self.config.model_architecture_config.base_num_kv_heads
    self.max_target_length = self.config.basic_training_config.max_target_length
    self.head_dim = self.config.model_architecture_config.head_dim
    self.embed_dim = self.config.model_architecture_config.base_emb_dim

    devices = jax.devices()
    self.mesh = Mesh(devices, ("data", "model"))

    self.query = jnp.ones(
        (self.batch_size, self.max_target_length, self.num_query_heads, self.head_dim),
        dtype=self.config.quantization_config.dtype,
    )
    self.key = jnp.ones(
        (self.batch_size, self.max_target_length, self.num_kv_heads, self.head_dim),
        dtype=self.config.quantization_config.dtype,
    )
    self.value = jnp.ones(
        (self.batch_size, self.max_target_length, self.num_kv_heads, self.head_dim),
        dtype=self.config.quantization_config.dtype,
    )
    self.bias = jnp.ones(
        (self.batch_size, self.num_query_heads, self.max_target_length, self.max_target_length),
        dtype=self.config.quantization_config.dtype,
    )

  @parameterized.parameters("dot_product", "flash", "cudnn_flash_te", "splash")
  @patch("jax.devices")
  def test_attention_variants(self, attention_str, devices_mock):
    """Test Attention variants."""
    # devices_mock.return_value = test_utils.create_devices_array()
    self.config.attention = attention_str
    if attention_str == "splash" and not attentions.is_splash_attention_supported():
      self.skipTest("splash attention is not supported")
    if attention_str == "cudnn_flash_te":
      if not attentions.is_cudnn_flash_attention_supported():
        self.skipTest("could not run test for cudnn flash attention")

    attention_layer = Attention(self.config, self.mesh, name="self_attention")
    output, _ = attention_layer._apply_attention(self.query, self.key, self.value, self.bias)

    self.assertEqual(output.shape, self.query.shape)
    self.assertEqual(output.dtype, jnp.dtype(self.config.quantization_config.dtype))

  def test_unfused_attention(self):
    """Test unfused Attention."""
    attention_layer = Attention(self.config, self.mesh, name="self_attention")
    rng = jax.random.PRNGKey(0)
    query_in = jnp.ones(
        (
            self.config.dataset_nesting_config.base.per_device_batch_size,
            self.config.basic_training_config.max_target_length,
            self.config.model_architecture_config.base_emb_dim,
        ),
        dtype=self.config.quantization_config.dtype,
    )
    params = attention_layer.init(rng, query_in)["params"]
    output = attention_layer.apply(
        {"params": params},
        query_in,
        rngs={"dropout": rng},
    )[0]
    expected_shape = (
        self.config.dataset_nesting_config.base.per_device_batch_size,
        self.config.basic_training_config.max_target_length,
        self.config.model_architecture_config.base_emb_dim,
    )

    self.assertEqual(output.shape, expected_shape)
    self.assertEqual(output.dtype, jnp.dtype(self.config.quantization_config.dtype))

  def test_fused_attention(self):
    """Test fused Attention."""
    self.config.attention_mechanism_config.fused_qkv = True
    attention_layer = Attention(self.config, self.mesh, name="self_attention")
    rng = jax.random.PRNGKey(0)
    query_in = jnp.ones(
        (
            self.config.dataset_nesting_config.base.per_device_batch_size,
            self.config.basic_training_config.max_target_length,
            self.config.model_architecture_config.base_emb_dim,
        ),
        dtype=self.config.quantization_config.dtype,
    )
    params = attention_layer.init(rng, query_in)["params"]
    output = attention_layer.apply(
        {"params": params},
        query_in,
        rngs={"dropout": rng},
    )[0]
    expected_shape = (
        self.config.dataset_nesting_config.base.per_device_batch_size,
        self.config.basic_training_config.max_target_length,
        self.config.model_architecture_config.base_emb_dim,
    )

    self.assertEqual(output.shape, expected_shape)
    self.assertEqual(output.dtype, jnp.dtype(self.config.quantization_config.dtype))

  def test_quantized_kv_cache_attention(self):
    self.config.quantization_config.quantize_kvcache = True
    attention_layer = Attention(self.config, self.mesh, name="self_attention")
    output, _ = attention_layer._apply_attention(self.query, self.key, self.value, self.bias)
    self.assertEqual(output.dtype, jnp.bfloat16)

class MLATest(parameterized.TestCase):
  """Test for the Multi-Headed Latent Attention"""

  def init_mla(self, rope_type):
    """Helper function to initialize MLA with different model names."""
    cfg = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=128,
        max_prefill_predict_length=16,
        attention_type=attentions.AttentionType.MLA.value,
        rope_type=rope_type,
    )
    rng = jax.random.PRNGKey(0)

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)

    global_batch_size = cfg.global_batch_size_to_train_on
    num_kv_heads = cfg.num_kv_heads
    num_query_heads = cfg.num_query_heads
    max_target_length = cfg.max_target_length
    max_prefill_predict_length = cfg.max_prefill_predict_length
    head_dim = cfg.head_dim
    embed_dim = cfg.base_emb_dim
    dtype = cfg.dtype
    attention_type = cfg.attention_type

    mla = MLA(
        config=cfg,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        max_prefill_predict_length=max_prefill_predict_length,
        mesh=mesh,
        attention_kernel="dot_product",
        dtype=dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        attention_type=attention_type,
        q_lora_rank=10,
        kv_lora_rank=20,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=192,
    )

    mla_variable = mla.init(
        {"params": rng, "aqt": rng},
        jnp.ones((global_batch_size, max_target_length, embed_dim)),
        jnp.ones((global_batch_size, max_target_length, embed_dim)),
        jnp.ones((global_batch_size, max_target_length)),
    )

    return cfg, mla, mla_variable, rng

  def get_data(self, cfg, rng, dtype):
    """get data"""
    lnx = jax.random.normal(
        rng,
        shape=(cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.base_emb_dim),
        dtype=dtype,
    )

    decoder_segment_ids = jax.random.randint(rng, (cfg.global_batch_size_to_train_on, cfg.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(
        rng, (cfg.global_batch_size_to_train_on, cfg.max_target_length), 0, cfg.max_target_length
    )

    return lnx, decoder_segment_ids, decoder_positions

  def get_structured_data(self, cfg, rng, dtype):
    """get structured data"""
    lnx = jax.random.normal(
        rng,
        shape=(
            cfg.global_batch_size_to_train_on,
            cfg.max_target_length,
            cfg.base_emb_dim,
        ),
        dtype=dtype,
    )

    decoder_positions = jnp.stack(
        [jnp.arange(cfg.max_target_length, dtype=jnp.int32) for _ in range(cfg.global_batch_size_to_train_on)]
    )

    decoder_segment_ids = (
        jax.numpy.zeros((cfg.global_batch_size_to_train_on, cfg.max_target_length)) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    )

    return lnx, decoder_segment_ids, decoder_positions

  @parameterized.named_parameters(
      {"testcase_name": "RoPE_Yarn_Autoregression", "rope_type": "yarn"},
      {"testcase_name": "Default_Autoregression", "rope_type": "default"},
  )
  @pytest.mark.tpu_only
  def test_autoregression(self, rope_type):
    cfg, mla, mla_variable, rng = self.init_mla(rope_type)
    prefill_length = cfg.max_prefill_predict_length
    decode_total_length = cfg.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(cfg, rng, cfg.dtype)

    mla_full = mla.apply(
        mla_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
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
        model_mode=MODEL_MODE_PREFILL,
        rngs={"aqt": rng},
        mutable=["cache"],
    )

    self.assertTrue(
        jax.numpy.allclose(mla_prefill, mla_full[:, :prefill_length, :], rtol=1e-02, atol=1e-02, equal_nan=False)
    )

    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      mla_variable.update(output_cache)
      mla_idx, output_cache = mla.apply(
          mla_variable,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          rngs={"aqt": rng},
          mutable=["cache"],
      )

      mla_full_this_idx = mla_full[:, idx : idx + 1, :]
      self.assertEqual(mla_full_this_idx.shape, mla_idx.shape)
      # TODO (b/394626702) uncomment last check when decode and kv_cache are implemented for MLA
      # self.assertTrue(jax.numpy.allclose(mla_full_this_idx, mla_idx, rtol=1e-02, atol=1e-02, equal_nan=False))


if __name__ == "__main__":
  unittest.main()
