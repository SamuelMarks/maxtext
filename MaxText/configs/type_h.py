# MaxText/configs/type_h.py
"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This module defines the Pydantic models for MaxText configuration.
Models are structured for readability, strong typing, and to facilitate
flattening into the desired legacy YAML/JSON format.
"""

from enum import Enum
from tempfile import gettempdir
from typing import List, Optional, Any, Literal
import os

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeInt,
    NonNegativeFloat,
    computed_field,
    ConfigDict,
    model_validator,
)


# -----------------------------------------------------------------------------
# Enumerations
# -----------------------------------------------------------------------------


class DecoderBlockType(str, Enum):
  """Specifies the type of decoder block to use in the model architecture."""

  DEFAULT = "default"
  LLAMA2 = "llama2"
  MISTRAL = "mistral"
  MIXTRAL = "mixtral"
  DEEPSEEK = "deepseek"
  GEMMA = "gemma"
  GEMMA2 = "gemma2"
  GEMMA3 = "gemma3"
  GPT3 = "gpt3"
  SIMPLE = "simple"
  SIMPLE_MLP = "simple_mlp"
  LLAMA4 = "llama4"


class AttentionType(str, Enum):
  """Defines variants of attention mechanisms."""

  GLOBAL = "global"
  LOCAL_SLIDING = "local_sliding"
  CHUNK = "chunk"
  MLA = "mla"
  FULL = "full"


class OptimizerType(str, Enum):
  """Specifies the optimizer algorithm to use for training."""

  ADAMW = "adamw"
  ADAM_PAX = "adam_pax"
  SGD = "sgd"


class MatMulPrecision(str, Enum):
  """Precision level for matrix multiplication operations in JAX."""

  DEFAULT = "default"
  HIGH = "high"
  HIGHEST = "highest"


class DatasetType(str, Enum):
  """Type of dataset pipeline to use for data loading."""

  SYNTHETIC = "synthetic"
  HF = "hf"
  GRAIN = "grain"
  TFDS = "tfds"
  C4_MLPERF = "c4_mlperf"


class GrainFileType(str, Enum):
  """File format for datasets loaded via the Grain library."""

  ARRAYRECORD = "arrayrecord"
  PARQUET = "parquet"


class HardwareType(str, Enum):
  """Target hardware platform for the training or inference job."""

  TPU = "tpu"
  GPU = "gpu"
  GPU_MULTIPROCESS = "gpu_multiprocess"
  CPU = "cpu"


class ProfilerType(str, Enum):
  """Type of performance profiler to use."""

  NONE = ""
  XPLANE = "xplane"
  NSYS = "nsys"


class AttentionKernel(str, Enum):
  """Specific algorithm/kernel used for computing attention scores."""

  AUTOSELECTED = "autoselected"
  DOT_PRODUCT = "dot_product"
  FLASH = "flash"
  CUDNN_FLASH_TE = "cudnn_flash_te"
  CUDNN_FLASH_JAX = "cudnn_flash_jax"
  PAGED = "paged"


class RematPolicy(str, Enum):
  """Rematerialization (gradient checkpointing) policy for model layers."""

  MINIMAL = "minimal"
  SAVE_DOT_WITH_CONTEXT_EXCEPT_MLP = "save_dot_with_context_except_mlp"
  SAVE_DOT_EXCEPT_MLPWI = "save_dot_except_mlpwi"
  SAVE_DOT_EXCEPT_MLP = "save_dot_except_mlp"
  SAVE_QKV_PROJ = "save_qkv_proj"
  QKV_PROJ_OFFLOADED = "qkv_proj_offloaded"
  CUSTOM = "custom"
  MINIMAL_OFFLOADED = "minimal_offloaded"
  SAVE_OUT_PROJ = "save_out_proj"
  FULL = "full"
  MINIMAL_FLASH = "minimal_flash"


class RematTensorConfigValue(str, Enum):
  """Configuration value for individual tensors in custom rematerialization policy."""

  REMAT = "remat"
  DEVICE = "device"
  OFFLOAD = "offload"


class ModelCallMode(str, Enum):
  """Mode in which the model's __call__ method is executed."""

  TRAIN = ""
  INFERENCE = "inference"
  AUTOREGRESSIVE = "autoregressive"
  PREFILL = "prefill"


class SamplingStrategy(str, Enum):
  """Strategy for sampling tokens during text generation."""

  GREEDY = "greedy"
  WEIGHTED = "weighted"
  NUCLEUS = "nucleus"
  TOPK = "topk"


class RoPEType(str, Enum):
  """Type of Rotary Position Embedding (RoPE) implementation."""

  DEFAULT = "default"
  LLAMA3_1 = "llama3.1"
  YARN = "yarn"


class TokenizerTypeEnum(str, Enum):
  """Specifies the tokenizer library or type to use."""

  SENTENCEPIECE = "sentencepiece"
  TIKTOKEN = "tiktoken"
  HUGGINGFACE = "huggingface"


class InferenceServerType(str, Enum):
  """Type of inference server to launch for serving the model."""

  MAXTEXT_INTERLEAVED = "MaxtextInterleavedServer"
  EXPERIMENTAL_MAXTEXT_DISAGGREGATED = "ExperimentalMaxtextDisaggregatedServer"


# -----------------------------------------------------------------------------
# Granular Configuration Models
# -----------------------------------------------------------------------------


class PathConfig(BaseModel):
  """Configuration for various important file system paths."""

  base_output_directory: str = Field(
      default="", description="Base directory for experiment outputs. Set to empty for test target matching."
  )
  metrics_file: Optional[str] = Field(
      default="", description="Path to local file for scalar metrics. Empty disables local logging."
  )
  tokenizer_path: str = Field(
      default=os.path.join("assets", "tokenizer.llama2"), description="Path to the tokenizer model file or assets directory."
  )
  prefill_cache_dir: Optional[str] = Field(default="", description="Directory for prefill KV cache read/write.")
  compiled_trainstep_file: Optional[str] = Field(default="", description="Path to AOT compiled train_step pickle file.")
  quant_cfg_path: Optional[str] = Field(default="", description="Path to JSON config for 'intmp' quantization.")

  run_name_for_paths_internal_only: Optional[str] = Field(default=None, exclude=True)  # Internal helper

  @computed_field(alias="checkpoint_dir")
  @property
  def computed_checkpoint_dir(self) -> str:
    run_name = self.run_name_for_paths_internal_only or "test"
    if self.base_output_directory == "" and run_name == "test":
      return os.path.join("test", "checkpoints", "")
    elif self.base_output_directory and run_name:
      return os.path.join(self.base_output_directory, run_name, "checkpoints", "")
    else:
      return os.path.join("default_checkpoint_dir", "")

  @computed_field(alias="metrics_dir")
  @property
  def computed_metrics_dir(self) -> str:
    run_name = self.run_name_for_paths_internal_only or "test"
    if self.base_output_directory == "" and run_name == "test":
      return os.path.join("test", "metrics", "")
    elif self.base_output_directory and run_name:
      return os.path.join(self.base_output_directory, run_name, "metrics", "")
    else:
      return os.path.join("default_metrics_dir", "")

  @computed_field(alias="tensorboard_dir")
  @property
  def computed_tensorboard_dir(self) -> str:
    run_name = self.run_name_for_paths_internal_only or "test"
    if self.base_output_directory == "" and run_name == "test":
      return os.path.join("test", "tensorboard", "")
    elif self.base_output_directory and run_name:
      return os.path.join(self.base_output_directory, run_name, "tensorboard", "")
    else:
      return os.path.join("default_tensorboard_dir", "")

  model_config = ConfigDict(populate_by_name=True)


class GeneralRunSetting(BaseModel):
  """General settings for the execution of a training or evaluation run."""

  run_name: str = Field(default="test", description="User-defined name for the run. Target: 'test'.")
  log_period: PositiveInt = Field(default=100, description="Frequency (steps) for TensorBoard/metrics logging.")
  steps: int = Field(default=150_001, description="Total training steps. -1 uses learning_rate_schedule_steps.")
  log_config: bool = Field(default=True, description="Print the final configuration at startup.")
  enable_tensorboard: bool = Field(default=True, description="Enable TensorBoard logging.")
  gcs_metrics: bool = Field(default=False, description="Save scalar metrics (loss, TFLOPS) to GCS.")
  save_config_to_gcs: bool = Field(default=False, description="Save final config file to GCS.")
  max_checkify: bool = Field(default=False, description="Enable jax.checkify for debugging; affects performance.")
  rep: NonNegativeInt = Field(default=0, description="For TPU perf testing, repeats execution of the same batch N times.")


class VertexAiSetting(BaseModel):
  """Settings specific to using Google Cloud Vertex AI for TensorBoard."""

  use_vertex_tensorboard: bool = Field(default=False, description="Use Vertex AI TensorBoard for logging.")
  vertex_tensorboard_project: Optional[str] = Field(default="", description="GCP project ID for Vertex AI TensorBoard.")
  vertex_tensorboard_region: Optional[str] = Field(default="", description="GCP region for Vertex AI TensorBoard.")


class CheckpointLoadSetting(BaseModel):
  """Configuration for loading model weights or full training state from checkpoints."""

  load_parameters_path: Optional[str] = Field(default="", description="Path to load parameters-only checkpoint from.")
  lora_input_adapters_path: Optional[str] = Field(default="", description="GCS path to parent directory of LoRA adapters.")
  load_full_state_path: Optional[str] = Field(default="", description="Path to load full training state from.")
  checkpoint_is_quantized: bool = Field(default=False, description="Indicates if loading a quantized (AQT) checkpoint.")


class CheckpointSaveSetting(BaseModel):
  """Configuration for saving model checkpoints during training."""

  enable_checkpointing: bool = Field(default=False, description="Enable checkpoint saving. Target: False.")
  async_checkpointing: bool = Field(default=True, description="Use asynchronous checkpointing if enabled.")
  checkpoint_period: NonNegativeInt = Field(default=10_000, description="Frequency (steps) for saving checkpoints.")
  force_unroll: bool = Field(default=False, description="Force unroll loop for param-only checkpoint generation.")
  save_quantized_params_path: Optional[str] = Field(
      default="", description="Path to save on-the-fly quantized model params (AQT)."
  )


class CheckpointStoreSetting(BaseModel):
  """Configuration for Orbax checkpoint storage backend."""

  target_data_file_size_bytes: int = Field(default=2_147_483_648, alias="checkpoint_storage_target_data_file_size_bytes")
  use_ocdbt: bool = Field(default=True, alias="checkpoint_storage_use_ocdbt")
  use_zarr3: bool = Field(default=True, alias="checkpoint_storage_use_zarr3")
  concurrent_gb: int = Field(default=96, alias="checkpoint_storage_concurrent_gb")
  model_config = ConfigDict(populate_by_name=True)


class EmergencyCkptSetting(BaseModel):
  """Configuration for Orbax emergency (local disk) checkpointing feature."""

  enable_emergency_checkpoint: bool = Field(default=False)
  local_checkpoint_directory: Optional[str] = Field(default="")
  local_checkpoint_period: NonNegativeInt = Field(default=0)
  use_replicator_service: bool = Field(default=False)
  replicator_backup_interval_minutes: NonNegativeInt = Field(default=0)


class CheckpointMiscSetting(BaseModel):
  """Miscellaneous settings related to checkpoint restoration and logging."""

  enable_single_replica_ckpt_restoring: bool = Field(default=False)
  enable_checkpoint_cloud_logger: bool = Field(default=False)


class ModelIdSetting(BaseModel):
  """Configuration for identifying the model type and allowing overrides."""

  model_name: str = Field(default="default", description="Identifier for model architecture (e.g., 'llama2-7b').")
  override_model_config: bool = Field(default=False, description="Allow CLI to override model parameters.")


class ModelArchitecture(BaseModel):
  """Core architectural parameters defining the model's size and structure."""

  decoder_block: DecoderBlockType = Field(default=DecoderBlockType.LLAMA2, description="Type of decoder block.")
  emb_dim: PositiveInt = Field(default=2048, description="Core embedding dimension.")
  mlp_dim: PositiveInt = Field(default=7168, description="Intermediate dimension of MLP layers.")
  num_decoder_layers: PositiveInt = Field(default=16, description="Total number of decoder layers.")
  num_query_heads: PositiveInt = Field(default=16, description="Number of attention heads for queries.")
  num_kv_heads: PositiveInt = Field(default=16, description="Number of heads for keys/values (for GQA/MQA).")
  head_dim: Optional[PositiveInt] = Field(default=128, description="Dimension of each attention head.")
  global_parameter_scale: int = Field(default=1, description="Global scaling factor for model parameters.")
  base_moe_mlp_dim: Optional[PositiveInt] = Field(default=7168, description="MLP hidden dim for MoE layers. Target: 7168.")


class ModelOperationConfig(BaseModel):
  """Configuration for model dtypes, norms, execution modes, and positional embeddings."""

  weight_dtype: str = Field(default="float32", description="Data type for model weights.")
  normalization_layer_epsilon: float = Field(default=1.0e-5, description="Epsilon for normalization layers.")
  model_call_mode: ModelCallMode = Field(default=ModelCallMode.TRAIN, description="Execution mode for model call.")
  param_scan_axis: int = Field(default=1, description="Axis for parameter scanning if `scan_layers` is true.")
  inhomogeneous_layer_cycle_interval: int = Field(default=1, description="Cycle interval for inhomogeneous layers.")


class ModelPositionalEmbedding(BaseModel):  # Correctly defined
  """Configuration for positional embedding strategies."""

  use_iota_embed: bool = Field(default=False, description="Use iota-based embedding for positional info.")
  use_untrainable_positional_embedding: bool = Field(
      default=False, description="Use fixed, untrainable positional embeddings."
  )
  trainable_position_size: int = Field(default=-1, description="Size of GPT3-style trainable pos-embeddings if > 0.")


class ActivationLogitConfig(BaseModel):
  """Configuration for MLP activations, dropout, and logits processing."""

  mlp_activations: List[str] = Field(default_factory=lambda: ["silu", "linear"])
  dropout_rate: NonNegativeFloat = Field(default=0.0)
  logits_via_embedding: bool = Field(default=False)
  normalize_embedding_logits: bool = Field(default=True)
  logits_dot_in_fp32: bool = Field(default=False)
  cast_logits_to_fp32: bool = Field(default=True)
  float32_qk_product: bool = Field(default=False)
  float32_logits: bool = Field(default=False)  # Name for target
  activations_in_float32: bool = Field(default=False)


class ModelQuantizeConfig(BaseModel):
  """Configurations for model quantization techniques."""

  dtype: str = Field(default="bfloat16", description="Primary data type for activations.")
  quantization: Optional[str] = Field(default="", description="Quantization type (e.g., 'int8'). Empty for none.")
  matmul_precision: MatMulPrecision = Field(default=MatMulPrecision.DEFAULT)
  replicate_quant_scale: bool = Field(default=False)
  quantize_kvcache: bool = Field(default=False)
  kv_quant_axis: str = Field(default="heads_and_dkv")
  kv_quant_dtype: str = Field(default="int8")
  quantization_local_shard_count: int = Field(default=1)  # Target


class MoEGeneral(BaseModel):
  """General Mixture of Experts (MoE) configurations."""

  num_experts: PositiveInt = Field(default=1)
  num_experts_per_tok: PositiveInt = Field(default=1)
  megablox: bool = Field(default=True)
  sparse_matmul: bool = Field(default=True)
  capacity_factor: float = Field(default=-1.0)
  load_balance_loss_weight: NonNegativeFloat = Field(default=0.01)
  use_random_routing: bool = Field(default=False)
  moe_mlp_dim: Optional[PositiveInt] = Field(default=7168, description="Intermediate MLP dim for MoE. Target 7168.")


class MoETiling(BaseModel):
  """Tunable tiling dimensions for Megablox MoE kernels."""

  tile_batch_seq: Optional[PositiveInt] = Field(default=512)
  tile_activation_dim: Optional[PositiveInt] = Field(default=1024)
  tile_weight_dim: Optional[PositiveInt] = Field(default=1024)


class DeepSeekMoEOverrides(BaseModel):  # Correctly defined
  """Specific architectural overrides for DeepSeek-style MoE models."""

  first_num_dense_layers: NonNegativeInt = Field(default=0)
  shared_experts: PositiveInt = Field(default=1)
  routed_scaling_factor: float = Field(default=1.0)
  routed_score_func: Optional[str] = Field(default="")
  routed_bias: bool = Field(default=False)
  n_routing_groups: int = Field(default=-1)
  topk_routing_group: int = Field(default=-1)


class PipelineConfig(BaseModel):
  """Configurations for enabling and tuning pipeline parallelism."""

  num_layers_per_pipeline_stage: PositiveInt = Field(default=1)
  num_pipeline_repeats: int = Field(default=-1)
  pipeline_parallel_layers: int = Field(default=-1)
  num_pipeline_microbatches: int = Field(default=-1)
  pipeline_delay_activation_forwarding: bool = Field(default=False)
  pipeline_fsdp_ag_once: bool = Field(default=False)
  scan_pipeline_iterations: bool = Field(default=True)
  scan_layers_per_stage: bool = Field(default=False)
  set_remat_policy_on_pipeline_iterations: bool = Field(default=True)
  set_remat_policy_on_layers_per_stage: bool = Field(default=False)
  using_pipeline_parallelism: bool = Field(default=False, description="Global flag if pipeline parallelism is active.")


class RematPolicyConfig(BaseModel):
  """Rematerialization (gradient checkpointing) policy settings."""

  remat_policy: RematPolicy = Field(default=RematPolicy.FULL)
  decoder_layer_input: RematTensorConfigValue = Field(default=RematTensorConfigValue.DEVICE)
  context: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  mlpwi: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  mlpwi_0: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  mlpwi_1: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  mlpwo: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  query_proj: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  key_proj: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  value_proj: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  qkv_proj: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  out_proj: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)


class AttentionKernelSetting(BaseModel):
  """Configuration for attention mechanism type and specific variants."""

  attention: AttentionKernel = Field(default=AttentionKernel.AUTOSELECTED)
  attention_type: AttentionType = Field(default=AttentionType.MLA)  # Target for test
  sliding_window_size: NonNegativeInt = Field(default=0)
  chunk_attn_window_size: NonNegativeInt = Field(default=0)
  mla_naive_kvcache: bool = Field(default=True)


class AttentionOpFusionSetting(BaseModel):
  """Configuration for fusing attention-related matrix multiplications."""

  fused_qkv: bool = Field(default=False)
  fused_mlp: bool = Field(default=False)


class AttentionExtraBehavior(BaseModel):
  """Configuration for attention logits capping, norms, and other specific behaviors."""

  attn_logits_soft_cap: Optional[NonNegativeFloat] = Field(default=None)  # Target null
  final_logits_soft_cap: Optional[NonNegativeFloat] = Field(default=None)  # Target null
  use_post_attn_norm: bool = Field(default=False)
  use_post_ffw_norm: bool = Field(default=False)
  stack_prefill_result_cache: bool = Field(default=False)
  enable_padding_causal_mask: bool = Field(default=True)
  use_ragged_attention: bool = Field(default=False)
  ragged_block_size: PositiveInt = Field(default=256)


class MlaParams(BaseModel):
  """Multi-Head Latent Attention (MLA) architectural parameters."""

  q_lora_rank: NonNegativeInt = Field(default=0)
  kv_lora_rank: NonNegativeInt = Field(default=512)
  qk_nope_head_dim: PositiveInt = Field(default=128)
  qk_rope_head_dim: PositiveInt = Field(default=64)
  v_head_dim: PositiveInt = Field(default=128)


class HardwarePlatform(BaseModel):
  """Hardware platform and JAX distributed system configurations."""

  hardware: HardwareType = Field(default=HardwareType.TPU)  # Target
  num_slices: int = Field(default=1, description="Number of TPU slices. Target '1'.")
  jax_cache_dir: str = Field(default=os.path.join(os.path.expanduser("~"), "jax_cache"))
  jax_distributed_initialization_timeout: PositiveInt = Field(default=300)
  jax_debug_log_modules: Optional[str] = Field(default="")
  skip_jax_distributed_system: bool = Field(default=False)
  enable_single_controller: bool = Field(default=False)


class AotCompileConfig(BaseModel):
  """Ahead-of-Time (AOT) compilation settings."""

  compiled_trainstep_file: Optional[str] = Field(default="")
  compile_topology: Optional[str] = Field(default="")
  compile_topology_num_slices: int = Field(default=-1)


class MeshConfig(BaseModel):
  """Configurations for device mesh layout and tensor sharding rules."""

  mesh_axes: List[str] = Field(
      default_factory=lambda: [
          "data",
          "stage",
          "fsdp",
          "fsdp_transpose",
          "sequence",
          "context",
          "context_autoregressive",
          "tensor",
          "tensor_transpose",
          "tensor_sequence",
          "expert",
          "autoregressive",
      ]
  )  # Target
  logical_axis_rules: List[List[Any]] = Field(
      default_factory=lambda: [
          ["activation_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
          ["activation_batch_no_exp", ["data", "fsdp", "fsdp_transpose"]],
          ["activation_embed_and_logits_batch", ["data", "stage", "fsdp", "fsdp_transpose", "expert"]],
          ["activation_heads", ["tensor", "tensor_transpose", "sequence", "tensor_sequence", "autoregressive"]],
          ["activation_kv_heads", ["tensor", "tensor_transpose", "sequence", "tensor_sequence"]],
          ["activation_length", ["sequence", "context"]],
          ["activation_length", ["context"]],
          ["activation_norm_length", ["tensor_sequence", "context", "sequence"]],
          ["activation_q_length", ["context"]],
          ["activation_kv_length", []],
          ["activation_embed", ["tensor", "tensor_transpose"]],
          ["activation_mlp", ["tensor", "tensor_transpose", "tensor_sequence"]],
          ["activation_kv", ["tensor", "tensor_transpose", "tensor_sequence"]],
          ["activation_prefill_kv_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
          ["activation_kv_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
          ["activation_kv_head_dim", ["tensor", "tensor_transpose", "tensor_sequence"]],
          ["activation_vocab", ["tensor", "tensor_transpose", "sequence", "tensor_sequence"]],
          ["activation_vocab", ["tensor", "tensor_transpose"]],
          ["activation_vocab", "tensor_sequence"],
          ["activation_vocab", ["sequence", "context"]],
          ["activation_stage", "stage"],
          ["activation_exp", ["expert"]],
          ["decode_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
          ["decode_length", ["sequence"]],
          ["mlp", ["fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"]],
          ["vocab", ["tensor", "tensor_transpose", "tensor_sequence", "autoregressive"]],
          ["heads", ["tensor", "tensor_transpose", "tensor_sequence", "autoregressive"]],
          ["q_heads", ["tensor", "tensor_transpose", "tensor_sequence", "autoregressive"]],
          ["kv_heads", ["tensor", "tensor_transpose", "tensor_sequence", "autoregressive"]],
          ["embed", ["fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"]],
          ["embed", ["fsdp", "sequence", "tensor_transpose", "context", "expert"]],
          ["embed", ["fsdp", "fsdp_transpose", "sequence", "context", "expert"]],
          ["embed", ["fsdp", "sequence", "context", "expert"]],
          ["embed_no_exp", ["fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context"]],
          ["embed_no_exp", ["fsdp", "sequence", "tensor_transpose", "context"]],
          ["embed_no_exp", ["fsdp", "fsdp_transpose", "sequence", "context"]],
          ["embed_no_exp", ["fsdp", "sequence", "context"]],
          ["q_lora", ["fsdp", "fsdp_transpose", "sequence", "context", "tensor_transpose", "expert"]],
          ["q_lora", ["fsdp", "sequence", "context", "tensor_transpose", "expert"]],
          ["q_lora", ["fsdp", "fsdp_transpose", "sequence", "context", "expert"]],
          ["q_lora", ["fsdp", "sequence", "context", "expert"]],
          ["kv_lora", ["fsdp", "fsdp_transpose", "sequence", "context", "tensor_transpose", "expert"]],
          ["kv_lora", ["fsdp", "sequence", "context", "tensor_transpose", "expert"]],
          ["kv_lora", ["fsdp", "fsdp_transpose", "sequence", "context", "expert"]],
          ["kv_lora", ["fsdp", "sequence", "context", "expert"]],
          ["norm", ["tensor", "tensor_transpose", "tensor_sequence"]],
          ["layers", "stage"],
          ["kv", []],
          ["kv_head_dim", []],
          ["cache_batch_prefill", []],
          ["cache_batch", []],
          ["cache_heads_none", []],
          ["cache_heads", ["autoregressive", "tensor", "tensor_transpose", "tensor_sequence"]],
          ["cache_heads", ["autoregressive", "tensor", "tensor_sequence"]],
          ["cache_kv", []],
          ["cache_sequence", []],
          ["exp", "expert"],
          ["paged_kv_heads", ["tensor"]],
          ["num_pages", []],
          ["tokens_per_page", []],
          ["paged_kv_head_dim_size", []],
      ]
  )  # Target
  data_sharding: List[List[str]] = Field(
      default_factory=lambda: [  # Target
          [
              "data",
              "stage",
              "fsdp",
              "fsdp_transpose",
              "sequence",
              "context",
              "context_autoregressive",
              "tensor",
              "tensor_transpose",
              "tensor_sequence",
              "expert",
              "autoregressive",
          ]
      ]
  )
  input_data_sharding_logical_axes: List[str] = Field(  # Target
      default_factory=lambda: ["activation_embed_and_logits_batch", "activation_norm_length"]
  )
  sharding_tolerance: float = Field(default=0.02, ge=0.0, le=1.0)
  custom_mesh: Optional[str] = Field(default="")
  allow_split_physical_axes: bool = Field(default=False)
  optimize_mesh_for_tpu_v6e: bool = Field(default=False)
  context_parallel_load_balance: bool = Field(default=True)


class DcnParallelismConfig(BaseModel):
  """Data Center Network (inter-slice) parallelism dimensions."""

  dcn_data_parallelism: int = Field(default=-1)
  dcn_fsdp_parallelism: int = Field(default=1)
  dcn_fsdp_transpose_parallelism: int = Field(default=1)
  dcn_sequence_parallelism: int = Field(default=1)
  dcn_context_parallelism: int = Field(default=1)
  dcn_context_autoregressive_parallelism: int = Field(default=1)
  dcn_tensor_parallelism: int = Field(default=1)
  dcn_tensor_transpose_parallelism: int = Field(default=1)
  dcn_tensor_sequence_parallelism: int = Field(default=1)
  dcn_pipeline_parallelism: int = Field(default=1)
  dcn_expert_parallelism: int = Field(default=1)
  dcn_autoregressive_parallelism: int = Field(default=1)


class IciParallelismConfig(BaseModel):
  """Inter-Chip Interconnect (intra-slice) parallelism dimensions."""

  ici_data_parallelism: int = Field(default=1)
  ici_fsdp_parallelism: int = Field(default=-1)  # Target -1
  ici_fsdp_transpose_parallelism: int = Field(default=1)
  ici_sequence_parallelism: int = Field(default=1)
  ici_context_parallelism: int = Field(default=1)
  ici_context_autoregressive_parallelism: int = Field(default=1)
  ici_tensor_parallelism: int = Field(default=1)
  ici_tensor_transpose_parallelism: int = Field(default=1)
  ici_tensor_sequence_parallelism: int = Field(default=1)
  ici_autoregressive_parallelism: int = Field(default=1)
  ici_pipeline_parallelism: int = Field(default=1)
  ici_expert_parallelism: int = Field(default=1)


class TokenizerConfig(BaseModel):
  """Configurations for the text tokenizer."""

  vocab_size: PositiveInt = Field(default=32_000)
  # tokenizer_path is in PathConfig
  tokenizer_type: TokenizerTypeEnum = Field(default=TokenizerTypeEnum.SENTENCEPIECE)  # Target
  use_chat_template: bool = Field(default=False)
  tokenize_train_data: bool = Field(default=True)
  tokenize_eval_data: bool = Field(default=True)
  add_bos: bool = Field(default=True)
  add_eos: bool = Field(default=True)


class DatasetBaseConfig(BaseModel):
  """Base dataset configurations common across sources."""

  per_device_batch_size: float = Field(default=1.0, gt=0.0)  # Target
  expansion_factor_real_data: int = Field(default=-1)
  eval_per_device_batch_size: NonNegativeFloat = Field(default=1.0)  # Target
  max_corpus_chars: Optional[PositiveInt] = Field(default=10_000_000)
  train_data_columns: List[str] = Field(default_factory=lambda: ["text"])
  eval_data_columns: List[str] = Field(default_factory=lambda: ["text"])
  packing: bool = Field(default=True)
  num_epoch: PositiveInt = Field(default=1)
  dataset_type: DatasetType = Field(default=DatasetType.TFDS)  # Target
  colocated_python_data_input: bool = Field(default=False)


class TFDSDatasetConfig(BaseModel):
  """TFDS-specific configurations."""

  # dataset_path is in PathConfig
  dataset_name: str = Field(default="c4/en:3.0.1")  # Target
  eval_dataset_name: str = Field(default="c4/en:3.0.1")  # Target
  train_split: str = Field(default="train")  # Target
  eval_split: str = Field(default="validation")  # Target


class HFDatasetConfig(BaseModel):
  """HuggingFace Datasets library specific configurations."""

  hf_path: Optional[str] = Field(default="")  # Target
  hf_data_dir: Optional[str] = Field(default="")
  hf_train_files: Optional[str] = Field(default="")
  hf_eval_split: Optional[str] = Field(default="")
  hf_eval_files: Optional[str] = Field(default="")
  hf_access_token: Optional[str] = Field(default=None)  # Target 'null'


class GrainDatasetConfig(BaseModel):
  """Google Grain library specific dataset configurations."""

  grain_train_files: Optional[str] = Field(default="")  # Target
  grain_eval_files: Optional[str] = Field(default="")  # Target
  grain_file_type: GrainFileType = Field(default=GrainFileType.ARRAYRECORD)  # Target
  grain_worker_count: NonNegativeInt = Field(default=1)
  grain_worker_count_eval: NonNegativeInt = Field(default=1)


class DatasetSourcesConfig(BaseModel):
  """Container for all dataset source configurations."""

  base: DatasetBaseConfig = Field(default_factory=DatasetBaseConfig)
  tfds: Optional[TFDSDatasetConfig] = Field(default_factory=TFDSDatasetConfig)
  hf: Optional[HFDatasetConfig] = Field(default_factory=HFDatasetConfig)
  grain: Optional[GrainDatasetConfig] = Field(default_factory=GrainDatasetConfig)


class TrainingConfig(BaseModel):
  """Configurations for the training loop process and data handling."""

  reuse_example_batch: NonNegativeInt = Field(default=0)
  max_target_length: PositiveInt = Field(default=128)  # Target
  max_prefill_predict_length: PositiveInt = Field(default=16)
  enable_dropout: bool = Field(default=True)
  enable_data_shuffling: bool = Field(default=True)
  data_shuffle_seed: NonNegativeInt = Field(default=0)
  init_weights_seed: NonNegativeInt = Field(default=0)
  gradient_clipping_threshold: NonNegativeFloat = Field(default=1.0)
  gradient_accumulation_steps: PositiveInt = Field(default=1)
  optimizer_memory_host_offload: bool = Field(default=False)
  parameter_memory_host_offload: bool = Field(default=False)
  scan_layers: bool = Field(default=True)


class LearningRateSch(BaseModel):  # Renamed
  """Configurations for the learning rate schedule."""

  learning_rate: NonNegativeFloat = Field(default=3.0e-5)
  cosine_learning_rate_final_fraction: NonNegativeFloat = Field(default=0.1, le=1.0)
  warmup_steps_fraction: NonNegativeFloat = Field(default=0.1, le=1.0)
  learning_rate_schedule_steps: int = Field(default=150_001)  # Target


class AdamWParams(BaseModel):  # Renamed
  """AdamW-specific optimizer hyperparameters."""

  adam_b1: float = Field(default=0.9, gt=0.0, lt=1.0)
  adam_b2: float = Field(default=0.95, gt=0.0, lt=1.0)
  adam_eps: float = Field(default=1.0e-8, gt=0.0)
  adam_eps_root: NonNegativeFloat = Field(default=0.0)
  adam_weight_decay: NonNegativeFloat = Field(default=0.1)
  mu_dtype: Optional[str] = Field(default="float32", description="Data type for AdamW 'mu'. Target 'float32'.")


class RoPESettingsConfig(BaseModel):  # Renamed
  """Rotary Position Embedding (RoPE) configurations, including YARN defaults."""

  rope_type: RoPEType = Field(default=RoPEType.DEFAULT)
  rope_use_scale: bool = Field(default=True)
  rope_min_timescale: PositiveInt = Field(default=1)
  rope_max_timescale: PositiveInt = Field(default=10_000)
  local_rope_max_timescale: int = Field(default=-1)
  max_position_embeddings: Optional[PositiveInt] = Field(default=163_840)  # Target
  original_max_position_embeddings: Optional[PositiveInt] = Field(default=4_096)  # Target
  rope_factor: Optional[PositiveInt] = Field(default=40)  # Target
  beta_fast: Optional[PositiveInt] = Field(default=32)  # Target
  beta_slow: Optional[PositiveInt] = Field(default=1)  # Target
  mscale: Optional[NonNegativeFloat] = Field(default=1.0)  # Target


class YarnRoPEOptionalConfig(BaseModel):  # Correctly defined and named
  """Optional dedicated container for YARN RoPE parameters if structured separately."""

  max_position_embeddings: PositiveInt = Field(default=163_840)
  original_max_position_embeddings: PositiveInt = Field(default=4_096)
  rope_factor: PositiveInt = Field(default=40)
  beta_fast: PositiveInt = Field(default=32)
  beta_slow: PositiveInt = Field(default=1)
  mscale: NonNegativeFloat = Field(default=1.0)


class GenerationPromptConfig(BaseModel):
  """Configurations for prompts used in text generation."""

  prompt: str = Field(default="I love to")
  load_from_prefill_dir: bool = Field(default=False)
  autoregressive_decode_assert: Optional[str] = Field(default="")


class DecodingAlgo(BaseModel):  # Renamed
  """Configurations for text decoding algorithms and sampling."""

  decode_sampling_strategy: SamplingStrategy = Field(default=SamplingStrategy.GREEDY)
  decode_sampling_nucleus_p: float = Field(default=-1.0, ge=-1.0, le=1.0)  # Target -1
  decode_sampling_top_k: NonNegativeInt = Field(default=0)
  decode_sampling_temperature: NonNegativeFloat = Field(default=1.0)


class EvaluationSetup(BaseModel):  # Renamed
  """Configurations for controlling evaluation runs during training."""

  eval_interval: int = Field(default=-1)
  eval_steps: int = Field(default=-1)
  target_eval_loss: NonNegativeFloat = Field(default=0.0)


class ProfilingSetup(BaseModel):  # Renamed
  """Configurations for performance profiling."""

  profiler: ProfilerType = Field(default=ProfilerType.NONE)  # Target ""
  upload_all_profiler_results: bool = Field(default=False)
  skip_first_n_steps_for_profiler: NonNegativeInt = Field(default=1)
  profiler_steps: PositiveInt = Field(default=5)
  profile_cleanly: bool = Field(default=True)
  profile_periodically_period: int = Field(default=-1)


class HloDumpSetup(BaseModel):  # Renamed
  """Configurations for dumping HLO graphs from XLA."""

  dump_hlo: bool = Field(default=False)
  dump_step: int = Field(default=-1)
  dump_hlo_local_dir: str = Field(default=os.path.join(gettempdir(), "xla_dump", ""))
  dump_hlo_delete_local_after: bool = Field(default=True)
  dump_hlo_gcs_dir: Optional[str] = Field(default="")
  dump_hlo_module_name: str = Field(default="jit_train_step")
  dump_hlo_xla_flags: Optional[str] = Field(default="")
  dump_hlo_upload_all: bool = Field(default=False)


class AttentionLayoutSetup(BaseModel):  # Renamed
  """Configurations for KV Cache memory layout and attention compute layout."""

  prefill_cache_axis_order: str = Field(default="1,2,0,3")
  ar_cache_axis_order: str = Field(default="1,2,0,3")
  compute_axis_order: str = Field(default="0,1,2,3")
  reshape_q: bool = Field(default=False)


class MaxEngineServerSetup(BaseModel):  # Renamed
  """Configurations for the MaxEngine (JetStream) server."""

  prometheus_port: NonNegativeInt = Field(default=0)
  enable_jax_profiler: bool = Field(default=False)
  jax_profiler_port: PositiveInt = Field(default=9999)
  inference_server: InferenceServerType = Field(default=InferenceServerType.MAXTEXT_INTERLEAVED)
  prefill_slice: Optional[str] = Field(default="v5e-16")
  generate_slice: Optional[str] = Field(default="v5e-16")


class SplashAttentionSetup(BaseModel):  # Renamed
  """Block size configurations for Splash Attention kernels."""

  sa_block_q: PositiveInt = Field(default=512)
  sa_block_kv: PositiveInt = Field(default=512)
  sa_block_kv_compute: PositiveInt = Field(default=512)
  sa_block_q_dkv: PositiveInt = Field(default=512)
  sa_block_kv_dkv: PositiveInt = Field(default=512)
  sa_block_kv_dkv_compute: PositiveInt = Field(default=512)
  sa_block_q_dq: PositiveInt = Field(default=512)
  sa_block_kv_dq: PositiveInt = Field(default=512)
  sa_use_fused_bwd_kernel: bool = Field(default=False)
  sa_q_layout: str = Field(default="HEAD_DIM_MINOR")
  sa_k_layout: str = Field(default="HEAD_DIM_MINOR")
  sa_v_layout: str = Field(default="HEAD_DIM_MINOR")


class PagedAttentionSetup(BaseModel):  # Renamed
  """Configurations for Paged Attention kernels."""

  pagedattn_num_pages: PositiveInt = Field(default=64)
  pagedattn_tokens_per_page: PositiveInt = Field(default=32)
  pagedattn_pages_per_compute_block: PositiveInt = Field(default=4)
  pagedattn_max_pages_per_group: int = Field(default=4)  # Target


class ChunkedPrefillSetup(BaseModel):  # Renamed
  """Configurations for chunked prefill optimization."""

  prefill_chunk_size: PositiveInt = Field(default=256)
  use_chunked_prefill: bool = Field(default=False)


class PrefixCachingSetup(BaseModel):  # Renamed
  """Configurations for prefix caching in JetStream."""

  enable_prefix_caching: bool = Field(default=False)
  prefix_caching_hbm_byte: PositiveInt = Field(default=10_000_000_000)
  prefix_caching_dram_byte: PositiveInt = Field(default=100_000_000_000)


class Llama4Model(BaseModel):  # Renamed
  """Llama4-specific model architecture configurations."""

  use_qk_norm: bool = Field(default=False)
  nope_layer_interval: int = Field(default=-1)
  interleave_moe_layer_step: PositiveInt = Field(default=1)
  temperature_tuning: bool = Field(default=False)


class Multimodal(BaseModel):  # Renamed
  """Configurations for multimodal models (e.g., Vision Transformer + LLM)."""

  use_multimodal: bool = Field(default=False)
  freeze_vision_encoder_params: bool = Field(default=True)
  dtype_mm: str = Field(default="float32")
  remat_policy_for_vit: RematPolicy = Field(default=RematPolicy.MINIMAL)
  image_size_for_vit: PositiveInt = Field(default=896)
  image_path: Optional[str] = Field(default="")


class Llama4VitConfig(BaseModel):  # Renamed
  """Llama4-specific Vision Transformer (ViT) architectural configurations."""

  hidden_size_for_vit: PositiveInt = Field(default=1408)
  intermediate_size_for_vit: PositiveInt = Field(default=5632)
  num_attention_heads_for_vit: PositiveInt = Field(default=16)
  num_channels_for_vit: PositiveInt = Field(default=3)
  patch_size_for_vit: PositiveInt = Field(default=14)
  num_hidden_layers_for_vit: PositiveInt = Field(default=34)
  projector_input_dim_for_vit: PositiveInt = Field(default=4096)
  projector_output_dim_for_vit: PositiveInt = Field(default=4096)
  rope_theta_for_vit: PositiveInt = Field(default=10000)
  vision_output_dim_for_vit: PositiveInt = Field(default=4096)
  pixel_shuffle_ratio_for_vit: float = Field(default=0.5, gt=0.0, lt=1.0)
  projector_dropout_for_vit: NonNegativeFloat = Field(default=0.0)


class DPOSetup(BaseModel):  # Renamed
  """Direct Preference Optimization (DPO) specific training configurations."""

  use_dpo: bool = Field(default=False)
  dpo_label_smoothing: NonNegativeFloat = Field(default=0.0)
  dpo_beta: NonNegativeFloat = Field(default=0.1)


class SFTSetup(BaseModel):  # Renamed
  """Supervised Fine-Tuning (SFT) specific training configurations."""

  use_sft: bool = Field(default=False)
  sft_train_on_completion_only: bool = Field(default=False)


class DebugStacktraceConfig(BaseModel):
  """Configurations for collecting Python stack traces for debugging."""

  collect_stack_trace: bool = Field(default=False)
  stack_trace_to_cloud: bool = Field(default=False)
  stack_trace_interval_seconds: PositiveInt = Field(default=600)


class GcpMonitorConfig(BaseModel):  # Renamed
  """Configurations for GCP workload monitoring and goodput metrics."""

  report_heartbeat_metric_for_gcp_monitoring: bool = Field(default=False)
  heartbeat_reporting_interval_in_seconds: PositiveInt = Field(default=5)
  report_performance_metric_for_gcp_monitoring: bool = Field(default=False)
  enable_gcp_goodput_metrics: bool = Field(default=True)
  enable_gcp_step_deviation_metrics: bool = Field(default=True)
  enable_goodput_recording: bool = Field(default=False)  # Differs from monitor_goodput
  goodput_upload_interval_seconds: PositiveInt = Field(default=30)
  enable_pathways_goodput: bool = Field(default=False)
  monitor_step_time_deviation: bool = Field(default=True)
  step_deviation_interval_seconds: PositiveInt = Field(default=30)


class DPOSettings(BaseModel):
  """Direct Preference Optimization (DPO) specific training configurations."""

  use_dpo: bool = Field(
      default=False, description="Enable Direct Preference Optimization (DPO) training mode. Target: False."
  )
  dpo_label_smoothing: NonNegativeFloat = Field(
      default=0.0, ge=0.0, le=1.0, description="Label smoothing factor for the DPO loss. Target: 0.0."
  )
  dpo_beta: NonNegativeFloat = Field(
      default=0.1, description="Beta parameter for DPO loss, controlling divergence from reference model. Target: 0.1."
  )


class SFTSettings(BaseModel):
  """Supervised Fine-Tuning (SFT) specific training configurations."""

  use_sft: bool = Field(default=False, description="Enable Supervised Fine-Tuning (SFT) training mode. Target: False.")
  sft_train_on_completion_only: bool = Field(
      default=False, description="Train only on completion tokens in SFT. Target: False."
  )


class InferenceBenchmarkConfig(BaseModel):  # Renamed
  """Configurations for running inference microbenchmarks."""

  inference_microbenchmark_prefill_lengths: str = Field(default="64,128,256,512,1024")
  inference_microbenchmark_stages: str = Field(default="prefill,generate")
  inference_microbenchmark_loop_iters: PositiveInt = Field(default=10)
  inference_microbenchmark_log_file_path: Optional[str] = Field(default="")
  inference_microbenchmark_num_samples: List[PositiveInt] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
  inference_metadata_file: Optional[str] = Field(default="")
  inference_benchmark_test: bool = Field(default=False)
  enable_model_warmup: bool = Field(default=False)
  enable_llm_inference_pool: bool = Field(default=False)
  multi_sampling: bool = Field(default=False)
  return_log_prob: bool = Field(default=False)


class GlobalBatchInfoConfig(BaseModel):  # Renamed
  """Configuration for various global and micro batch sizes."""

  global_batch_size_to_eval_on: int = Field(default=1)
  global_batch_size_to_load: int = Field(default=1)
  global_batch_size_to_load_eval: int = Field(default=1)
  global_batch_size_to_train_on: int = Field(default=1)
  micro_batch_size_to_eval_on: int = Field(default=1)
  micro_batch_size_to_train_on: int = Field(default=1)


# -----------------------------------------------------------------------------
# Main Configuration Model
# -----------------------------------------------------------------------------


class MaxTextConfig(BaseModel):
  """Top-level configuration model for MaxText training and inference runs."""

  # Core Settings
  path_config: PathConfig = Field(default_factory=PathConfig)
  general_run_setting: GeneralRunSetting = Field(default_factory=GeneralRunSetting)
  vertex_ai_setting: VertexAiSetting = Field(default_factory=VertexAiSetting)

  # Checkpointing
  checkpoint_load: CheckpointLoadSetting = Field(default_factory=CheckpointLoadSetting)
  checkpoint_save: CheckpointSaveSetting = Field(default_factory=CheckpointSaveSetting)
  checkpoint_store: CheckpointStoreSetting = Field(default_factory=CheckpointStoreSetting)
  emergency_ckpt: EmergencyCkptSetting = Field(default_factory=EmergencyCkptSetting)
  checkpoint_meta: CheckpointMiscSetting = Field(default_factory=CheckpointMiscSetting)

  # Model Definition
  model_id: ModelIdSetting = Field(default_factory=ModelIdSetting)
  model_architecture: ModelArchitecture = Field(default_factory=ModelArchitecture)
  model_operations: ModelOperationConfig = Field(default_factory=ModelOperationConfig)
  model_positional_embedding: ModelPositionalEmbedding = Field(default_factory=ModelPositionalEmbedding)
  activations_logits: ActivationLogitConfig = Field(default_factory=ActivationLogitConfig)

  # Specialized Model Features
  model_quantization: ModelQuantizeConfig = Field(default_factory=ModelQuantizeConfig)
  moe_general: MoEGeneral = Field(default_factory=MoEGeneral)
  moe_tiling: MoETiling = Field(default_factory=MoETiling)
  deepseek_moe: Optional[DeepSeekMoEOverrides] = Field(default=None)
  pipeline_config: PipelineConfig = Field(default_factory=PipelineConfig)
  remat_policy_config: RematPolicyConfig = Field(default_factory=RematPolicyConfig)

  # Attention Mechanisms
  attention_kernel: AttentionKernelSetting = Field(default_factory=AttentionKernelSetting)
  attention_fusion: AttentionOpFusionSetting = Field(default_factory=AttentionOpFusionSetting)
  attention_behavior: AttentionExtraBehavior = Field(default_factory=AttentionExtraBehavior)
  mla_parameters: Optional[MlaParams] = Field(default=None)

  # Hardware, Parallelism, and Compilation
  hardware: HardwarePlatform = Field(default_factory=HardwarePlatform)
  aot_compilation: AotCompileConfig = Field(default_factory=AotCompileConfig)
  mesh_layout: MeshConfig = Field(default_factory=MeshConfig)
  dcn_dims: DcnParallelismConfig = Field(default_factory=DcnParallelismConfig)
  ici_dims: IciParallelismConfig = Field(default_factory=IciParallelismConfig)

  # Data Pipeline
  tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
  dataset_sources: DatasetSourcesConfig = Field(default_factory=DatasetSourcesConfig)

  # Training Process & Optimization
  training_settings: TrainingConfig = Field(default_factory=TrainingConfig)
  learning_rate_schedule: LearningRateSch = Field(default_factory=LearningRateSch)
  adamw_optimizer: AdamWParams = Field(default_factory=AdamWParams)
  opt_type: OptimizerType = Field(default=OptimizerType.ADAMW, description="Global optimizer type selection.")

  # RoPE and Generation
  rope: RoPESettingsConfig = Field(default_factory=RoPESettingsConfig)
  yarn_rope_optional: Optional[YarnRoPEOptionalConfig] = Field(default=None)
  generation_prompt: GenerationPromptConfig = Field(default_factory=GenerationPromptConfig)
  decoding_algorithm: DecodingAlgo = Field(default_factory=DecodingAlgo)

  # Evaluation, Profiling, Debugging
  evaluation_setup: EvaluationSetup = Field(default_factory=EvaluationSetup)
  profiling_setup: ProfilingSetup = Field(default_factory=ProfilingSetup)
  hlo_dump_setup: HloDumpSetup = Field(default_factory=HloDumpSetup)
  debug_stacktrace: DebugStacktraceConfig = Field(default_factory=DebugStacktraceConfig)

  # Advanced/Specialized Features for Inference and Kernels
  attention_layout: AttentionLayoutSetup = Field(default_factory=AttentionLayoutSetup)
  maxengine_server_setup: MaxEngineServerSetup = Field(default_factory=MaxEngineServerSetup)
  splash_attention_setup: SplashAttentionSetup = Field(default_factory=SplashAttentionSetup)
  paged_attention_setup: PagedAttentionSetup = Field(default_factory=PagedAttentionSetup)
  chunked_prefill_setup: ChunkedPrefillSetup = Field(default_factory=ChunkedPrefillSetup)
  prefix_caching_setup: PrefixCachingSetup = Field(default_factory=PrefixCachingSetup)

  # Model-Specific Blocks (Optional, will be None -> null if not applicable)
  llama4_parameters: Optional[Llama4Model] = Field(default=None)
  multimodal_config: Multimodal = Field(default_factory=Multimodal)
  llama4_vit_parameters: Optional[Llama4VitConfig] = Field(default=None)
  dpo_config: Optional[DPOSettings] = Field(
      default=None,
      alias="dpo_specific_config",
      description="Settings specific to Direct Preference Optimization, if applicable.",
  )
  sft_config: Optional[SFTSettings] = Field(
      default=None, alias="sft_specific_config", description="Settings specific to Supervised Fine-Tuning, if applicable."
  )

  # GCP Monitoring and Inference Benchmarking
  gcp_monitor_config: GcpMonitorConfig = Field(default_factory=GcpMonitorConfig)
  inference_benchmark_config: InferenceBenchmarkConfig = Field(default_factory=InferenceBenchmarkConfig)

  # Global Batch Sizes (matching "Expect This" direct fields)
  global_batch_details: GlobalBatchInfoConfig = Field(default_factory=GlobalBatchInfoConfig)  # Renamed attribute

  @computed_field()
  @property
  def ici_parallelism(self) -> List[int]:
    """Computed list of ICI parallelism values in a predefined order."""
    p = self.ici_dims
    return [
        p.ici_data_parallelism,
        p.ici_pipeline_parallelism,
        p.ici_fsdp_parallelism,
        p.ici_fsdp_transpose_parallelism,
        p.ici_sequence_parallelism,
        p.ici_context_parallelism,
        p.ici_context_autoregressive_parallelism,
        p.ici_tensor_parallelism,
        p.ici_tensor_transpose_parallelism,
        p.ici_tensor_sequence_parallelism,
        p.ici_expert_parallelism,
        p.ici_autoregressive_parallelism,
    ]

  @computed_field()
  @property
  def dcn_parallelism(self) -> List[int]:
    """Computed list of DCN parallelism values in a predefined order."""
    p = self.dcn_dims
    return [
        p.dcn_data_parallelism,
        p.dcn_pipeline_parallelism,
        p.dcn_fsdp_parallelism,
        p.dcn_fsdp_transpose_parallelism,
        p.dcn_sequence_parallelism,
        p.dcn_context_parallelism,
        p.dcn_context_autoregressive_parallelism,
        p.dcn_tensor_parallelism,
        p.dcn_tensor_transpose_parallelism,
        p.dcn_tensor_sequence_parallelism,
        p.dcn_expert_parallelism,
        p.dcn_autoregressive_parallelism,
    ]

  @model_validator(mode="before")
  @classmethod
  def _populate_path_config_run_name(cls, data: Any) -> Any:
    if isinstance(data, dict):
      run_name_from_data = None
      # Check if run_name is under a general run settings dict or top-level
      if (
          "general_run_setting" in data
          and isinstance(data["general_run_setting"], dict)
          and "run_name" in data["general_run_setting"]
      ):
        run_name_from_data = data["general_run_setting"]["run_name"]
      elif "run_name" in data:  # Fallback if run_name was passed top-level (less likely with new structure)
        run_name_from_data = data["run_name"]

      if run_name_from_data:
        if "path_config" not in data:  # Ensure path_config dict exists
          data["path_config"] = {}
        if isinstance(data["path_config"], dict):  # Ensure it's a dict before assigning
          data["path_config"]["run_name_for_paths_internal_only"] = run_name_from_data
    return data

  model_config = ConfigDict(populate_by_name=True, extra="forbid")
