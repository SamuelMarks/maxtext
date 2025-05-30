#!/bin/bash

# This file is both an integration test that runs once a day on a v4-8 and documentation for how to get started with Gemma3-12b.

# The flow of this file is as follows:
# 1. Convert the checkpoint downloaded from Kaggle to make it compatible with MaxText
# 2. Run decoding, finetuning of Gemma3-12b with the converted checkpoint. Also, run pretraining of Gemma3-12b
# 3. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.


set -ex
idx=$(date +%Y-%m-%d-%H-%M)
export MODEL_VARIATION='27b'
export MODEL_NAME=gemma3-${MODEL_VARIATION}

# Installing torch for deps in forward_pass_logit_checker
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Non-Googlers please remember to use separate GCS paths for uploading model weights from kaggle ($CHKPT_BUCKET) and MaxText compatible weights ($MODEL_BUCKET).
# Non-Googlers please remember to point these variables to GCS buckets that you own, this script uses internal buckets for testing.
# You can use the Flax checkpoint available on Kaggle:
# https://www.kaggle.com/models/google/gemma-3/flax/

export CHKPT_BUCKET=gs://maxtext-gemma/gemma3/flax
export MODEL_BUCKET=gs://maxtext-gemma/gemma3

python3 -m MaxText.convert_gemma3_chkpt --base_model_path ${CHKPT_BUCKET}/${MODEL_VARIATION} --maxtext_model_path ${MODEL_BUCKET}/${MODEL_VARIATION}/${idx} --model_size ${MODEL_VARIATION}


# Non-Googlers please remember to point `DATASET_PATH` to the GCS bucket where you have your training data
export DATASET_PATH=gs://maxtext-dataset
# Non-Googlers please remember to point `BASE_OUTPUT_DIRECTORY` to a GCS bucket that you own, this bucket will store all the files generated by MaxText during a run
export BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
# We define `CONVERTED_CHECKPOINT` to refer to the checkpoint subdirectory. This way it is easier to use this path in the `train` and `decode` commands
export CONVERTED_CHECKPOINT=${MODEL_BUCKET}/${MODEL_VARIATION}/${idx}/0/items
export RUN_NAME=unscanned_chkpt_${idx}
# Note that the `CONVERTED_CHECKPOINT` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
# We can do this by running `MaxText.generate_param_only_checkpoint` on `CONVERTED_CHECKPOINT` with `force_unroll=true`.
JAX_PLATFORMS=cpu python3 -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} load_parameters_path=${CONVERTED_CHECKPOINT} run_name=${RUN_NAME} model_name=${MODEL_NAME} force_unroll=true

export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints/0/items

# We run decoding on the `UNSCANNED_CKPT_PATH` for efficient decoding on the unscanned version of the checkpoint. Note that this checkpoint only has parameters and no optimizer state.
# So, we use it by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`
python3 -m MaxText.decode MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma3 load_parameters_path=${UNSCANNED_CKPT_PATH} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false scan_layers=false model_name=${MODEL_NAME} prompt="I love to"

# # to get higher precision (eg. float32) run on CPU with `JAX_PLATFORMS=cpu`
python3 -m MaxText.tests.forward_pass_logit_checker  MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma3 load_parameters_path=${UNSCANNED_CKPT_PATH} run_name=forward_pass_test_${MODEL_NAME} per_device_batch_size=1 model_name=${MODEL_NAME} max_prefill_predict_length=4 max_target_length=4 dataset_type=synthetic scan_layers=false --atol=1.0 --rtol=1.0

# Finetune by using the scanned converted checkpoint by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`. For Googlers, uncomment the line below if you want to use the pre-converted checkpoint.
# export CONVERTED_CHECKPOINT=gs://maxtext-model-checkpoints/gemma3-27b/2025-03-20-00-12/scanned/0/items
FINETUNE_RUN_NAME=runner_finetune_${idx}
python3 -m MaxText.train MaxText/configs/base.yml model_name=$MODEL_NAME base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} load_parameters_path=${CONVERTED_CHECKPOINT} tokenizer_path=assets/tokenizer.gemma3 per_device_batch_size=1 run_name=$FINETUNE_RUN_NAME steps=10 enable_checkpointing=true sharding_tolerance=0.03

# We also run pre-training, this is similar to the finetuning command except we don't pass any checkpoint directory to load_parameters_path
PRETRAIN_RUN_NAME=runner_pretrain_${idx}
python3 -m MaxText.train MaxText/configs/base.yml model_name=$MODEL_NAME base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} tokenizer_path=assets/tokenizer.gemma3 per_device_batch_size=1 run_name=$PRETRAIN_RUN_NAME steps=10 enable_checkpointing=false sharding_tolerance=0.03
