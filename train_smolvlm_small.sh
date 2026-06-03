#!/bin/bash
# SimVLA Training Script for LIBERO (Small Model)
# 
# Key features:
#   - 384x384 image resolution (SmolVLM requirement)
#   - All views processed together by VLM (no aux_visual_inputs)
#   - Smaller action transformer configuration

set -e

# =============================================================================
# Command line arguments (with defaults)
# =============================================================================

BATCH_SIZE=${1:-64}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./runs/simvla_libero_small}
RESUME_CKPT=${4:-""}
GRADIENT_ACCUMULATION_STEPS=${5:-${GRADIENT_ACCUMULATION_STEPS:-1}}

echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"
echo "   gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}"

# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Suppress TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONNOUSERSITE=1

# =============================================================================
# Path configuration
# =============================================================================
LIBERO_DATA_DIR="./datasets/metas"
NORM_STATS_PATH="./norm_stats/libero_norm.json"
TRAIN_METAS_PATH="./datasets/metas/libero_train.json"
LIBERO_SUBSETS="libero_goal libero_object libero_spatial libero_100"
EXPECTED_LIBERO_STEPS=1007618

# SmolVLM backbone (can be local path or HuggingFace repo)
SMOLVLM_MODEL=${SMOLVLM_MODEL:-./pretrained/SmolVLM-500M-Instruct}

# =============================================================================
# Training hyperparameters
# =============================================================================
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_ACTIONS=10          # Action horizon
ITERS=800000
WARMUP_STEPS=0
FREEZE_STEPS=1000
SAVE_INTERVAL=20000
LOG_INTERVAL=20
NUM_WORKERS=4
MAX_GRAD_NORM=1.0
NUM_PROCESSES=${NUM_PROCESSES:-4}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29504}
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))
MAX_LEN_SEQ=2048
VLM_TORCH_DTYPE=${VLM_TORCH_DTYPE:-float32}

# Model architecture (Small configuration)
HIDDEN_SIZE=768         
DEPTH=12                 
NUM_HEADS=12             
USE_ADALN=false          # DiT-style conditioning

# =============================================================================
# Step 1: Validate dataset and refresh training metadata
# =============================================================================
echo "Refreshing training metadata..."
python create_libero_meta.py \
    --data_dir $LIBERO_DATA_DIR \
    --subsets $LIBERO_SUBSETS \
    --output $TRAIN_METAS_PATH

# =============================================================================
# Step 2: Compute normalization statistics (if not exists)
# =============================================================================
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "Computing normalization statistics..."
    python compute_libero_norm_stats.py \
        --data_dir $LIBERO_DATA_DIR \
        --subsets $LIBERO_SUBSETS \
        --output $NORM_STATS_PATH
else
    python - "$NORM_STATS_PATH" "$EXPECTED_LIBERO_STEPS" <<'PY'
import json
import sys

path = sys.argv[1]
expected_steps = int(sys.argv[2])
with open(path) as f:
    data = json.load(f)
metadata = data.get("metadata", {})
num_steps = metadata.get("num_steps")
if num_steps != expected_steps:
    raise SystemExit(
        f"ERROR: {path} has metadata.num_steps={num_steps}, expected {expected_steps} "
        "for the full LIBERO training split. Delete/regenerate it after repairing the dataset."
    )
print(f"Existing norm stats OK: num_steps={num_steps}")
PY
fi

# =============================================================================
# Step 3: Build training arguments
# =============================================================================
ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode libero_joint \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --num_actions ${NUM_ACTIONS} \
    --iters ${ITERS} \
    --warmup_steps ${WARMUP_STEPS} \
    --freeze_steps ${FREEZE_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --image_size 384 \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --max_len_seq ${MAX_LEN_SEQ} \
    --vlm_torch_dtype ${VLM_TORCH_DTYPE}"

# Add AdaLN flag if enabled
if [ "${USE_ADALN}" = true ]; then
    ARGS="${ARGS} --use_adaln"
fi

# Add resume checkpoint if specified
if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

# =============================================================================
# Step 4: Start training
# =============================================================================
echo "============================================================"
echo "Starting SimVLA Training on LIBERO (Small Action Transformer)"
echo "============================================================"
echo "SmolVLM backbone: ${SMOLVLM_MODEL}"
echo "Data directory: $LIBERO_DATA_DIR"
echo "Normalization stats: $NORM_STATS_PATH"
echo "Action mode: libero_joint"
echo "Batch size: ${BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective global batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Learning coef: ${LEARNING_COEF}"
echo "Num actions: ${NUM_ACTIONS}"
echo "Image size: 384x384"
echo "============================================================"
echo "Action Transformer configuration:"
echo "   Hidden size: ${HIDDEN_SIZE}"
echo "   Depth: ${DEPTH}"
echo "   Num heads: ${NUM_HEADS}"
echo "   Use AdaLN: ${USE_ADALN}"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"

# Multi-GPU training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port ${MAIN_PROCESS_PORT} \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
