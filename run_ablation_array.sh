#!/bin/bash
#SBATCH --job-name=f1_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --array=0-8

FOLD="f1"
REGIME="BALD"

DATA_DIR="/users/scbiw/DeepLearning/fine_tuning_experiments/data"
IMAGES_DIR="/mnt/scratch/scbiw/ALL_PATCHES"
MODEL_DIR="/mnt/scratch/scbiw/fine_tuning_experiments/models"
WEIGHTS_DIR="/users/scbiw/DeepLearning/fine_tuning_experiments/pretrained"

NEW_LABELS_JSON="${DATA_DIR}/${FOLD}_training_regime_BALD.json"
LABELS_JSON="${DATA_DIR}/${FOLD}_training_regime_CR07.json"
VAL_LABELS_JSON="${DATA_DIR}/${FOLD}_val_CR07.json"
INITIAL_WEIGHTS="${WEIGHTS_DIR}/best_dropout_attention_unet_fl_${FOLD}.h5"

NUM_PATCHES=1024
NUM_VAL_PATCHES=-1
LEARNING_RATE=1e-7
WARMUP_STEPS=1024

SCHEDULES=(half_life linear cosine)
DECAY_RATES=(10240 20480 40860)

NUM_RATES=${#DECAY_RATES[@]}
SCH_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_RATES ))
RATE_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_RATES ))

DECAY_SCHEDULE=${SCHEDULES[$SCH_IDX]}
HALF_LIFE=${DECAY_RATES[$RATE_IDX]}

MODEL_NAME="${FOLD}_${REGIME}_${DECAY_SCHEDULE}_t${HALF_LIFE}_run_w_cm"

for f in "$LABELS_JSON" "$VAL_LABELS_JSON" "$INITIAL_WEIGHTS"; do
  [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }
done

module load miniforge/24.7.1
module load cuda/12.4.1
conda activate tf215gpu

export PYTHONPATH="/users/scbiw/DeepLearning/Attention-UNET:$PYTHONPATH"
echo "Host: $(hostname)"
echo "Task $SLURM_ARRAY_TASK_ID ? schedule=$DECAY_SCHEDULE, half_life=$HALF_LIFE"
nvidia-smi

python /users/scbiw/DeepLearning/Attention-UNET/DeepLearning/training/fine_tuning.py \
  --labels_json       "$LABELS_JSON" \
  --images_dir        "$IMAGES_DIR" \
  --new_labels_json   "$NEW_LABELS_JSON" \
  --new_images_dir    "$IMAGES_DIR" \
  --warmup_steps      "$WARMUP_STEPS" \
  --decay_schedule    "$DECAY_SCHEDULE" \
  --half_life         "$HALF_LIFE" \
  --val_labels_json   "$VAL_LABELS_JSON" \
  --val_images_dir    "$IMAGES_DIR" \
  --initial_weights   "$INITIAL_WEIGHTS" \
  --model_dir         "$MODEL_DIR" \
  --model_name        "$MODEL_NAME" \
  --num_patches       "$NUM_PATCHES" \
  --num_val_patches   "$NUM_VAL_PATCHES" \
  --learning_rate     "$LEARNING_RATE" \
  --validate_every    1 \
  --calibrate_every   0 \
  --shuffle_buffer_size 1024 \
  --epochs 128
