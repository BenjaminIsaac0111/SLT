#!/bin/bash
#SBATCH --job-name=fine_tune_array
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-17

# Example Slurm job array for running fine-tuning experiments across
# multiple regimes and cross-validation folds. Adjust the paths below
# to match your dataset structure.

module load anaconda || true
source activate SLT || true

REGIMES=(B0 C0 F0 F1 M0 M1)
FOLDS=(fold1 fold2 fold3)

REGIME_INDEX=$(( SLURM_ARRAY_TASK_ID % ${#REGIMES[@]} ))
FOLD_INDEX=$(( SLURM_ARRAY_TASK_ID / ${#REGIMES[@]} ))

REGIME=${REGIMES[$REGIME_INDEX]}
FOLD=${FOLDS[$FOLD_INDEX]}

LABELS_JSON=/path/to/${FOLD}/${REGIME}_labels.json
IMAGES_DIR=/path/to/${FOLD}/${REGIME}_images
VAL_LABELS_JSON=/path/to/${FOLD}/val_labels.json
VAL_IMAGES_DIR=/path/to/${FOLD}/val_images
INITIAL_WEIGHTS=/path/to/pretrained_weights.h5
MODEL_DIR=/path/to/output

python DeepLearning/training/fine_tuning.py \
    --labels_json "$LABELS_JSON" \
    --images_dir "$IMAGES_DIR" \
    --val_labels_json "$VAL_LABELS_JSON" \
    --val_images_dir "$VAL_IMAGES_DIR" \
    --initial_weights "$INITIAL_WEIGHTS" \
    --model_dir "$MODEL_DIR" \
    --model_name "${FOLD}_${REGIME}" \
    --validate_every 5 \
    --calibrate_every 10

