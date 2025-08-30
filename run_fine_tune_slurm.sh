#!/bin/bash
#SBATCH --job-name=fine_tune
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Usage: sbatch run_fine_tune_slurm.sh --labels_json labels.json --images_dir imgs \
#        --initial_weights model.h5 --model_name exp1

module load anaconda || true
source activate SLT || true

python DeepLearning/training/fine_tuning.py "$@" --use_batch_alpha
