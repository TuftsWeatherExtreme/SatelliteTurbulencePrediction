#!/bin/bash -l

# smoke_test.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Runs a quick smoke test of the training pipeline on 50 samples
#           to confirm the full pipeline works before committing to a full run.
# Usage: sbatch smoke_test.sh

#SBATCH -J smoke_test
#SBATCH --time=00:30:00
#SBATCH -p batch
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH --output=smoke_test.%j.out
#SBATCH --error=smoke_test.%j.err


cd $SAT_REPO_PATH
source $SAT_REPO_PATH/hpc_scripts/load_modules.sh

echo "Starting smoke test..."
echo "SAT_REPO_PATH: $SAT_REPO_PATH"
echo "Data dir: $SAT_REPO_PATH/model_inputs"

python3 $SAT_REPO_PATH/src/train_and_test_model.py 42 conv3d \
    --max-samples 50 \
    --data-dir $SAT_REPO_PATH/model_inputs

echo "Smoke test exit code: $?"
echo "Smoke test complete!"

