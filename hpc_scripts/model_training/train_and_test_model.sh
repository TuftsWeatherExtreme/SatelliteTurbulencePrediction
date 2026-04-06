#!/bin/bash -l

# train_and_test_model.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Train and test the ConvLSTM satellite turbulence model using GPU resources.

#SBATCH -J sat_train
#SBATCH --time=02-00:00:00
#SBATCH -p preempt
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=32g
#SBATCH --output=sat_train.%j.%N.out
#SBATCH --error=sat_train.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd $SAT_REPO_PATH
source $SAT_REPO_PATH/hpc_scripts/load_modules.sh
nvidia-smi

seed=$1
model_type=$2

echo "About to train the $model_type model with seed $seed"
python -u $SAT_REPO_PATH/src/train_and_test_model.py $seed $model_type \
    --data-dir $SAT_REPO_PATH/model_inputs
echo "Finished training and testing the model!"

source $SAT_REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"
