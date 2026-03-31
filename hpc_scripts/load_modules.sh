#!/bin/bash

# load_modules.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Load all modules and activate specified environment.
# Important: Must be run as `source load_modules.sh`

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: Please run this script as: source ${0}"
    exit 1
fi

echo "Resetting environment"
source "$(dirname ${BASH_SOURCE[0]})/unload_modules.sh"

# echo "Allocating resources with slurm..."
# srun -p gpu --gres=gpu:1 -n 4 --mem=8g -t 1-0 --pty bash

echo "Loading cuda and miniforge..."
module load cuda/12.9 miniforge/25.3.0

echo "Activating conda environment..."
conda activate $SAT_ENV_PATH

echo "Redirecting into the satellite repo..."
cd $SAT_REPO_PATH

echo "Done! Environment has now been set up"
