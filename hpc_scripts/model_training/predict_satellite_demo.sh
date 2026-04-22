#!/bin/bash -l

# predict_satellite_demo.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Generate 16 GeoJSON files for 8 hours of satellite predictions (demo).
# Usage: sbatch predict_satellite_demo.sh <model_type> <weights_path> [start_time]

#SBATCH -J sat_demo
#SBATCH --time=12:00:00
#SBATCH -p preempt
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32g
#SBATCH --output=sat_demo.%j.%N.out
#SBATCH --error=sat_demo.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd $SAT_REPO_PATH
source $SAT_REPO_PATH/hpc_scripts/load_modules.sh
nvidia-smi

model_type=$1
weights=$2
start_time=$3

OUTPUT_DIR=$SAT_REPO_PATH/src/demo_satellite

if [ -n "$start_time" ]; then
    echo "Generating satellite demo GeoJSONs: model=$model_type, start=$start_time"
    python -u $SAT_REPO_PATH/src/predict_satellite_demo.py \
        --model-type $model_type --weights $weights \
        --output-dir $OUTPUT_DIR --start-time "$start_time"
else
    echo "Generating satellite demo GeoJSONs: model=$model_type, start=8hrs ago"
    python -u $SAT_REPO_PATH/src/predict_satellite_demo.py \
        --model-type $model_type --weights $weights \
        --output-dir $OUTPUT_DIR
fi

echo "Python exit code: $?"
echo "Output files:"
ls -la $OUTPUT_DIR/

source $SAT_REPO_PATH/hpc_scripts/unload_modules.sh
echo "All done!"
