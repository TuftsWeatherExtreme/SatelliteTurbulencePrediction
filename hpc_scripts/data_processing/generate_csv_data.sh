#!/bin/bash -l

# generate_csv_data.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Download and clean PIREP data for binary classification.
# Outputs SEV+ (label=1) and NONE/SMOOTH (label=0) PIREPs with sample weights.
# Note: Only 2017+ PIREPs are useful for satellite (GOES-16 availability).

#SBATCH -J csv_gen
#SBATCH --time=01:00:00
#SBATCH -p batch,preempt
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH --output=csv_gen.%j.%a.%N.out
#SBATCH --error=csv_gen.%j.%a.%N.err
#SBATCH --array=0-107
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd $SAT_REPO_PATH
source $SAT_REPO_PATH/hpc_scripts/load_modules.sh

idx=$SLURM_ARRAY_TASK_ID

YEARS=("2017" "2018" "2019" "2020" "2021" "2022" "2023" "2024" "2025")
MONTHS=("january" "february" "march" "april" "may" "june" "july" "august" "september" "october" "november" "december")

num_months=${#MONTHS[@]}

year_idx=$((idx / num_months))
month_idx=$((idx % num_months))

year=${YEARS[$year_idx]}
month=${MONTHS[$month_idx]}

echo "Processing $month $year"
python $SAT_REPO_PATH/src/clean_pireps.py -month $month -year $year -o FILE

source $SAT_REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"
