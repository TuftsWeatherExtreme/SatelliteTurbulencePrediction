#!/bin/bash -l

# generate_satellite_data_chunk.sh
# Authors: Sophie Clemens
# Spring 2026
# Overview: For each month/year of cleaned PIREPs, fetch GOES-16 satellite images
# (1 frame every 15 minutes for 8 hours, 6 bands) cropped around each PIREP location and save as .npz.
# Then compress into tar.xz archives.
# Note: GOES-16 data only available from March 2017 onward.

#SBATCH -J sat_gen
#SBATCH --time=02-00:00:00
#SBATCH -p batch,preempt
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH --output=sat_output/sat_gen.%j.%a.%N.out
#SBATCH --error=sat_output/sat_gen.%j.%a.%N.err
#SBATCH --array=0-755   # 108 months × 7 chunks
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd $SAT_REPO_PATH
source $SAT_REPO_PATH/hpc_scripts/load_modules.sh

idx=$SLURM_ARRAY_TASK_ID

# GOES-16 available from March 2017 onward
YEARS=("2017" "2018" "2019" "2020" "2021" "2022" "2023" "2024" "2025")
MONTHS=("january" "february" "march" "april" "may" "june" "july" "august" "september" "october" "november" "december")

num_months=${#MONTHS[@]}

CHUNK_SIZE=150
CHUNKS_PER_MONTH=7

chunk_idx=$((idx % CHUNKS_PER_MONTH))
month_idx=$(( (idx / CHUNKS_PER_MONTH) % num_months ))
year_idx=$((idx / CHUNKS_PER_MONTH / num_months))

year=${YEARS[$year_idx]}
month_num=$(printf "%02d" $((month_idx + 1)))

# Skip Jan/Feb 2017 (GOES-16 not yet available)
if [ "$year" == "2017" ] && [ "$month_num" -lt "03" ]; then
    echo "Skipping $year/$month_num (GOES-16 not available)"
    source $SAT_REPO_PATH/hpc_scripts/unload_modules.sh
    exit 0
fi

if [ "$year" == "2025" ] && [ "$month_num" -gt "04" ]; then
    echo "Skipping $year/$month_num (GOES-16 not yet available on AWS)"
    source $SAT_REPO_PATH/hpc_scripts/unload_modules.sh
    exit 0
fi

PIREP_CSV=$SAT_REPO_PATH/pireps/$year/${month_num}_turb_pireps.csv
OUTPUT_DIR=$SAT_REPO_PATH/model_inputs/${year}_${month_num}

if [ ! -f "$PIREP_CSV" ]; then
    echo "No PIREP CSV found at $PIREP_CSV, skipping"
    source $SAT_REPO_PATH/hpc_scripts/unload_modules.sh
    exit 0
fi

mkdir -p $OUTPUT_DIR

CHUNK_START=$((chunk_idx * CHUNK_SIZE))
CHUNK_END=$((CHUNK_START + CHUNK_SIZE))

echo "Processing $year/$month_num chunk [$CHUNK_START:$CHUNK_END]"
echo "Fetching satellite data for PIREPs in $PIREP_CSV"
python3 $SAT_REPO_PATH/src/fetch_satellite_for_pireps.py \
    $PIREP_CSV $OUTPUT_DIR $CHUNK_START $CHUNK_END
echo "Python script exit code: $?"

# source $SAT_REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"