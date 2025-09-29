#!/bin/bash
#SBATCH --job-name=generate_dataframe    # Job name
#SBATCH --output=generate_dataframe.out  # Output file
#SBATCH --error=generate_dataframe.err   # Error file
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a serial job)
#SBATCH --time=24:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=8G                         # Memory required per node

# Load necessary modules (if any)
# module load python/your_version_here

# Retry logic
RETRIES=1
COUNT=0

# Record start time
START_TIME=$(date +%s)

while [ $COUNT -lt $RETRIES ]; do
    python generate_dataframe.py dataset_lcforest_LOF_bin15_th3_80m_1kmsmallbox_noprior_ta_v7 --width 4 --height 4 --small_box 1 --threshold 3 --alt_thresh 80 --rebinned 15 --method normal --site all --outlier_removal 10 --landcover forest --trim_atmospheric 1 --sat_flag 1 --DW 0 #--width 4 --height 4 --small_box 1 --threshold 3 --alt_thresh 80 --rebinned 30 --method normal --site all --outlier_removal 0.22 --landcover forest --trim_atmospheric 1 --sat_flag 1 --DW 1
    if [ $? -eq 0 ]; then
        break
    fi
    COUNT=$((COUNT+1))
    echo "Retry attempt $COUNT of $RETRIES"
done

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Convert to human-readable format
echo "Job completed in $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
