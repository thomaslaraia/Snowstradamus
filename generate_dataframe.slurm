#!/bin/bash
#SBATCH --job-name=generate_dataframe    # Job name
#SBATCH --output=generate_dataframe.out  # Output file
#SBATCH --error=generate_dataframe.err   # Error file
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a serial job)
#SBATCH --time=24:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=28G                         # Memory required per node

# Load necessary modules (if any)
# module load python/your_version_here

# Retry logic
RETRIES=1
COUNT=0

while [ $COUNT -lt $RETRIES ]; do
    python generate_dataframe.py dataset --width 4 --height 4 --small_box 8.05 --threshold 1 --alt_thresh 80 --rebinned 30 --method bimodal --site all --outlier_removal .1
    if [ $? -eq 0 ]; then
        break
    fi
    COUNT=$((COUNT+1))
    echo "Retry attempt $COUNT of $RETRIES"
done
