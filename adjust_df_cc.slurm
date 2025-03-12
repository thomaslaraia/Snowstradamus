#!/bin/bash
#SBATCH --job-name=adjust_df_scc    # Job name
#SBATCH --output=adjust_df_cc.out  # Output file
#SBATCH --error=adjust_df_cc.err   # Error file
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a serial job)
#SBATCH --time=10:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=16G                         # Memory required per node

# Load necessary modules (if any)
# module load python/your_version_here

# Run the Python script
python adjust_df_cc.py --buffer 1000
