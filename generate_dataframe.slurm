#!/bin/bash
#SBATCH --job-name=generate_dataframe    # Job name
#SBATCH --output=generate_dataframe.out  # Output file
#SBATCH --error=generate_dataframe.err   # Error file
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a serial job)
#SBATCH --time=06:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=4G                         # Memory required per node

# Load necessary modules (if any)
# module load python/your_version_here

# Run the Python script
python generate_dataframe.py five_sites_data --width 0.05 --height 0.05 --small_box 0.005 --threshold 2 --alt_thresh 90
