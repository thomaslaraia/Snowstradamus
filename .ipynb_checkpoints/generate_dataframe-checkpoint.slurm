#!/bin/bash
#SBATCH --job-name=generate_dataframe    # Job name
#SBATCH --output=generate_dataframe.out  # Output file
#SBATCH --error=generate_dataframe.err   # Error file
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a serial job)
#SBATCH --time=24:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=32G                         # Memory required per node

# Load necessary modules (if any)
# module load python/your_version_here

# Run the Python script
python generate_dataframe.py five_sites_data --width 5 --height 5 --small_box 10 --threshold 1 --alt_thresh 80 --rebinned 30 --method bimodal
