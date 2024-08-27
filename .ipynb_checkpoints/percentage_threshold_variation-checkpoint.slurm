#!/bin/bash
#SBATCH --job-name=percentage_threshold_variation    # Job name
#SBATCH --output=percentage_threshold_variation.out  # Output file
#SBATCH --error=percentage_threshold_variation.err   # Error file
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a serial job)
#SBATCH --time=5:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=4G                         # Memory required per node

# Load necessary modules (if any)
# module load python/your_version_here

# Run the Python script
python percentage_threshold_variation.py
