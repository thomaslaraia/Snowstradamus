#!/bin/bash
#SBATCH --job-name=scfg    # Job name
#SBATCH --output=scfg.out  # Output file
#SBATCH --error=scfg.err   # Error file
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a serial job)
#SBATCH --time=10:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=4G                         # Memory required per node

# Load necessary modules (if any)
# module load python/your_version_here

# Run the Python script
python scfg.py
