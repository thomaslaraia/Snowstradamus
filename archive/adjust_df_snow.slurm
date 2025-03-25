#!/bin/bash
#SBATCH --job-name=adjust_df_snow    # Job name
#SBATCH --output=adjust_df_snow.out  # Output file
#SBATCH --error=adjust_df_snow.err   # Error file
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a serial job)
#SBATCH --time=06:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=4G                         # Memory required per node

# Load necessary modules (if any)
# module load python/your_version_here

# Run the Python script
python adjust_df_snow.py five_sites_data_snow --input_pickle five_sites_data
