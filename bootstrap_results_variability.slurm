#!/bin/bash
#SBATCH --job-name=snow_bootstrap
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=bootstrap_results_w_variability.out
#SBATCH --error=bootstrap_results_w_variability.err

python -u bootstrap_results_w_variability.py \
    -E 80 \
    -aspect_fraction 0.75 \
    -gradient_p84_max 4.0 \
    -n_boot_30m 1000
