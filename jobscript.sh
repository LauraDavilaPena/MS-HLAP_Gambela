#!/bin/bash
#SBATCH --job-name=CUAMM
#SBATCH --nodes=2-8
#SBATCH --time=27:00:00
#SBATCH --output=CUAMM%A_%a.out
#SBATCH --error=CUAMM_%A_%a.err

# Your job commands go here
echo "This is job $SLURM_ARRAY_TASK_ID"

# If we had to run the script for different values of the argument, we could use a CSV file to store the arguments
# CSVFILE=jobs.csv

# Activate the conda environment if you are using packages that are not installed on the HPC
# source activate env_name

# Extract the line corresponding to the SLURM_ARRAY_TASK_ID
# args=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CSVFILE")

# Execute the Python script with the extracted arguments
#add "$args"  to the end of the command to pass the argument to the python script
python3 Main.py 
