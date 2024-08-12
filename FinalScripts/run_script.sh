#!/bin/bash
#SBATCH -n 64                  # Cores
#SBATCH --mem=256000           # Memory
#SBATCH -t 05:00:00             # Max runtime (format: hh:mm:ss)
#SBATCH -J full_data_callanv   # Job name

# Loading Python module
module load python/3.11.6-gcc-13.1.0-rri7oiq

# Activating my virtual environment
source /home/users/cids/myenv/bin/activate

# Running my script
python /home/users/cids/bow_train_explore_call.py
