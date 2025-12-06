#!/bin/bash
#SBATCH --job-name=train-models
#SBATCH --time=2-00:00:00
#SBATCH -o job-%J.txt
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=knagalla@stanford.edu
#SBATCH --cpus-per-task=8

eval "$(conda shell.bash hook)"
source activate tf_env
python define_model.py