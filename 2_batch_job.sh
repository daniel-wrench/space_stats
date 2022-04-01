#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=parallel
##SBATCH --nodelist=c01n01
##SBATCH --reservation=spacejam
#SBATCH --time=12:00:00
#SBATCH --job-name="2_process_data"
#SBATCH -o results/2_process_data.out
#SBATCH -e results/2_process_data.err

source ~/bin/python_env
python 2_process_data.py
