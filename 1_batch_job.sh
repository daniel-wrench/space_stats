#!/bin/bash
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=parallel
##SBATCH --nodelist=c01n01
##SBATCH --reservation=spacejam
#SBATCH --time=12:00:00
#SBATCH --job-name="1_read_data"
#SBATCH -o 1_singularity_submit.out
#SBATCH -e 1_singularity_submit.err

source ~/bin/python_env
python 1_read_data.py
