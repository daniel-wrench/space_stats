#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=15G
#SBATCH --partition=parallel
##SBATCH --nodelist=c01n01
##SBATCH --reservation=spacejam
#SBATCH --time=00:05:00
#SBATCH --job-name="1_read"
#SBATCH -o results/1_read_data.out
#SBATCH -e results/1_read_data.err

source ~/bin/python_env
python 1_read_data.py
