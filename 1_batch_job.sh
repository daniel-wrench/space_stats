#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=parallel
##SBATCH --nodelist=c01n01
##SBATCH --reservation=spacejam
#SBATCH --time=00:05:00
#SBATCH --job-name="1_read_data"
#SBATCH -o results/1_read_data.out
#SBATCH -e results/1_read_data.err

source ~/bin/python_env
python 1_read_data.py
