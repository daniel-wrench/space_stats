#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=parallel
##SBATCH --nodelist=c01n01
##SBATCH --reservation=spacejam
#SBATCH --time=00:30:00
#SBATCH --job-name="check_missingness"
#SBATCH -o results/check_missingness.out
#SBATCH -e results/check_missingness.err

source ~/bin/python_env
python check_missingness.py
