#!/bin/bash

#SBATCH --job-name="3b_train"
#SBATCH -o results/3b_train_neural_net_manual.out
#SBATCH -e results/3b_train_neural_net_manual.err
#SBATCH --time=00:30:00
##SBATCH --nodelist=c01n01
##SBATCH --reservation=spacejam
#SBATCH --partition=parallel
#SBATCH --constraint=Intel
#SBATCH --cpus-per-task=5	
#SBATCH --ntasks=1
#SBATCH --mem=20G

module load singularity

#run the container with the runscript defined when we created it
singularity run tensorflow.sif 3b_train_neural_net_manual.py
