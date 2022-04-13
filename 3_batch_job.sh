#!/bin/bash

#SBATCH --job-name="3_train"
#SBATCH -o results/3_train_neural_net.out
#SBATCH -e results/3_train_neural_net.err
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
singularity run tensorflow.sif 3_train_neural_net.py
