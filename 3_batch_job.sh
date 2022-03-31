#!/bin/bash

#SBATCH --job-name=tf_ann
#SBATCH -o 3_train_neural_net.out
#SBATCH -e 3_train_neural_net.err
#SBATCH --time=00:30:00
#SBATCH --partition=parallel
#SBATCH --constraint=Intel
#SBATCH --cpus-per-task=2	
#SBATCH --ntasks=1
#SBATCH --mem=8G

module load singularity

#run the container with the runscript defined when we created it
singularity run tensorflow.sif 3_train_neural_net.py
