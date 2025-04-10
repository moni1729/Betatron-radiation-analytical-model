#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --job-name=quickpic
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

mpirun -np 2 model1 input
