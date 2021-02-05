#!/bin/bash
#SBATCH -N 1
#SBATCH -n 40
#SBATCH -t 10-00:00:00
#SBATCH --mem=100GB

python3 ~/HVQE/ground_state.py $PWD 2


