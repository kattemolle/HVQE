#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 5-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

python3 ~/HVQE/HVQE.py $PWD 480 1 --cost_fn infidelity

