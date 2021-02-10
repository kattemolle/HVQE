#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 5-00:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=1GB
#SBATCH --gres=gpu:1


python3 ~/HVQE/HVQE.py $PWD 180 1
