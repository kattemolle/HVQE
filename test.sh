#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:05:00
#SBATCH -p gpu_shared
#SBATCH --mem=1GB
#SBATCH --gres=gpu:1

python3 test_HVQE.py
