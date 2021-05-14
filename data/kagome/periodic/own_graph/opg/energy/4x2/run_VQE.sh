#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -t 5-00:00:00
#SBATCH --mem=150GB
#SBATCH -p fat_soil_shared

python3 ~/HVQE/HVQE.py $PWD 192 1
