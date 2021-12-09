#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 1-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=500MB
##SBATCH --gres=gpu:1




python3 ~/HVQE_/HVQE.py $PWD 0 1
