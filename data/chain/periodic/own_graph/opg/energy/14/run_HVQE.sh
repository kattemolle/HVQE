#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -t 1-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=3GB
#SBATCH --gres=gpu:1

python3 ~/HVQE_/HVQE.py $PWD 70 1 &
python3 ~/HVQE_/HVQE.py $PWD 84 1 &
python3 ~/HVQE_/HVQE.py $PWD 98 1 &

