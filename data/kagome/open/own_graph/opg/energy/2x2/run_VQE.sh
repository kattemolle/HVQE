#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 1-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=100MB
#SBATCH --gres=gpu:1

for npar in {30..1000..30}
do
python3 ~/HVQE/HVQE.py $PWD $npar 1 
done
