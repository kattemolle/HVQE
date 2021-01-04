#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 1-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=50MB
##SBATCH --gres=gpu:1

for npar in {24..240..12}
do
python3 ~/HVQE/HVQE.py $PWD $npar 2 --cost_fn infidelity
done
