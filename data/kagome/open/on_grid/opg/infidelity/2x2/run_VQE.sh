#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 2-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=100MB
#SBATCH --gres=gpu:1

for npar in {60..1000..60}
do
python3 ~/HVQE/HVQE.py $PWD $npar 1 --cost_fn infidelity
done
