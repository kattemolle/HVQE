#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -t 5-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=15GB
#About 1GB per job
#SBATCH --gres=gpu:1

for mult in {1..5}
do
python3 ~/HVQE/HVQE.py $PWD 864 1 &
sleep .1
done
wait

