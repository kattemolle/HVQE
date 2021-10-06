#!/bin/bash
#SBATCH -N 1
#SBATCH -n 15
#SBATCH -t 5-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=10GB
#About 1GB per job
#SBATCH --gres=gpu:1

for mult in {1..10}
do
python3 ~/HVQE/HVQE.py $PWD 216 1 &
sleep .1
done
wait

