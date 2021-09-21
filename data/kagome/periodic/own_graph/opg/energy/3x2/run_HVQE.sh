#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 5-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1


for mult in {1..2}
do
python3 ~/HVQE/HVQE.py $PWD 1224 1 &
sleep .1
done
wait

