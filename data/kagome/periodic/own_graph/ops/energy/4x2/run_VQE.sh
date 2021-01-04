#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 90-00:00:00
#SBATCH --mem=120GB

python3 ~/HVQE/HVQE.py $PWD 108 4 --dump_interval 64

