#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 90-00:00:00
#SBATCH --mem=120GB

for i in {12..10000..12}
do
python3 ~/HVQE/HVQE.py $PWD $i 4 --dump_interval 64
done
