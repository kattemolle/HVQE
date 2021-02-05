#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 90-00:00:00
#SBATCH --mem=200GB

for i in {12..1000..12}
do
python3 ~/HVQE/HVQE.py $PWD $i 4
done
