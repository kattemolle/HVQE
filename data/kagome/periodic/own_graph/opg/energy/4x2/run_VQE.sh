#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 90-00:00:00
#SBATCH --mem=150GB

for n_par in {48..100000..48}
do
python3 ~/HVQE/HVQE.py $PWD $n_par 1 --dump_interval 64
done
