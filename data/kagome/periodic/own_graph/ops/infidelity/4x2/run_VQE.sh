#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 90-00:00:00
#SBATCH --mem=120GB

for n_par in {12..100000..12}
do
python3 ~/HVQE/HVQE.py $PWD $n_par 4 --dump_interval 64 --cost_fn infidelity
done
