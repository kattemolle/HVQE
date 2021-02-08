#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=1GB
#SBATCH --gres=gpu:1

for n_par in {0..30..2}
do
	 python3 ~/HVQE/HVQE.py $PWD $n_par 9  
done
