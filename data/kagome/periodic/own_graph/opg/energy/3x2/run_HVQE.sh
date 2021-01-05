#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 5-00:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=1GB
#SBATCH --gres=gpu:1

for n_par in {468..468}
do
	 python3 ~/HVQE/HVQE.py $PWD $n_par 1  
done
