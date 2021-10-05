#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:01:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=10GB


python3 ~/HVQE/qem.py


#for mult in {1..10}
#do
#python3 ~/HVQE/HVQE.py $PWD 1332 1 --cost_fn infidelity&
#sleep .1
#done
#wait

