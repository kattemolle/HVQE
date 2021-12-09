#!/bin/bash
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -t 5-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=2GB
#SBATCH --gres=gpu:1




python3 ~/HVQE_/HVQE.py /home/jorisk2/HVQE_/data/chain/periodic/own_graph/opg/energy/22 154 1
