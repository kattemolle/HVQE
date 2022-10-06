#!/bin/bash                                                                                                                   
#$ -N noisy_scc_job_out
#$ -j yes
#$ -pe openmp 4                                                                                                               
#$ -l h_vmem=2G                                                                                                               
#$ -l h_rt=72:00:00                                                                                                           
#$ -t 1-161

python3 ~/HVQE/noise.py bitflip 0.00001 $[$SGE_TASK_ID-1] $[2**10]
