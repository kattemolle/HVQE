#!/bin/bash                                                                                                                     
#$ -N n=4_scc_job_out
#$ -j yes
#$ -pe openmp 4                                                                                                               
#$ -l h_vmem=1G                                                                                                               
#$ -l h_rt=02:00:00                                                                                                           
#$ -t 1-32

NPAR=$[1*4]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1



