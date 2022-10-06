#!/bin/bash                                                                                                                     
#$ -N n=8_scc_job_out
#$ -j yes
#$ -pe openmp 4                                                                                                               
#$ -l h_vmem=1G                                                                                                               
#$ -l h_rt=02:00:00                                                                                                           
#$ -t 1-32

NPAR=$[5*8]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1



