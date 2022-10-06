#!/bin/bash                                                                                                                     
#$ -N n=10_scc_job_out
#$ -j yes
#$ -pe openmp 4                                                                                                               
#$ -l h_vmem=1G                                                                                                               
#$ -l h_rt=20:00:00                                                                                                           
#$ -t 1-32

NPAR=$[6*10]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1

NPAR=$[7*10]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1

NPAR=$[8*10]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1



