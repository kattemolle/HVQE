#!/bin/bash                                                                                                                     
#$ -N n=22_scc_job_out
#$ -j yes
#$ -pe openmp 4                                                                                                               
#$ -l h_vmem=20G                                                                                                              
#$ -l h_rt=40:00:00                                                                                                           
#$ -t 1-32

NPAR=$[14*22]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1



