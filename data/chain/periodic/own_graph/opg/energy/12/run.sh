#!/bin/bash                                                                                                                     
#$ -N n=12_scc_job_out
#$ -j yes
#$ -pe openmp 4                                                                                                               
#$ -l h_vmem=1G                                                                                                               
#$ -l h_rt=23:00:00                                                                                                           
#$ -t 1-32

NPAR=$[6*12]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1

NPAR=$[7*12]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1

NPAR=$[8*12]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1

NPAR=$[9*12]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1

NPAR=$[10*12]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1


