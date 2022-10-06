#!/bin/bash                                                                                                                     
#$ -N n=16_scc_job_out
#$ -j yes
#$ -pe openmp 4                                                                                                               
#$ -l h_vmem=2G                                                                                                               
#$ -l h_rt=23:00:00                                                                                                           
#$ -t 1-32

NPAR=$[11*16]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1

NPAR=$[12*16]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1

NPAR=$[13*16]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1

NPAR=$[14*16]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1

NPAR=$[15*16]
python3 ~/HVQE/HVQE.py $PWD $NPAR 1


