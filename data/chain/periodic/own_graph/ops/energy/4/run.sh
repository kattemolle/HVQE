#!/bin/bash                                                                                                                     
#$ -N n=4
#$ -j yes
#$ -pe openmp 4                                                                                                                
#$ -l h_vmem=1G                                                                                                                 
#$ -l h_rt=01:00:00                                                                                                            
#$ -t 1-64

NPAR=$(echo "print((2*($SGE_TASK_ID-1))%4+2)" | python3 )
python3 ~/HVQE/HVQE.py $PWD $NPAR 2 --n_iter 32
