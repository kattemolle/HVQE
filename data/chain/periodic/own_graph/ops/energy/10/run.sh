#!/bin/bash                                                                                                                     
#$ -N n=10
#$ -j yes
#$ -pe openmp 4                                                                                                                
#$ -l h_vmem=1G                                                                                                                 
#$ -l h_rt=20:00:00                                                                                                            
#$ -t 1-256

NPAR=$(echo "print((2*($SGE_TASK_ID-1))%10+2+20)" | python3 )
python3 ~/HVQE/HVQE.py $PWD $NPAR 5 --n_iter 32
