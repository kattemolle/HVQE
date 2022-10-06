#!/bin/bash                                                                                                                     
#$ -N n=4
#$ -j yes
#$ -pe openmp 4                                                                                                                
#$ -l h_vmem=1G                                                                                                                 
#$ -l h_rt=05:00:00                                                                                                            
#$ -t 1-160

NPAR=$(echo "print((2*($SGE_TASK_ID-1))%10+2)" | python3 )
python3 ~/HVQE/HVQE.py $PWD $NPAR 3 --n_iter 32
