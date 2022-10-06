#!/bin/bash                                                                                                                     
#$ -N scc_job_out_n=12

#$ -j yes

#$ -pe openmp 4                                                                                                                
#$ -l h_vmem=1G                                                                                                                 
#$ -l h_rt=40:00:00                                                                                                            
#$ -t 1-384

NPAR=$(echo "print((2*($SGE_TASK_ID-1))%10+40)" | python3 )
python3 ~/HVQE/HVQE.py $PWD $NPAR 6 --n_iter 32
