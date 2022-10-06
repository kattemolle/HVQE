#!/bin/bash                                                                                                                     
#$ -N HVQEn24n_par168                                                                                                                     
#$ -pe openmp 4                                                                                                                
#$ -l h_vmem=25G                                                                                                                 
#$ -l h_rt=440:00:00                                                                                                            
#$ -l m1000=1




python3 ~/HVQE/HVQE.py /home/watt/joris.kattemoelle/HVQE/data/chain/periodic/own_graph/opg/energy/24 168 1
