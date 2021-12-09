for i in {1..32}
    do for n_par in $(seq 0 28 308)
       do sed -i '$ d' run_HVQE.sh
          echo "python3 ~/HVQE_/HVQE.py $PWD $n_par 1" >> run_HVQE.sh
          sbatch run_HVQE.sh
          sleep 0.1
       done
    done
done     
