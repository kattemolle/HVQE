for n_par in {0..48..8}
do for i in {1..16}
        do sed -i '$ d' run.sh
            echo "python3 ~/HVQE/HVQE.py $PWD $n_par 1" >> run.sh
            qsub run.sh
            sleep 0.1
        done
    done 
