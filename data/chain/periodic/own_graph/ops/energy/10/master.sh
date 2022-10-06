for n_par in {0..20..2}
do for i in {1..32}
        do sed -i '$ d' run.sh
            echo "python3 ~/HVQE/HVQE.py $PWD $n_par 5" >> run.sh
            qsub run.sh
            sleep 0.1
        done
    done 
