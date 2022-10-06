for n_par in {60..100..10}
do for i in {1..32}
        do sed -i '$ d' run.sh
            echo "python3 ~/HVQE/HVQE.py $PWD $n_par 1" >> run.sh
            qsub run.sh
            sleep 0.1
        done
    done 
