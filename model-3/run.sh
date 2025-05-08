#!/bin/bash

files=(
    ann.py
    collect_results.py
    predict.py
    run_predict.sh
    run_train.sh
    test.csv
    train.csv
)

for f in "${files[@]}"; do
    if [ -L "$f" ]; then
        rm "$f"
    fi
    ln -s "../model/$f" .
done

source run_train.sh

# Define seed lists
seeds_wopt=(30 31 32 33 51 70 90 99 300 310 320 330 510 700 900 990)
seeds_rhodmax=(1 7 201 202 203 204 207 307 10 70 2010 2020 2030 2040 2070 3070)

nb_hidden=3

# Run both experiments
train wopt $nb_hidden "${seeds_wopt[@]}"
train rhodmax $nb_hidden "${seeds_rhodmax[@]}"

source run_predict.sh

predict wopt $nb_hidden "${seeds_wopt[@]}" > wopt.out 2>&1
predict rhodmax $nb_hidden "${seeds_rhodmax[@]}" > rhodmax.out 2>&1

./collect_results.py rhodmax $nb_hidden
./collect_results.py wopt $nb_hidden
