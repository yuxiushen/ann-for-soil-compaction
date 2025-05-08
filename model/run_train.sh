#!/bin/bash

# Common parameters
learning_rate=0.001
epochs=10000
cache_dir="cache"
train_val_split=0.5
activation="tanh"

if [ ! -d "$cache_dir" ]; then
    mkdir -p "$cache_dir"
fi

# Generic function for running ANN
train() {
    local property=$1
    local hidden_layer=$2
    shift 2
    local seeds=("$@")

    for seed in "${seeds[@]}"; do
        echo "===================================="
        echo "Running $property with seed = $seed"
        echo "===================================="
        ./ann.py \
            --learning_rate "$learning_rate" \
            --epochs "$epochs" \
            --seed "$seed" \
            --cache_dir "$cache_dir" \
            --property "$property" \
            --hidden_layers "$hidden_layer" \
            --train_val_split "$train_val_split" \
            --activation "$activation" \
            > "${cache_dir}/${property}_seed_${seed}.log"
    done
}
