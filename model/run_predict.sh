#!/bin/bash

# Common parameters
activation="tanh"
cache_dir="cache"

if [ ! -d "$cache_dir" ]; then
    mkdir -p "$cache_dir"
fi

# Generic function for running predict.py
predict() {
    local property=$1
    local hidden_layer=$2
    shift 2
    local seeds=("$@")

    for seed in "${seeds[@]}"; do
        echo "===================================="
        echo "Running $property with seed = $seed"
        echo "===================================="
        python predict.py \
            --model_path "${cache_dir}/best_model_${property}_seed-${seed}.pth" \
            --hidden_layers "$hidden_layer" \
            --activation "$activation" \
            --seed "$seed" \
            --property "$property" \
            --verbose
        echo ""
    done
}
