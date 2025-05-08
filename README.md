# Code for **"Artificial Neural Networks for Soil Compaction Prediction: Accuracy, Variability, and Generalization"**

## Installation

First, install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Create a new directory and navigate into it:

```bash
mkdir model-2
cd model-2
```

Then, create a bash script called `run.sh` with the following content. Modify the `nb_hidden` variable to set the number of hidden layer nodes (e.g., `2`):

```bash
#!/bin/bash

# List of required files to link
files=(
    ann.py
    collect_results.py
    predict.py
    run_predict.sh
    run_train.sh
    test.csv
    train.csv
)

# Create symbolic links to the files from ../model
for f in "${files[@]}"; do
    if [ -L "$f" ]; then
        rm "$f"
    fi
    ln -s "../model/$f" .
done

# Source training script
source run_train.sh

# Define seed lists
seeds_wopt=(30 31 32 33 51 70 90 99 300 310 320 330 510 700 900 990)
seeds_rhodmax=(1 7 201 202 203 204 207 307 10 70 2010 2020 2030 2040 2070 3070)

# Set number of hidden layer nodes
nb_hidden=2

# Run training
train wopt $nb_hidden "${seeds_wopt[@]}"
train rhodmax $nb_hidden "${seeds_rhodmax[@]}"

# Source prediction script
source run_predict.sh

# Run prediction and save output logs
predict wopt $nb_hidden "${seeds_wopt[@]}" > wopt.out 2>&1
predict rhodmax $nb_hidden "${seeds_rhodmax[@]}" > rhodmax.out 2>&1

# Collect results
./collect_results.py rhodmax $nb_hidden
./collect_results.py wopt $nb_hidden
```

Finally, make the script executable and run it:

```bash
chmod +x run.sh
./run.sh
```
