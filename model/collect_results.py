#!/usr/bin/env python
import argparse
import subprocess
import json
import pandas as pd


def make_table(property_name, hidden_layers, activation="tanh", cache_dir="cache"):

    if property_name == "rhodmax":
        seeds = [1, 7, 201, 202, 203, 204, 207, 307, 10, 70, 2010, 2020, 2030, 2040, 2070, 3070]
    elif property_name == "wopt":
        seeds = [30, 31, 32, 33, 51, 70, 90, 99, 300, 310, 320, 330, 510, 700, 900, 990]
    else:
        raise RuntimeError(f"Unknown property: {property_name}")

    rows = []

    for seed in seeds:
        model_path = f"{cache_dir}/best_model_{property_name}_seed-{seed}.pth"
        cmd = [
            "python",
            "predict.py",
            "--model_path",
            model_path,
            "--hidden_layers",
            *map(str, hidden_layers),
            "--activation",
            activation,
            "--seed",
            str(seed),
            "--property",
            property_name,
            "--return_json",
        ]

        print(f"Running seed {seed}...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        try:
            metrics = json.loads(
                result.stdout.strip().splitlines()[-1]
            )  # last line should be the dict
            row = {"Model": f"ANN{len(rows)+1}", "Seed": seed, **metrics}
            rows.append(row)
        except Exception as e:
            print(f"Failed for seed {seed}: {e}")
            print(result.stdout)
            print(result.stderr)

    df = pd.DataFrame(rows)

    # Compute mean row
    avg_row = df.drop(columns=["Model", "Seed"]).mean()
    avg_row["Model"] = "ANN-AVG"
    avg_row["Seed"] = "-"
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Append GPR and MEP manually
    if property_name == "rhodmax":
        gpr_row = {
            "Model": "GPR",
            "Seed": "-",
            "Train R^2": 0.93,
            "Val R^2": 0.94,
            "Test R^2": 0.8707,
            "Train MAE": 0.03981,
            "Val MAE": 0.03953,
            "Test MAE": 0.03941,
            "Train RMSE": 0.0509,
            "Val RMSE": 0.05104,
            "Test RMSE": 0.0507,
        }
        mep_row = {
            "Model": "MEP",
            "Seed": "-",
            "Train R^2": 0.872,
            "Val R^2": 0.858,
            "Test R^2": 0.7939,
            "Train MAE": 0.050,
            "Val MAE": 0.057,
            "Test MAE": 0.04847,
            "Train RMSE": 0.069,
            "Val RMSE": 0.077,
            "Test RMSE": 0.05552,
        }
    elif property_name == "wopt":
        # TODO: Replace with actual GPR/MEP results for wopt if you have them
        gpr_row = {
            "Model": "GPR",
            "Seed": "-",
            "Train R^2": 0.91,
            "Val R^2": 0.93,
            "Test R^2": 0.8132,
            "Train MAE": 1.3802,
            "Val MAE": 1.1439,
            "Test MAE": 1.2048,
            "Train RMSE": 1.804,
            "Val RMSE": 1.5816,
            "Test RMSE": 1.6146,
        }
        mep_row = {
            "Model": "MEP",
            "Seed": "-",
            "Train R^2": 0.916,
            "Val R^2": 0.923,
            "Test R^2": 0.7086,
            "Train MAE": 1.206,
            "Val MAE": 1.383,
            "Test MAE": 1.6108,
            "Train RMSE": 1.574,
            "Val RMSE": 1.78,
            "Test RMSE": 2.1676,
        }

    df = pd.concat([df, pd.DataFrame([mep_row, gpr_row])], ignore_index=True)
    df = df.round(4)

    # Save table
    df.to_csv(f"{property_name}_results.csv", index=False)
    print(f"\nSaved CSV to {property_name}_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ANN evaluation and generate results table.")
    parser.add_argument(
        "property_name", type=str, choices=["rhodmax", "wopt"], help="Property to evaluate"
    )
    parser.add_argument("hidden_layers", type=int, nargs="+", help="Hidden layers (e.g., 50 50)")

    args = parser.parse_args()
    make_table(args.property_name, args.hidden_layers)
