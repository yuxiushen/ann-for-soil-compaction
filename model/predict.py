#!/bin/env python
import argparse

import pandas as pd
import torch
import torch.nn as nn
from ann import FlexibleANN, measure, set_seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json


def print_model(model):
    # Print model parameters nicely
    print("\nModel Parameters:")
    for name, param in model.named_parameters():
        print(f"\n{name} (shape: {tuple(param.shape)}):")
        # Format tensor to 4 decimal places and aligned columns
        array = param.detach().numpy()
        if array.ndim == 2:  # weights
            for row in array:
                formatted_row = "  ".join(f"{v:8.4f}" for v in row)
                print(f"  [{formatted_row}]")
        elif array.ndim == 1:  # biases
            formatted_row = "  ".join(f"{v:8.4f}" for v in array)
            print(f"  [{formatted_row}]")
        else:
            print(array)


def load_and_print_model_parameters(
    model_path, hidden_layers, activation, verbose=False, return_json=False
):
    set_seed(args.seed)

    # Load and preprocess the data
    df = pd.read_csv(args.train_data_path)
    df.set_index("No.", inplace=True)

    df2 = pd.read_csv(args.test_data_path)
    df2.set_index("No.", inplace=True)

    # Separate the features and the targets
    labels = ["CG (%)", "CS (%)", "CF (%)", "LL (%)", "PL (%)", "E (kJ/m3)"]
    X = df[labels]
    X_test = df2[labels]
    if args.property == "wopt":
        y = df["wopt (%)"]
        y_test = df2["wopt (%)"]
    elif args.property == "rhodmax":
        y = df["ρdmax (Mg/m3)"]
        y_test = df2["ρdmax (Mg/m3)"]
    else:
        raise ValueError("Property must be 'wopt' or 'rhodmax'")

    # Bin the target variable for stratification
    y_binned = pd.qcut(
        y, q=10, duplicates="drop"
    )  # You can adjust the number of bins (q) as needed

    # Split the data based on the specified ratio
    if args.train_val_split == 0.5:
        # Use the odd/even split
        X_train = X[df.index % 2 != 0]
        X_val = X[df.index % 2 == 0]
        y_train = y[df.index % 2 != 0]
        y_val = y[df.index % 2 == 0]
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=args.train_val_split,
            random_state=args.seed,
            stratify=y_binned,
        )

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Scale the target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    input_size = X_train_tensor.shape[1]
    output_size = 1

    act_options = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    activation_fn = act_options[args.activation]
    model = FlexibleANN(input_size, hidden_layers, output_size, activation_fn)
    criterion = nn.MSELoss()
    model.load_state_dict(torch.load(model_path))

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_tensor)
        y_val_pred = model(X_val_tensor)
        y_test_pred = model(X_test_tensor)

        train_loss = criterion(y_train_pred, y_train_tensor)
        val_loss = criterion(y_val_pred, y_val_tensor)
        test_loss = criterion(y_test_pred, y_test_tensor)

        # Inverse transform the predictions
        y_train_pred_original = y_scaler.inverse_transform(y_train_pred.numpy())
        y_val_pred_original = y_scaler.inverse_transform(y_val_pred.numpy())
        y_test_pred_original = y_scaler.inverse_transform(y_test_pred.numpy())
        y_train_original = y_scaler.inverse_transform(y_train_tensor.numpy())
        y_val_original = y_scaler.inverse_transform(y_val_tensor.numpy())
        y_test_original = y_scaler.inverse_transform(y_test_tensor.numpy())

        train_r2, train_mae, train_rmse = measure(
            torch.tensor(y_train_original), torch.tensor(y_train_pred_original)
        )
        val_r2, val_mae, val_rmse = measure(
            torch.tensor(y_val_original), torch.tensor(y_val_pred_original)
        )
        test_r2, test_mae, test_rmse = measure(
            torch.tensor(y_test_original), torch.tensor(y_test_pred_original)
        )

    if verbose:
        print(f"[{args.property}]")
        print(
            f"[{args.property}] Train Loss: {train_loss.item():.4f}, "
            f"Val Loss: {val_loss.item():.4f}, Test Loss: {test_loss.item():.4f}"
        )
        print(
            f"[{args.property}] Train R^2 : {train_r2:.4f}, "
            f"Val R^2 : {val_r2:.4f}, Test R^2 : {test_r2:.4f}"
        )
        print(
            f"[{args.property}] Train MAE : {train_mae:.4f}, "
            f"Val MAE : {val_mae:.4f}, Test MAE : {test_mae:.4f}"
        )
        print(
            f"[{args.property}] Train RMSE: {train_rmse:.4f}, "
            f"Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}"
        )

        print_model(model)

    if return_json:
        return {
            "Train R^2": train_r2,
            "Val R^2": val_r2,
            "Test R^2": test_r2,
            "Train MAE": train_mae,
            "Val MAE": val_mae,
            "Test MAE": test_mae,
            "Train RMSE": train_rmse,
            "Val RMSE": val_rmse,
            "Test RMSE": test_rmse,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model and print parameters.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model.")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="train.csv",
        help="Path to the train dataset CSV file.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="test.csv",
        help="Path to the test dataset CSV file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--hidden_layers",
        type=int,
        nargs="+",
        required=True,
        help="List of hidden layer sizes.",
    )
    parser.add_argument(
        "--property",
        type=str,
        choices=["wopt", "rhodmax"],
        required=True,
        help="Property to model: 'wopt' or 'rhodmax'.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "leaky_relu", "tanh", "sigmoid"],
        required=True,
        help="The activation function used.",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.5,
        help="Ratio of validation set size to the entire dataset, e.g., 0.2 for 20%.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more detailed info.",
    )

    parser.add_argument(
        "--return_json",
        action="store_true",
        help="Whether to return JSON info.",
    )

    args = parser.parse_args()

    results = load_and_print_model_parameters(
        args.model_path,
        args.hidden_layers,
        args.activation,
        verbose=args.verbose,
        return_json=args.return_json,
    )

    if results:
        print(json.dumps(results))
