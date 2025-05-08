#!/usr/bin/env python

import argparse
import os
import random

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# Define the flexible neural network
class FlexibleANN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=1, activation_fn=nn.ReLU):
        super().__init__()
        layers = []
        in_features = input_size
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(activation_fn())
            in_features = hidden_units
        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Function to calculate R^2 score
def r2_score(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()


# Function to calculate MAE
def mae_score(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae.item()


# Function to calculate RMSE
def rmse_score(y_true, y_pred):
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return rmse.item()


# Function to measure R^2, MAE, and RMSE
def measure(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mae_score(y_true, y_pred)
    rmse = rmse_score(y_true, y_pred)
    return r2, mae, rmse


def objective(
    trial,
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor,
    input_size,
    output_size,
):
    set_seed(args.seed)

    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 1)
    hidden_layers = [trial.suggest_int(f"n_units_l{i}", 1, 10) for i in range(n_layers)]
    activation_name = trial.suggest_categorical(
        "activation", ["tanh", "leaky_relu", "tanh", "sigmoid"]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-3, log=True)

    # Choose activation function
    activation_fn = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }[activation_name]

    # Define the model
    model = FlexibleANN(input_size, hidden_layers, output_size, activation_fn)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    patience = 10  # Patience for early stopping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=patience, factor=0.5
    )

    # Train the model
    epochs = 10000
    best_val_loss = float("inf")
    early_stopping_counter = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor)
            val_loss = criterion(y_val_pred, y_val_tensor)

        # Adjust the learning rate based on validation loss
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            break

    return best_val_loss.item()


def main(args):
    os.makedirs(args.cache_dir, exist_ok=True)
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

    if args.optimize:
        # Optimize hyperparameters with Optuna
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            lambda trial: objective(
                trial,
                X_train_tensor,
                y_train_tensor,
                X_val_tensor,
                y_val_tensor,
                input_size,
                output_size,
            ),
            n_trials=100,
        )

        print(f"Best trial: {study.best_trial.params}")

        # Train the final model with the best hyperparameters
        best_params = study.best_trial.params
        hidden_layers = [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])]
        activation_fn = act_options[best_params["activation"]]
        learning_rate = best_params["learning_rate"]
        exit()
    else:
        # Default hyperparameters if not optimizing
        hidden_layers = args.hidden_layers
        activation_fn = act_options[args.activation]
        learning_rate = args.learning_rate

    model = FlexibleANN(input_size, hidden_layers, output_size, activation_fn)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)

    best_val_loss = float("inf")
    early_stopping_counter = 0
    patience = 10

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
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

                print(f"[{args.property}] Epoch [{epoch+1}/{args.epochs}]")
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

                # Early stopping
                if val_loss < best_val_loss and np.abs(val_loss - best_val_loss) > 1e-4:
                    early_stopping_counter = 0
                    best_val_loss = val_loss

                    # Save the best model
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            args.cache_dir,
                            f"best_model_{args.property}_seed-{args.seed}.pth",
                        ),
                    )
                else:
                    early_stopping_counter += 1

                # Optional: log training
                print(
                    f"Epoch {epoch+1:4d} | Best Val Loss: {best_val_loss.item():.5f} | Val Loss: {val_loss.item():.5f} | Patience: {early_stopping_counter}/{patience}"
                )

                if early_stopping_counter >= patience:
                    print("Early stopping")
                    break

                # Adjust the learning rate based on validation loss
                scheduler.step(val_loss)

            model.train()

    # Save the final model
    os.makedirs(args.cache_dir, exist_ok=True)
    model_path = os.path.join(args.cache_dir, f"model_{args.property}_seed-{args.seed}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model {args.property} saved to {model_path}")

    # Print model parameters
    print("Model Parameters:")
    for name, param in model.named_parameters():
        print(name, param.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a flexible ANN model for regression tasks.")
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
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for training."
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory to save the trained model.",
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
        required=False,
        default="relu",
        help="The activation function",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Flag to turn on hyperparameter optimization.",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        nargs="+",
        default=[64, 8],
        help="List of hidden layer sizes, e.g., --hidden_layers 64 32.",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.5,
        help="Ratio of validation set size to the entire dataset, e.g., 0.2 for 20%.",
    )
    args = parser.parse_args()

    main(args)
