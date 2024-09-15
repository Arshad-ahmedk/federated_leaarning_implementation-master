import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(file_path, input_dim, num_classes, is_testing=False):
    df = pd.read_csv(file_path)

    if is_testing:
        # For testing dataset, no labels are provided
        if df.shape[1] != input_dim:
            raise ValueError(f"Expected {input_dim} features, but got {df.shape[1]}")
        X = df.values  # Only features, no labels
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        return dataloader
    else:
        # For training dataset, features and labels are provided
        if df.shape[1] != input_dim + 1:
            raise ValueError(f"Expected {input_dim + 1} columns, but got {df.shape[1]}")
        X = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Labels

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        return dataloader
