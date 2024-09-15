import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_data(file_path, input_dim, num_classes):
    # Load dataset
    df = pd.read_csv(file_path)

    # Assuming the last column is the target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode target labels for multi-class classification
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Convert to integer labels

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Ensure the number of features matches input_dim
    if X.shape[1] != input_dim:
        raise ValueError(f"Expected {input_dim} features, but got {X.shape[1]}")

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)  # For multi-class classification

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    return train_loader, test_loader
