import flwr as fl
import torch
import numpy as np
from model import Model
from dataloader import load_data

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, file_path, input_dim, num_classes):
        self.model = Model(input_dim=input_dim, num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader, self.test_loader = load_data(file_path, input_dim, num_classes)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # Perform one epoch of training
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": loss.item()}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return float(accuracy), len(self.test_loader.dataset), {"accuracy": accuracy}


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Federated Learning Client for Heart Disease Prediction.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--input_dim', type=int, required=True, help='Dimension of the input features.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes.')

    args = parser.parse_args()

    file_path = args.file_path
    input_dim = args.input_dim
    num_classes = args.num_classes

    client = HeartDiseaseClient(file_path, input_dim, num_classes)

    # Start the Flower client and connect to the server
    fl.client.start_numpy_client(
        server_address="localhost:8080",  # Replace with actual server address if needed
        client=client
    )


if __name__ == "__main__":
    main()
