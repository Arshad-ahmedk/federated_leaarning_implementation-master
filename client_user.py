import flwr as fl
import torch
import numpy as np
import pandas as pd
from model import Model


class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, input_dim, num_classes):
        self.model = Model(input_dim=input_dim, num_classes=num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Training method."""
        self.set_parameters(parameters)
        # Implement local training logic here
        # Return the updated parameters and metrics
        return self.get_parameters(config), 0, {"loss": 0.0}

    def evaluate(self, parameters, config):
        """Evaluation method."""
        self.set_parameters(parameters)
        # Implement evaluation logic here
        # Return evaluation metrics
        return 0.0, 0, {"accuracy": 0.0}

    def predict(self, user_input):
        """Perform prediction on user input."""
        self.model.eval()
        user_input = torch.tensor(user_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(user_input)
        return output.numpy()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Federated Learning Client for Heart Disease Prediction.')
    parser.add_argument('--input_dim', type=int, required=True, help='Dimension of the input features.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes.')

    args = parser.parse_args()

    input_dim = args.input_dim
    num_classes = args.num_classes

    # Create the client instance
    client = HeartDiseaseClient(input_dim, num_classes)

    # Get user input and create DataFrame
    data = {
        "SLEEP": [float(input("ENTER SLEEP: "))],
        "APPETITE": [float(input("ENTER APPETITE: "))],
        "INTEREST": [float(input("ENTER INTEREST: "))],
        "FATIGUE": [float(input("ENTER FATIGUE: "))],
        "WORTHLESS": [float(input("ENTER WORTHLESS: "))],
        "CONCENTRATION": [float(input("ENTER CONCENTRATION: "))],
        "AGITATION": [float(input("ENTER AGITATION: "))],
        "SUICIDE_THOUGHTS": [float(input("ENTER SUICIDE_THOUGHTS: "))],
        "SLEEP_DISTURBANCE": [float(input("ENTER SLEEP_DISTURBANCE: "))],
        "AGRESSION": [float(input("ENTER AGRESSION: "))],
        "PANIC_ATTACK": [float(input("ENTER PANIC_ATTACK: "))],
        "HOPELESSNESS": [float(input("ENTER HOPELESSNESS: "))],
        "RESTLESSNESS": [float(input("ENTER RESTLESSNESS: "))],
        "LOW_ENERGY": [float(input("ENTER LOW_ENERGY: "))]
    }

    df = pd.DataFrame(data)

    # Ensure input dimensions match
    if df.shape[1] != input_dim:
        raise ValueError(f"Expected {input_dim} features, but got {df.shape[1]}")

    # Local training with the input data
    def local_train_and_predict():
        # Simulate local training (you would need actual data for real training)
        # For this example, we are not performing actual training

        # Perform prediction with the current model
        user_input = df.values[0]  # Convert DataFrame to NumPy array
        prediction = client.predict(user_input)
        print(f"Prediction: {prediction}")

        # Get the index of the highest value (predicted class)
        predicted_class_index = np.argmax(prediction)

        # Use the index to determine the corresponding class
        classes = ["No depression", "Mild depression", "Moderate depression", "Severe depression"]
        print(classes[predicted_class_index])

    # Run the function
    local_train_and_predict()


if __name__ == "__main__":
    main()
