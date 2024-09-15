import flwr as fl
import torch
from model import Model
from flwr.server import ServerConfig


class HeartDiseaseServer(fl.server.Server):
    def __init__(self, client_manager):
        super().__init__(client_manager=client_manager)
        self.model = None
        self.model_params = None
        self.input_dim = None

    def set_parameters(self, parameters):
        if self.model is None:
            raise ValueError("Model is not initialized.")

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        self.model_params = parameters

    def get_parameters(self):
        if self.model_params is None:
            raise ValueError("Model parameters are not initialized.")
        return self.model_params

    def update_model_input_dim(self, input_dim):
        """Update model with the received input dimension from the client."""
        self.input_dim = input_dim
        self.model = Model(input_dim=input_dim)  # Initialize model with the received input dimension
        self.model_params = [val.cpu().numpy() for val in
                             self.model.state_dict().values()]  # Initialize model parameters


def main():
    # Initialize the server
    client_manager = fl.server.SimpleClientManager()
    server = HeartDiseaseServer(client_manager=client_manager)

    # Define the federated learning strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    # Define the server configuration
    config = ServerConfig(
        num_rounds=5,  # Set the number of rounds here
        round_timeout=None,
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        server=server,
        strategy=strategy,
        config=config,
    )
    try:
        while True:
                fl.server.start_server(
                server_address="localhost:8080",
                server=server,
                strategy=strategy,
                config=config,
                )

                time.sleep(10)  # Sleep to keep the server running
    except KeyboardInterrupt:
        print("Server is stopping...")


if __name__ == "__main__":
    main()
