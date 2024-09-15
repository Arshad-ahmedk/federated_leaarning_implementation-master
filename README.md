# Federated Learning for Mental Health Prediction

This project implements federated learning to predict mental health conditions based on user input. It leverages federated learning for privacy-preserving model training across distributed datasets. The project provides a web interface where individual users can manually enter data and therapists can upload CSV files with multiple patient data.

## What is Federated Learning?

Federated learning is a machine learning approach where the model is trained across multiple decentralized devices or servers, each holding their own local data samples. Instead of sharing data, only the model updates (gradients) are sent to a central server, ensuring data privacy and security. This approach is especially useful when sensitive data, such as personal health information, cannot be shared across different parties.

### Key Differences Between Federated Learning and Traditional Machine Learning:

- **Data Privacy**: Federated learning keeps data local on users' devices, while traditional machine learning typically requires centralized data collection.
- **Decentralization**: Federated learning enables model training on decentralized data across multiple locations, while traditional machine learning relies on data stored in a single, central location.
- **Security**: Since data is not transferred in federated learning, the risk of data breaches is reduced. Traditional machine learning exposes data to higher security risks through data transfer and central storage.
- **Performance**: Federated learning can handle data heterogeneity (i.e., non-iid data) across different devices, whereas traditional machine learning requires homogenized data for better performance.
- **Security**:
   - *Federated Learning*: Only model updates are shared; raw data is not exposed.
   - *Traditional ML*: Data transfer to central storage increases security risks.

## Project Overview
This Project presents a federated learning framework for heart disease prediction using Flower (flwr), where individual users and therapists can participate. Here is an overview and a few suggestions for improvement or clarification:

Structure Overview:
Server (server.py):

Implements a federated learning server using Flower, with the FedAvg strategy for aggregation.
The server receives updated model parameters from clients and distributes the aggregated parameters back to them.
Client (client_initial.py & client_user.py):

Two types of clients: individual users and therapists.
client_initial.py focuses on initializing the model using a dataset and performs federated learning.
client_user.py allows individual users to provide inputs manually for model predictions without explicit training.
Model (model.py):

A neural network model with three fully connected layers and ReLU activations.
Uses Softmax for multi-class classification.
Data Loaders (dataloader.py & data_loader_therapy.py):

dataloader.py: Loads data for training and testing with normalization and label encoding for supervised learning.
data_loader_therapy.py: Supports both training and inference modes, specifically designed for therapists uploading files.
Key Functionality:
Prediction and Training:
For individual users, predictions are made based on their manual input data.
Therapists upload datasets and receive an augmented file with predictions.
Federated Learning Workflow:
The server aggregates model updates from clients using the FedAvg strategy.
Clients perform local training and send updates to the server.



## Steps to Run This Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/FederatedLearningProject.git
cd FederatedLearningProject
   

