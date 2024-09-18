# Federated Learning for Mental Health Prediction

This project implements federated learning to predict mental health conditions based on user input. It leverages federated learning for privacy-preserving model training across distributed datasets. The project provides a web interface where individual users can manually enter data and therapists can upload CSV files with multiple patient data.

## What is Federated Learning?

Federated learning is a machine learning approach where the model is trained across multiple decentralized devices or servers, each holding their own local data samples. Instead of sharing data, only the model updates (gradients) are sent to a central server, ensuring data privacy and security. This approach is especially useful when sensitive data, such as personal health information, cannot be shared across different parties.
*For More Clarification Watch This Youtube Video [What is Federated Learning? by Google Cloud Tech](https://youtu.be/X8YYWunttOY?si=Mf1OgyA0-LUxXaIz)*.
*Want More Resources? Do Your Own Research!*

![fl-graph](https://github.com/user-attachments/assets/2307fcbe-6053-475b-8168-d5968fc279d9)

### How Federated Learning Differs from Traditional Machine Learning:

- **Data Centralization**:
   - *Traditional ML*: Data from various sources is centralized in one location (typically on a server) for training.
   - *Federated Learning*: Data remains decentralized across multiple devices, and only model parameters are shared.
- **Privacy and Security**:
   - *Traditional ML*: All data must be transferred to a single server, which can expose it to privacy risks.
   - *Federated Learning*: Data remains on local devices, reducing the risk of data breaches or leaks.
- **Communication Overhead**:
   - *Traditional ML*: Involves less frequent communication between servers and devices (as data is transferred once for training).
   - *Federated Learning*: Requires constant communication between devices and the server to update the global model, but the size of communication is smaller (only model updates).
- **Scalability**:
   - *Traditional ML*: May face limitations due to data storage and computing power requirements on a central server.
   - *Federated Learning*: Scales well with the number of devices since computations are distributed across local clients.
- **Resource Efficiency**:
   - *Traditional ML*: Relies on centralized infrastructure with high computational power.
   - *Federated Learning*: Utilizes local devices (like smartphones or edge devices) for training, making it more resource-efficient.
 
## Market Value of Federated Learning
The global **Federated Learning market** is expected to grow significantly in the next decade, driven by increased concerns around privacy, the demand for decentralized computing, and growth in edge devices (IoT, mobile devices, etc.).

- **Market Size**: The market was valued at around $94 million in 2021 and is expected to reach $210 million by 2026, growing at a compound annual growth rate (CAGR) of over 17% (estimates vary depending on the source).
- **Key Drivers**: Data privacy regulations (like GDPR), the growing adoption of IoT and edge devices, and industries like healthcare and finance needing secure ways to leverage sensitive data.

## Real-Time Implementations of Federated Learning with Examples

### 1.Google (Gboard)
- **Use Case**: Google was one of the first companies to apply federated learning with Gboard, its on-device keyboard app.
- **Implementation**: Federated learning is used to improve next-word predictions and autocorrect functionalities on users’ keyboards without sending the typed text to Google’s servers. The model is updated based on the users' typing patterns while keeping their data on the device.

### 2.Apple (Siri and iOS)
- **Use Case**: Apple uses federated learning in Siri and other iOS features.
- **Implementation**: It helps improve voice recognition and personalization features while keeping sensitive data, such as voice commands, on the device rather than sending them to Apple's cloud servers.

### 3.NVIDIA (Healthcare)
- **Use Case**: NVIDIA has been working on federated learning in healthcare.
- **Implementation**: In collaboration with King’s College London and Owkin, NVIDIA developed a federated learning system to train models on medical imaging data from multiple hospitals without having to centralize the data. This helps hospitals improve predictive models for disease diagnosis while complying with stringent privacy regulations.

### 4.Intel (Edge Computing and Autonomous Driving)
- **Use Case**: Intel has applied federated learning in edge computing and autonomous driving.
- **Implementation**: In collaboration with BMW and other automotive companies, Intel is developing decentralized models that allow vehicles to improve their driving algorithms by sharing model updates across a network of connected cars, without sharing raw driving data.

## Project Overview
This Project presents a federated learning framework for heart disease prediction using Flower (flwr), where individual users and therapists can participate. Here is an overview and a few suggestions for improvement or clarification:

### Structure Overview:

#### Server (server.py):
- Implements a federated learning server using Flower, with the ```FedAvg``` strategy for aggregation.
- The server receives updated model parameters from clients and distributes the aggregated parameters back to them.

#### Client (client_initial.py & client_user.py):
Two types of clients: individual users and therapists.
- ```client_initial.py``` focuses on initializing the model using a dataset and performs federated learning.
- ```client_user.py``` allows individual users to provide inputs manually for model predictions without explicit training.

#### Model (model.py):
- A neural network model with three fully connected layers and ```ReLU``` activations.
- Uses ```Softmax``` for multi-class classification.

#### Data Loaders (dataloader.py & data_loader_therapy.py):
- ```dataloader.py```: Loads data for training and testing with normalization and label encoding for supervised learning.
- ```data_loader_therapy.py```: Supports both training and inference modes, specifically designed for therapists uploading files.

#### Key Functionality:
- Prediction and Training:
   - For individual users, predictions are made based on their manual input data.
   - Therapists upload datasets and receive an augmented file with predictions.
- Federated Learning Workflow:
   - The server aggregates model updates from clients using the FedAvg strategy.
   - Clients perform local training and send updates to the server.

## Steps to Run This Project

### 1. Clone the Repository
```git clone https://github.com/Arshad-ahmedk/federated_leaarning_implementation-master.git```

### 2. Move to project
```cd federated_leaarning_implementation-master```

### 3. Start the server
```python server.py```

### 4. Run Initial client 
```python client_initial.py --file_path dataset.csv --input_dim 14 --num_classes 4```

*run clien_initial.py 2 times with different dataset to initialise the global model.*

### 4. Run Individual clients
```python client_user.py --input_dim 14 --num_classes 4```

### 5. Run Therapist clients
```python client_therapist.py --file_path test_data.csv --input_dim 14 --num_classes 4 --is_testing```

*Delete Predicted class index colunm if previously exixted in testdataset before executing*





