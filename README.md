# Optimizing Neural Network Architectures for Autonomous Vehicle Steering Prediction

This project focuses on optimizing the architecture of convolutional neural networks (CNNs) to predict the steering angle for autonomous vehicles. The optimization is performed using a genetic algorithm that evolves the hyperparameters and structure of the network, with the main goal of reducing the number of parameters in the CNN.

## Components

### Encoding and Decoding
- **encoding.py**: Implements the encoding of network architectures.
- **decoding.py**: Decodes network architectures into PyTorch models.

### Genetic Algorithm
- **operators.py**: Defines genetic operators such as crossover and mutation for evolving the network architectures.

### Training and Evaluation
- **Training.py**: Contains functions for training the CNNs and evaluating their performance using metrics such as MAPE and SMAPE.
- **DeepGA-Steering Angle-Encoding.ipynb**: Integrates all components and runs the genetic algorithm to find the optimal network architecture.

## Training and Evaluation Functions

### entrenamiento(device, cnn, max_epochs, loss_func, optimizer, train_dl)
- Configures the network in training mode and iterates over epochs and batches of data.
- Calculates the network's output, loss function value, and adjusts the gradients.

### validacion(device, cnn, test_dl)
- Configures the network in evaluation mode and iterates over test data batches.
- Evaluates performance metrics SMAPE and MAPE to measure the model's accuracy.

### training(device, model, n_epochs, loss_func, train_dl, test_dl, lr, w, max_params)
- Configures the model and optimizer.
- Trains the model using the entrenamiento function.
- Validates the model using the validacion function.
- Calculates a fitness function based on accuracy and the number of parameters.

### SMAPE and MAPE Functions
- **SMAPE (Symmetric Mean Absolute Percentage Error)** and **MAPE (Mean Absolute Percentage Error)** are implemented to evaluate the performance of the model.

## Modifications
This code was modified from the original project to focus on regression instead of classification. The main goal of these modifications is to reduce the number of parameters in the CNN. The original code can be found at [GustavoVargasHakim/DeepGA](https://github.com/GustavoVargasHakim/DeepGA).

## Publications
- **Publication**: [Reducing Parameters by Neuroevolution in CNN for Steering Angle Estimation](#)
- **Original Article**: [Hybrid encodings for neuroevolution of convolutional neural networks: a case study](#)

