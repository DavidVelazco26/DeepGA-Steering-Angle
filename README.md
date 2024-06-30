#  Neural Network Architectures for Autonomous Vehicle Steering Prediction

## Project Description

This project focuses on optimizing the architecture of convolutional neural networks (CNNs) to predict the steering angle for autonomous vehicles. The optimization is performed using a genetic algorithm that evolves the hyperparameters and structure of the network, with the main goal of reducing the number of parameters in the CNN.

## Components

### Encoding and Decoding

#### `encoding.py`
Implements the encoding of network architectures.

#### `decoding.py`
Decodes network architectures into PyTorch models.

### Genetic Algorithm

#### `operators.py`
Defines genetic operators such as crossover and mutation for evolving the network architectures.

### Training and Evaluation

#### `Training.py`
Contains functions for training the CNNs and evaluating their performance using metrics such as MAPE and SMAPE.

#### `DeepGA-Steering Angle.ipynb`
Integrates all components and runs the genetic algorithm to find the optimal network architecture.

## Training and Evaluation Functions

### `entrenamiento(device, cnn, max_epochs, loss_func, optimizer, train_dl)`
Configures the network in training mode and iterates over epochs and batches of data.
- Calculates the network's output, loss function value, and adjusts the gradients.

### `validacion(device, cnn, test_dl)`
Configures the network in evaluation mode and iterates over test data batches.
- Evaluates performance metrics SMAPE and MAPE to measure the model's accuracy.

### `training(device, model, n_epochs, loss_func, train_dl, test_dl, lr, w, max_params)`
Configures the model and optimizer.
- Trains the model using the `entrenamiento` function.
- Validates the model using the `validacion` function.
- Calculates a fitness function based on accuracy and the number of parameters.

## Performance Metrics

### Mean Absolute Percentage Error (MAPE)
The MAPE metric measures the accuracy of a predictive model by calculating the percentage difference between the predicted and actual values. It is defined as:

$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100$

where:
- $y_i$ is the actual value,
-  $\hat{y}_i$ is the predicted value,
-  $n$ is the number of data points.

MAPE provides an indication of how much, on average, the predictions deviate from the actual values in percentage terms.

### Symmetric Mean Absolute Percentage Error (SMAPE)
The SMAPE metric is a variation of MAPE that addresses the issue of asymmetry in MAPE. It is defined as:

$\text{SMAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{\frac{|y_i| + |\hat{y}_i|}{2}} \times 100$

SMAPE normalizes the absolute error by the average of the actual and predicted values, making it symmetric and providing a balanced perspective on the prediction accuracy.

## Modifications
This code was modified from the original project to focus on regression instead of classification. The main goal of these modifications is to reduce the number of parameters in the CNN. The original code can be found at [GustavoVargasHakim/DeepGA](https://github.com/GustavoVargasHakim/DeepGA).

There is a publication about this approach:

**Publication**: [Reducing Parameters by Neuroevolution in CNN for Steering Angle Estimation](https://doi.org/10.1007/978-3-031-62836-8_35)

The original article of DeepGA is also published:

**Publication**: [Hybrid encodings for neuroevolution of convolutional neural networks: a case study](https://dl.acm.org/doi/abs/10.1145/3449726.3463133)


