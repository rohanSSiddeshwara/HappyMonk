# HappyMonk
This repository contains the solution to an AI internship assignment. It includes code and necessary files for analysing and modeling a given dataset. The goal is to generate insights and predictions using machine learning techniques.

## Introduction:

In this report, we will discuss the implementation details of a neural network using an Ada-Act activation function for training. The Ada-Act activation function is defined as k0 + k1z + k2z^2, where z is the input and k0, k1, k2 are coefficients that are learned during training. The network architecture consists of three layers: an input layer, two hidden layers, and an output layer. The input and hidden layers use the Ada-Act activation function, while the output layer uses the softmax function. The network is trained using the categorical cross-entropy loss function.

## Algorithm:

Initialize the network parameters: weights and biases for the three layers (w1, b1, w2, b2, w3, b3) and coefficients for the Ada-Act activation function (k). The initial values for the weights and biases can be sampled from a random distribution, such as a normal distribution with mean 0 and standard deviation 1. The initial values for the Ada-Act coefficients can be set to some small constant, such as 0.01.

. `Preprocess the data`: split the dataset into training and test sets, and standardize the features.

. `Perform forward propagation`: compute the activation values for each layer using the input data and the current network parameters.

. `Compute the loss`: use the output activation values and the true labels to compute the categorical cross-entropy loss.

. `Perform backpropagation`: compute the gradients for the weights, biases, and Ada-Act coefficients using the output error and the activations from the hidden layers.

. `Update the network parameters`: use the computed gradients to update the weights, biases, and Ada-Act coefficients using some learning rate alpha.

. Repeat steps 3-6 for a specified number of epochs.

.`Evaluate the model`: use the test set to evaluate the model's performance by computing the accuracy, confusion matrix, and classification report.

## Implementation Details:

The code provided includes functions for each of the steps in the above algorithm. The function "g" computes the activation values for the Ada-Act activation function, and the function "g_derivative" computes the derivative of the activation function. The function "softmax" computes the activation values for the output layer using the softmax function. The function "CategoricalCrossEntropyLoss" computes the categorical cross-entropy loss between the true labels and the predicted probabilities. The function "backprop" computes the gradients for the weights, biases, and Ada-Act coefficients using the output error and the activations from the hidden layers. The function "update_parameters" updates the network parameters using the computed gradients and the learning rate. The function "forward_prop" performs a single forward propagation step, and the function "fit" trains the model by repeatedly calling the forward_prop and backprop functions. The function "predict" generates predictions using the trained model.

## Code Description:

. `g()`: This function defines the Ada-Act activation function, which has the form k0 + k1 * x + k2 * x^2. It takes in an input z and a matrix of coefficients k, and returns the output of the activation function.

. `g_derivative()`: This function defines the derivative of the Ada-Act activation function. It takes in an input z and a matrix of coefficients k, and returns the derivative of the activation function.

. `softmax()`: This function defines the Softmax activation function, which is often used in the output layer of an ANN for classification tasks. It takes in an input x and returns the Softmax activation of x.

. `CategoricalCrossEntropyLoss()`: This function defines the Categorical Cross-Entropy Loss, which is a commonly used loss function for classification tasks. It takes in the true labels y and the predicted labels y_pred, and returns the loss.

. `backprop()`: This function performs the backpropagation algorithm, which is used to calculate the gradients of the loss function with respect to the weights and biases in the ANN. It takes in the true labels y, the activations of each layer a0, a1, a2, a3, and the weights and biases of each layer w1, w2, w3, b1, b2, b3, and the matrix of coefficients for the Ada-Act activation function k, and returns the gradients for each parameter.

. `update_parameters()`: This function updates the weights and biases in the ANN using the calculated gradients and a learning rate alpha. It takes in the weights and biases of each layer w1, w2, w3, b1, b2, b3, the matrix of coefficients for the Ada-Act activation function k, and the gradients for each parameter dw3, db3, dw2, db2, dw1, db1, dK, and the learning rate alpha, and updates the parameters accordingly.

. `forward_prop()`: This function performs the forward propagation algorithm, which is used to make predictions using the ANN. It takes in the activations of the input layer a0, the weights and biases of each layer w1, w2, w3, b1, b2, b3, and the matrix of coefficients for the Ada-Act activation function k, and returns the activations of each layer and the predicted labels.

. `fit()`: This is the main function for training the ANN. It takes in the input data X, the true labels y, the number of epochs to train for epochs, the batch size batch_size, and the learning rate alpha, and returns the trained weights and biases of each layer and the matrix of coefficients for the Ada-Act activation function.
We use some libraries, such as NumPy, TensorFlow, and scikit-learn, and define some constants such as the number of nodes in each layer and the number of classes in the classification task.

## Conclusion:

In this report, we have described the implementation details of a neural network using an Ada-Act activation function for training. The network architecture consists of three layers and is trained using the categorical cross-entropy loss function. The provided code includes functions for each step in the training process, including forward propagation, backpropagation, and parameter updates.
