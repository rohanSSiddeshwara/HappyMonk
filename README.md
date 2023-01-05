# HappyMonk
This repository contains the solution to an AI internship assignment. It includes code and necessary files for analysing and modeling a given dataset. The goal is to generate insights and predictions using machine learning techniques.

##Introduction:

In this report, we will discuss the implementation details of a neural network using an Ada-Act activation function for training. The Ada-Act activation function is defined as k0 + k1z + k2z^2, where z is the input and k0, k1, k2 are coefficients that are learned during training. The network architecture consists of three layers: an input layer, two hidden layers, and an output layer. The input and hidden layers use the Ada-Act activation function, while the output layer uses the softmax function. The network is trained using the categorical cross-entropy loss function.

##Algorithm:

Initialize the network parameters: weights and biases for the three layers (w1, b1, w2, b2, w3, b3) and coefficients for the Ada-Act activation function (k). The initial values for the weights and biases can be sampled from a random distribution, such as a normal distribution with mean 0 and standard deviation 1. The initial values for the Ada-Act coefficients can be set to some small constant, such as 0.01.

Preprocess the data: split the dataset into training and test sets, and standardize the features.

Perform forward propagation: compute the activation values for each layer using the input data and the current network parameters.

Compute the loss: use the output activation values and the true labels to compute the categorical cross-entropy loss.

Perform backpropagation: compute the gradients for the weights, biases, and Ada-Act coefficients using the output error and the activations from the hidden layers.

Update the network parameters: use the computed gradients to update the weights, biases, and Ada-Act coefficients using some learning rate alpha.

Repeat steps 3-6 for a specified number of epochs.

Evaluate the model: use the test set to evaluate the model's performance by computing the accuracy, confusion matrix, and classification report.

Implementation Details:

The code provided includes functions for each of the steps in the above algorithm. The function "g" computes the activation values for the Ada-Act activation function, and the function "g_derivative" computes the derivative of the activation function. The function "softmax" computes the activation values for the output layer using the softmax function. The function "CategoricalCrossEntropyLoss" computes the categorical cross-entropy loss between the true labels and the predicted probabilities. The function "backprop" computes the gradients for the weights, biases, and Ada-Act coefficients using the output error and the activations from the hidden layers. The function "update_parameters" updates the network parameters using the computed gradients and the learning rate. The function "forward_prop" performs a single forward propagation step, and the function "fit" trains the model by repeatedly calling the forward_prop and backprop functions. The function "predict" generates predictions using the trained model.

Conclusion:

In this report, we have described the implementation details of a neural network using an Ada-Act activation function for training. The network architecture consists of three layers and is trained using the categorical cross-entropy loss function. The provided code includes functions for each step in the training process, including forward propagation, backpropagation, and parameter updates.
