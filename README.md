Julia MNIST NN
==============

This is a Julia implementation of the sigmoid neuron, quadratic cost based neural net described in *Neural networks and deep learning* by Michael Nielsen.  The network trains itself to recognize handwritten digits using the MNIST dataset.

Usage
-----

Type 'make run' to run the default network setup.  This will import the dataset, and then train and test the network.

To play around with the network hyper parameters, follow the model shown in main.jl. Import the SigmoidNeuralNetwork module and declare a NeuralNetwork type object using create_network() to get started
