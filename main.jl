include("net/NeuralNetworks.jl")
importall SigmoidNeuralNetwork

#example
net = create_network([784,30,10])