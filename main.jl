include("net/NeuralNetworks.jl")
include("loader/MNIST.jl")

importall SigmoidNeuralNetwork
importall MNIST

#example
data = MNIST_getdata()

train = []
test = []

push!(train, data.trainingdata)
push!(train, data.traininglabel)
push!(test, data.testdata)
push!(test, data.testlabel)

net = create_network([784,30,10])
gradient_descent(net, train, 30, 5, 1.5, test)
