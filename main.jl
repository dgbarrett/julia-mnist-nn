include("net/NeuralNetworks.jl")
include("loader/MNIST.jl")

importall SigmoidNeuralNetwork
importall MNIST

#example
data = MNIST_getdata()

train = Array(Array{Real,2}, 0)
test = Array(Array{Real,2}, 0)

push!(train, data.trainingdata)
push!(train, data.traininglabel)
push!(test, data.testdata)
push!(test, data.testlabel)

net = create_network([784,100,10])
gradient_descent(net, train, 30, 10, 3.0, test)
