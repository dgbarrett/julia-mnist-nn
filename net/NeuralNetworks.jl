
module SigmoidNeuralNetwork

export NeuralNetwork, create_network, sigmoid

type NeuralNetwork
	num_layers::Int64
	layer_sizes::Vector{Int64}

	biases
	weights

	NeuralNetwork() = new(0, [], [], [])
end

function create_network( layer_sizes::Vector{Int64} )
	net = NeuralNetwork()
	total_layers = size(layer_sizes, 1)

	if total_layers < 3
		println("Error. Minimum layer size is 3.")
		return nothing
	end

	net.num_layers = total_layers
	net.layer_sizes = layer_sizes


	#=
		Loading the bias vectors for each layer.  No bias vector for input 
		layer, thus the bias vector for the first hidden layer resides at 
		net.biases[1].
	=#
	for i = 1:(total_layers - 1)
		layer_biasvector = Array(Float64, layer_sizes[i + 1], 1)
		randn!(layer_biasvector)
		push!(net.biases, layer_biasvector)

		layer_weightmatrix = Array(Float64, layer_sizes[i + 1], layer_sizes[i])
		randn!(layer_weightmatrix)
		push!(net.weights, layer_weightmatrix)
	end

	return net
end

function feed_forward( net::NeuralNetwork, activation::Array{Float64, 1} )
	for (bias, weight) in zip(net.biases, net.weights)
		activation = sigmoid(weight*activation + bias)
	end
	return activation
end

function sigmoid( input::Array{Float64, 2} )
	for i = 1:size(input,1)
		input[i] = 1.0/(1 + exp(-input[i]))
	end

	return input
end

end

