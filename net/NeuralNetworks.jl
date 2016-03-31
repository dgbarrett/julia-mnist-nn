
module SigmoidNeuralNetwork

export NeuralNetwork, create_network, gradient_descent, create_randomized_minibatches

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

function gradient_descent( net::NeuralNetwork, training_data::Array{Float64,2}, training_epochs::Int64, training_batchsize::Int64, learning_rate::Float64, test_data::Array{Float64,2}=nothing)
	num_testitems = 0
	if test_data != nothing 
		num_testitems = size(test_data, 2)
	end
	num_trainitems = size(training_data, 2)

	for i = 1:training_epochs
		mini_batches = create_randmomized_minibatches( training_data, training_batchsize )
	end

	return
end

function create_randomized_minibatches( training_data::Array{Float64, 2}, training_batchsize::Int64 )
	mini_batches = []
	added_vectors = []
	remaining = dataset_size = size(training_data, 2)
	datavector_size = size(training_data, 1)

	srand()

	while remaining >= training_batchsize
		batch = Array(Float64, datavector_size, training_batchsize)
		for i = 1:training_batchsize
			
			randv = rand(1:dataset_size)
			while randv in added_vectors
				randv = rand(1:dataset_size)
			end
			push!(added_vectors, randv)
			copyvector( batch, i, training_data, randv )
			remaining -= 1

		end
		push!(mini_batches, batch)
	end

	return mini_batches
end

function copyvector( m1::Array{Float64, 2}, m1_index, m2::Array{Float64, 2}, m2_index )
	for i = 1:size(m1,1)
		m1[i, m1_index] = m2[i, m2_index]
	end
end

function sigmoid( input::Array{Float64, 2} )
	for i = 1:size(input,1)
		input[i] = 1.0/(1 + exp(-input[i]))
	end

	return input
end

end

