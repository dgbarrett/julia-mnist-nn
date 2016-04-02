
module SigmoidNeuralNetwork

export NeuralNetwork, create_network, gradient_descent, generate_randombatches

type NeuralNetwork
	num_layers::Int64
	layer_sizes::Vector{Int64}

	biases
	weights

	NeuralNetwork() = new(0, [], [], [])
end

function create_network( layer_sizes::Vector{Int64} )
	println("\nCreating network...")
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
	println("\tInitializing weights and biases.")
	for i = 1:(total_layers - 1)
		layer_biasvector = Array(Float64, layer_sizes[i + 1], 1)
		randn!(layer_biasvector)
		push!(net.biases, layer_biasvector)

		layer_weightmatrix = Array(Float64, layer_sizes[i + 1], layer_sizes[i])
		randn!(layer_weightmatrix)
		push!(net.weights, layer_weightmatrix)
	end
	println("\tDone initializing.")

	return net
end

function feed_forward( net::NeuralNetwork, activation::Array{Float64, 2} )
	for (bias, weight) in zip(net.biases, net.weights)
		activation = sigmoid(weight*activation + bias)
	end
	return activation
end

function gradient_descent( net::NeuralNetwork, 
	training_dataset::Array{Array{Real,2}, 1}, 
	training_epochs::Int64, 
	training_batchsize::Int64, 
	learning_rate::Float64, 
	test_dataset::Array{Array{Real,2},1}	)
	println("\nStarting gradient descent...")

	#TODO, verify data and label arrays are same size
	num_testitems = size(test_dataset[1], 2)
	num_trainitems = size(training_dataset[1], 2)

	for i = 1:training_epochs
		mini_batches = generate_randombatches( training_dataset, training_batchsize )
		batches = size(mini_batches, 1)

		for batch in mini_batches
			updatenetwork_with_batch( net, batch, learning_rate )
			batches -= 1 
			println("\t\tRemaining batches: ", batches)
		end

		correct = testnetwork( net, test_dataset )
		println(string("\tEpoch ", i, " completed: ", correct, " / ", num_testitems))
	end

	return
end

function updatenetwork_with_batch( net::NeuralNetwork, batch, learning_rate::Float64 )
	batch_size = size(batch[1], 2)
	nabla_b = []
	nabla_w = []

	for i = 1:(net.num_layers - 1)
		push!(nabla_b, zeros(Float64, net.layer_sizes[i + 1], 1))
		push!(nabla_w, zeros(Float64, net.layer_sizes[i + 1], net.layer_sizes[i]))
	end

	# calculate error for each input vector and adjust nabla_x accordingly
	for i = 1:batch_size
		inp_vector = slicedim(batch[1], 2, i)
		out_vector = slicedim(batch[2], 2, i)

		delta_nabla_b, delta_nabla_w = back_propagate( net, inp_vector, out_vector )

		for i = 1:net.num_layers-1
			nabla_b[i] += delta_nabla_b[i]
			nabla_w[i] += delta_nabla_w[i]
		end
	end

	# calculating new weights/biases while averaging gradient change over batch size
	for i = 1:(net.num_layers-1)
		net.weights[i] = (net.weights[i] - ((learning_rate/batch_size) * nabla_w[i]))
		net.biases[i] = (net.biases[i] - ((learning_rate/batch_size) * nabla_b[i]))
	end

	return
end

function back_propagate( net::NeuralNetwork, inp::Array{Float64, 2}, out::Array{Int64,2} )
	nabla_b = []
	nabla_w = []

	for i = 1:(net.num_layers - 1)
		push!(nabla_b, zeros(Float64, net.layer_sizes[i + 1], 1))
		push!(nabla_w, zeros(Float64, net.layer_sizes[i + 1], net.layer_sizes[i]))
	end

	activation = inp
	activations = []
	push!(activations, activation)

	z_vectors = []

	for i = 1:(net.num_layers-1)
		z = net.weights[i]*activation + net.biases[i]
		push!(z_vectors, z)
		activation = sigmoid(z)
		push!(activations, activation)
	end

	delta = cost_derivative(activations[end], out) * sigmoid_prime(z_vectors[end])
	nabla_b[end] = delta
	nabla_w[end] = delta * transpose(activations[end-1])

	for i = (net.num_layers-1):-1:2
		z = z_vectors[i]
		sp = sigmoid_prime(z)
		delta = (transpose(net.weights[i]) * delta) * sp

		nabla_b[i-1] = delta
		nabla_w[i-1] = delta * transpose(activations[i-1])
	end

	return (nabla_b, nabla_w)
end

function testnetwork( net::NeuralNetwork, dataset::Array{Array{Real, 2}, 1} )
	data = dataset[1]
	solutions = dataset[2]

	dataset_size = size(data,2)
	correct = 0

	for i = 1:dataset_size
		if findmax(feed_forward(net, getcolumn(data, i)))[2] == unvectorize(solutions[i])
			correct += 1
		end
	end

	return correct
end

function unvectorize( matrix::Array{Float64,2} )
	imax = size(matrix, 1)
	for i = i:imax
		if matrix[i] == 1 
			return i+1 
		end
	end
end

function generate_randombatches( dataset::Array{Array{Real,2}, 1}, training_batchsize::Int64 )
	println("\n\tGenerating randomized batches...")
	data = dataset[1]
	solutions = dataset[2]
	added = zeros(Int8, size(data, 2))

	data_vectorsize = size(data, 1)
	soln_vectorsize = size(solutions, 1)

	dataset_size = remaining = size(data, 2)

	batches = []

	while remaining >= training_batchsize
		batch = Array(Any,0)
		batch_data = Array(Float64, data_vectorsize, training_batchsize)
		batch_solns = Array(Int64, soln_vectorsize, training_batchsize)

		for i = 1:training_batchsize
			#random index to determine which vector to include in the batch
			v = rand(1:dataset_size)
			while added[v] == 1
				v = rand(1:dataset_size)
			end
			added[v] = 1

			data_dest_start = (data_vectorsize*(i-1))+1
			data_dest_end = data_vectorsize*i

			soln_dest_start = (soln_vectorsize*(i-1))+1
			soln_dest_end = soln_vectorsize*i

			data_src_start = (data_vectorsize*(v-1))+1
			data_src_end = data_vectorsize*v

			soln_src_start = (soln_vectorsize*(v-1))+1
			soln_src_end = soln_vectorsize*v

			# copying column vectors
			batch_data[data_dest_start:data_dest_end] = data[data_src_start:data_src_end]
			batch_solns[soln_dest_start:soln_dest_end] = solutions[soln_src_start:soln_src_end]
			remaining -= 1
		end
		push!(batch, batch_data)
		push!(batch, batch_solns)

		push!(batches, batch)
	end
	println("\tDone generating batches.")

	return batches
end

function getcolumn( matrix::Array{Float64,2}, index::Int64 )
	return matrix[((size(matrix,1)*(index-1))+1):((size(matrix,1))*index)]
end

function getcolumn( matrix::Array{Int64,2}, index::Int64 )
	return matrix[((size(matrix,1)*(index-1))+1):((size(matrix,1))*index)]
end

function sigmoid( input::Array{Float64, 2} )
	imax = size(input,1) * size(input,2)
	for i = 1:imax
		input[i] = 1.0/(1 + exp(-input[i]))
	end
	return input
end

function cost_derivative( out::Array{Float64,2}, expected_out::Array{Int64,2} )
	return out - expected_out
end

function sigmoid_prime( z::Array{Float64,2} )
	return dot( getcolumn(sigmoid(z),1),getcolumn(1-sigmoid(z),1) )
end

end

