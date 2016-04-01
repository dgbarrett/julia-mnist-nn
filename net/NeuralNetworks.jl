
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

function gradient_descent( net::NeuralNetwork, 
	training_dataset::Array{Array{Float64,2}, 1}, 
	training_epochs::Int64, 
	training_batchsize::Int64, 
	learning_rate::Float64, 
	test_dataset::Array{Array{Float64,2},1}	)

	#TODO, verify data and label arrays are same size
	num_testitems = size(test_dataset[1], 2)
	num_trainitems = size(training_dataset[1], 2)

	for i = 1:training_epochs
		mini_batches = generate_randombatches( training_dataset, training_batchsize )

		for batch in mini_batches
			updatenetwork_with_batch( net, batch, learning_rate )
		end

		correct = testnetwork( test_dataset )
		println(string("Epoch ", i, " completed: ", correct, " / ", num_testitems))
	end

	return
end

function getcolumn( matrix::Array{Float64,2}, index::Int64 )
	return matrix[((size(matrix,1)*(index-1))+1):((size(matrix,1))*index)]
end

function updatenetwork_with_batch( net:NeuralNetwork, batch, learning_rate::Float64 )
	batch_size = size(batch[1], 2)
	nabla_b = []
	nabla_w = []

	for i = 1:(total_layers - 1)
		push!(nabla_b, zeros(Float64, layer_sizes[i + 1], 1))
		push!(nabla_w, zeros(Float64, layer_sizes[i + 1], layer_sizes[i]))
	end

	# calculate error for each input vector and adjust nabla_x accordingly
	for i = 1:batch_size
		inp_vector = getcolumn(batch[1], i)
		out_vector = getcolumn(batch[2], i)

		delta_nabla_b, delta_nabla_w = back_propagate( net, inp_vector, out_vector )

		for i = 1:net.num_layers
			nabla_b[i] += delta_nabla_b[i]
			nabla_w[i] += delta_nabla_w[i]

		end
	end

	# calculating new weights/biases while averaging gradient change over batch size
	for i = 1:net.num_layers
		net.weights[i] = (net.weights[i] - ((learning_rate/batch_size) * nabla_w[i]))
		net.biases[i] = (net.biases[i] - ((learning_rate/batch_size) * nabla_b[i]))
	end

	return
end

function back_propagate( net::NeuralNetwork, inp::Array{Float64, 2}, out::Array{Float64,2} )
	return
end

function testnetwork( dataset::Array{Array{Float64, 2}, 1} )
	return
end


function generate_randombatches( dataset::Array{Array{Float64,2}, 1}, training_batchsize::Int64 )
	data = dataset[1]
	solutions = dataset[2]
	added = []

	data_vectorsize = size(data, 1)
	soln_vectorsize = size(solutions, 1)

	dataset_size = remaining = size(data, 2)

	batches = []

	while remaining >= training_batchsize
		batch = Array(Any,0)
		batch_data = Array(Float64, data_vectorsize, training_batchsize)
		batch_solns = Array(Float64, soln_vectorsize, training_batchsize)

		for i = 1:training_batchsize
			#random index to determine which vector to include in the batch
			v = rand(1:dataset_size)
			while v in added
				v = rand(1:dataset_size)
			end
			push!(added,v)

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

	return batches
end

function sigmoid( input::Array{Float64, 2} )
	for i = 1:size(input,1)
		input[i] = 1.0/(1 + exp(-input[i]))
	end

	return input
end

end

