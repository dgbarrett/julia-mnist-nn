
module SigmoidNeuralNetwork

export NeuralNetwork, create_network, gradient_descent

#Object type representing the network
type NeuralNetwork
	num_layers::Int64
	layer_sizes::Vector{Int64}

	biases::Array{Array{Float64},1}
	weights::Array{Array{Float64}, 1}

	NeuralNetwork() = new(0, [], [], [])
end

#=Initalize a network with a list of layer sizes, layer_sizes[1] = input layer size,
 layer_size[n] = output_layer_size =#
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

#Pass an activation vector to the network and return the output vector
function feed_forward( net::NeuralNetwork, activation::Array{Float64, 2} )
	for (bias, weight) in zip(net.biases, net.weights)
		activation = sigmoid(weight*activation + bias)
	end
	return activation
end

#= 
Perform the gradient descent algorithm on training_dataset, for 
training_epochs, with training_batchsize and learing_rate.  After
each training epoch, test the effectivness of the network on test_dataset 
=#
function gradient_descent( net::NeuralNetwork, 
	training_dataset, 
	training_epochs::Int64, 
	training_batchsize::Int64, 
	learning_rate::Float64, 
	test_dataset	)
	println("\nStarting gradient descent...")

	#TODO, verify data and label arrays are same size
	num_testitems = size(test_dataset[1], 2)
	num_trainitems = size(training_dataset[1], 2)

	for i = 1:training_epochs
		println(string("\tStarting epoch ", i, "..."))
		mini_batches = generate_randombatches( training_dataset, training_batchsize )

		for batch in mini_batches
			updatenetwork_with_batch( net, batch, learning_rate )
		end

		correct = testnetwork( net, test_dataset )
		println(string("\tEpoch ", i, " completed: ", correct, " / ", num_testitems, "\n"))
	end

	return
end

#=
Update the weights and biases of the network by applying the backpropogation 
algorithm for each input/output vector pair in batch. 
=#
function updatenetwork_with_batch( net::NeuralNetwork, batch, learning_rate::Float64 )
	batch_size = size(batch[1], 2)
	nabla_b = []
	nabla_w = []

	imax = net.num_layers - 1
	for i = 1:imax
		push!(nabla_b, zeros(Float64, net.layer_sizes[i + 1], 1))
		push!(nabla_w, zeros(Float64, net.layer_sizes[i + 1], net.layer_sizes[i]))
	end

	# calculate error for each input vector and adjust nabla_x accordingly
	for i = 1:batch_size
		#get input vector and corresponding desired output vector from batch
		inp_vector = slicedim(batch[1], 2, i)::Array{Float64,2}
		out_vector = slicedim(batch[2], 2, i)::Array{Int64,2}

		#= calculate change in b and change in w due to networks 
		performance on the given input/ouput pair =#
		delta_nabla_b, delta_nabla_w = back_propagate( net, inp_vector, out_vector )

		for i = 1:net.num_layers-1
			nabla_b[i] += delta_nabla_b[i]
			nabla_w[i] += delta_nabla_w[i]
		end
	end

	# Update weights and biases by the average of the change in each over the mini-batch
	for i = 1:(net.num_layers-1)
		net.weights[i] = (net.weights[i] - ((learning_rate/batch_size) * nabla_w[i]))
		net.biases[i] = (net.biases[i] - ((learning_rate/batch_size) * nabla_b[i]))
	end

	return
end

#= Returns a change in w and change in b according to the computed 
gradient vector of the cost function =#
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
		# calculate weighted input to each neuron in the layer
		z = net.weights[i]*activation + net.biases[i]
		push!(z_vectors, z)
		# calculate the output activation of the neuron over z
		activation = sigmoid(z)
		push!(activations, activation)
	end

	#error in the output layer
	delta = cost_derivative(activations[end], out) * sigmoid_prime(z_vectors[end])
	#changes in weights and biases for output layer
	nabla_b[end] = delta
	nabla_w[end] = delta * transpose(activations[end-1])

	#backpropogate the error in the output layer through remaining layers
	for i = (net.num_layers-1):-1:2
		z = z_vectors[i]
		sp = sigmoid_prime(z)
		delta = (transpose(net.weights[i]) * delta) * sp

		nabla_b[i-1] = delta
		nabla_w[i-1] = delta * transpose(activations[i-1])
	end

	return (nabla_b, nabla_w)
end

#Test the networks performance using dataset
function testnetwork( net::NeuralNetwork, dataset )
	data = dataset[1]
	solutions = dataset[2]

	dataset_size = size(data,2)
	correct = 0

	for i = 1:dataset_size
		if (findmax(feed_forward(net, slicedim(data, 2, i)))[2]-1) == unvectorize(slicedim(solutions, 2, i))
			correct += 1
		end
	end

	return correct
end

#= 
Turn the vector representing the output from the network to a value [0,9]
representing the value of the digit described by the corresponding input vector
=#
function unvectorize( matrix )
	imax = size(matrix, 1)
	for i = 1:imax
		if matrix[i] == 1 
			return i-1 
		end
	end
end

#=
Generate batches with size training_batchsize from a given dataset
=#
function generate_randombatches( dataset, training_batchsize::Int64 )
	println("\t\tGenerating randomized batches...")
	data = dataset[1]
	solutions = dataset[2]
	added = zeros(Int8, size(data, 2))

	data_vectorsize = size(data, 1)
	soln_vectorsize = size(solutions, 1)

	dataset_size = remaining = size(data, 2)

	batches = []

	#= while there are still enough vectors in dataset to make a 
	correctly sized batch =#
	while remaining >= training_batchsize
		batch = Array(Any,0)
		batch_data = Array(Float64, data_vectorsize, training_batchsize)
		batch_solns = Array(Int64, soln_vectorsize, training_batchsize)

		for i = 1:training_batchsize
			srand()
			#random index to determine which vector to include in the batch
			v = rand(1:dataset_size)
			while added[v] == 1
				v = rand(1:dataset_size)
			end
			added[v] = 1

			#calculating start and end point of vector given the random index
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
	println("\t\tDone generating batches.")

	return batches
end

function getcolumn( matrix, index )
	return matrix[((size(matrix,1)*(index-1))+1):((size(matrix,1))*index)]
end

function sigmoid( input::Array{Float64, 2} )
	imax = size(input,1) * size(input,2)
	for i = 1:imax
		input[i] = 1.0/(1 + exp(-input[i]))
	end
	return input
end

#derivative of the quadratic cost function
function cost_derivative( out::Array{Float64,2}, expected_out::Array{Int64,2} )
	return out - expected_out
end

function sigmoid_prime( z::Array{Float64,2} )
	return dot( getcolumn(sigmoid(z),1),getcolumn(1-sigmoid(z),1) )
end

end

