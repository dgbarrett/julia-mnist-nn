module MNIST

export MNIST_getdata, MNIST_iscomplete

# info and files at http://yann.lecun.com/exdb/mnist/
TEST_DATA = "loader/data/t10k-images.idx3-ubyte"
TEST_LABELS = "loader/data/t10k-labels.idx1-ubyte"
TRAINING_DATA = "loader/data/train-images.idx3-ubyte"
TRAINING_LABELS = "loader/data/train-labels.idx1-ubyte"

LABEL_MAGICNUMBER = 2049
LABEL_MAX = 9
LABEL_MIN = 0

DATA_MAGICNUMBER = 2051
DATA_MAX = 255
DATA_MIN = 0

MNISTIMAGE_WIDTH = 28
MNISTIMAGE_HEIGHT = 28

#=
	@type MNISTData
		Container type for holding MNIST data and labels (solutions). 
=#
type MNISTData
	trainingsize::Int32
	trainingdata::Array{Float64, 2}
	traininglabel::Array{Int64, 2}

	testsize::Int32
	testdata::Array{Float64, 2}
	testlabel::Array{Int64, 2}

	completeload::Bool

	MNISTData() = new(	0, 
						Array(Float64,0,0),
						Array(Int64,0,0), 
						0, 
						Array(Float64,0,0),
						Array(Int64,0,0),
						false		)
end


#=
	@function MNIST_getdata
		Top level function that creates, loads, and returns a MNISTData object.
=#
function MNIST_getdata() 
	data = MNISTData()
	loadall_datasets( data )
	return data
end


#= 
	@function MNIST_iscomplete
		Check if a MNISTData object is complete.
	@return
		true
			Object contains all records.
		false
			Object is incompletley loaded. Records may be missing or incorrect.
=#
function MNIST_iscomplete( data::MNISTData )
	return data.completeload
end 


#=
	@function MNIST_loaddata
		Load data from the file paths specifed above by TRAINING_DATA, 
		TRAINING_LABELS, etc.., into a MNISTData container object.
=#
function loadall_datasets( data::MNISTData )
	lflag = dflag = false
	if load_data(data, TRAINING_DATA) 
		println("[Julia-MNIST] Training data loaded.")
		if load_labels( data, TRAINING_LABELS )
			println("[Julia-MNIST] Training labels loaded.")
			dflag = true
		end
	else
		println("[Julia-MNIST] ^^ Skipping loading of training labels. Training labels not loaded.")
	end

	if load_data(data, TEST_DATA)
		println("[Julia-MNIST] Test data loaded.")
		if load_labels( data, TEST_LABELS )
			println("[Julia-MNIST] Test labels loaded.")
			lflag = true
		end
	else
		println("[Julia-MNIST] ^^ Skipping loading of test labels. Test labels not loaded.")
	end

	(dflag && lflag) ? 
		( println("\n[Julia-MNIST] All data and labels loaded successfully."); data.completeload = true;) : 
		println("\n[Julia-MNIST] Incomplete loading. Dataset not complete.")
end


#=
	@function load_data
		Load MNIST data from the data file specifed by filename.
	@return
		true
			Data successfully parsed from filename.
		false
			Passed filename is not TEST_DATA or TRAINING DATA.
			Filename is not a file.
			Magic number read from file does not match DATA_MAGICNUMBER.
			Data images are not of size MNISTIMAGE_HEIGHT x MNISTIMAGE_WIDTH
			Data file contains out of bounds value (not in (DATA_MIN:DATA_MAX)).
=#
function load_data( data::MNISTData, filename::ASCIIString )
	if filename == TEST_DATA || filename == TRAINING_DATA
		if !isfile( filename ) 
			if filename == TRAINING_DATA 
				println("\n[Julia-MNIST] Could not locate training data file. Training data not loaded.")
			elseif filename == TEST_DATA
				println("[Julia-MNIST] Could not locate test data file. Test data not loaded.") 
			end
			return false
		end

		open( filename ) do datafile
			if filename == TRAINING_DATA
				println("\n[Julia-MNIST] Loading training data...")
			elseif filename == TEST_DATA
				println("[Julia-MNIST] Loading test data...")
			end

			if DATA_MAGICNUMBER != flip( read(datafile, UInt32) )
				println("[Julia-MNIST] !!ERROR!! Format error detected in data file. Ensure file is valid.")
				return false
			end

			if filename == TRAINING_DATA
				datasize = data.trainingsize = flip( read(datafile,UInt32) )
			elseif filename == TEST_DATA
				datasize = data.testsize = flip( read(datafile,UInt32) )
			end

			if MNISTIMAGE_HEIGHT != flip( read(datafile, UInt32) ) || MNISTIMAGE_WIDTH != flip( read(datafile, UInt32) )
				println("[Julia-MNIST] !!ERROR!! Data image is unexpected size. Ensure data file is valid.")
				return false
			end

			dense_data = Array(Float64, MNISTIMAGE_HEIGHT * MNISTIMAGE_WIDTH, datasize)

			if !read_densedata( datafile, dense_data )
				println("[Julia-MNIST]  !!ERROR!! Data file contains out of bounds values.")
				return false
			end

			if filename == TRAINING_DATA
				data.trainingdata = dense_data
			elseif filename == TEST_DATA
				data.testdata = dense_data
			end
			return true
		end
	else
		println("[Julia-MNIST] Data filename does not match any path specified in MNIST.jl. Specify incoming path names in MNIST.jl.")
		return false
	end
end 


#=
	@function read_densedata
		Read MNIST formatted data from the datafile IOStream into a dense 
		matrix (matrix).
	@return
		true
			All values in (DATA_MIN:DATA_MAX). Data is valid.
		false
			Values outside above specified range. Data is invalid.
=#
function read_densedata( datafile::IOStream, matrix::Matrix{Float64} )
	for i = 1:size(matrix, 2)
		for j = 1:size(matrix, 1)
			byte = read(datafile, UInt8)
			( byte in (DATA_MIN:DATA_MAX) ? matrix[j,i] = byte : return false)
		end
	end
	return true
end


#=
	@function load_labels
		Load MNIST labels  (solutions) from the label file specifed by 
		filename.
	@return
		true
			Labels successfully parsed from filename.
		false
			Passed filename is not TEST_LABELS or TRAINING_LABELS.
			Filename is not a file.
			Magic number read from file does not match LABEL_MAGICNUMBER.
			Label file contains out of bounds value (not in (LABEL_MIN:LABEL_MAX)).
=#
function load_labels(data::MNISTData, filename::ASCIIString )
	if filename == TEST_LABELS || filename == TRAINING_LABELS
		if !isfile( filename )
			if filename == TRAINING_LABELS
				println("[Julia-MNIST] Could not locate training label file. Training labels not loaded.")
			elseif filename == TEST_LABELS
				println("[Julia-MNIST] Could not locate test label file. Test labels not loaded.") 
			end
			return false
		end

		open( filename ) do datafile
			if filename == TRAINING_LABELS
				println("[Julia-MNIST] Loading training labels...")
			elseif filename == TEST_LABELS
				println("[Julia-MNIST] Loading test labels...")
			end

			if LABEL_MAGICNUMBER != flip(read(datafile, UInt32))
				println("[Julia-MNIST] !!ERROR!! Format error detected in training label file. Ensure data file is valid.")
				return false
			end

			if filename == TRAINING_LABELS
				samesize = ( data.trainingsize == flip( read(datafile, UInt32) ) )
			elseif filename == TEST_LABELS
				samesize = ( data.testsize == flip( read(datafile, UInt32) ) )
			end

			if samesize 
				lflag = false
				if filename == TRAINING_LABELS
					data.traininglabel = Array(Int64, 10 , data.trainingsize)
					if !read_labelvector( datafile, data.traininglabel )
						data.traininglabel = Array(Int8, 0)
						println("[Julia-MNIST]  !!ERROR!! Training label file contains out of bounds values.")
						return false
					end
				elseif filename == TEST_LABELS
					data.testlabel = Array(Int64, 10, data.testsize)
					if !read_labelvector( datafile, data.testlabel )
						data.testlabel = Array(Int8, 0)
						println("[Julia-MNIST]  !!ERROR!! Test label file contains out of bounds values.")
						return false
					end
				end
				return true
			else
				println("[Julia-MNIST] !!ERROR!! Data set size does not match solution set size.  Labels not loaded.")
				return false
			end
		end
	else
		println("[Julia-MNIST] Label filename does not match any path specified in MNIST.jl. Specify incoming path names in MNIST.jl.")
		return false
	end
end


#=
	@function read_label
		Read MNIST formatted lables (solutions) from the datafile IOStream into
		a vector (vector).
	@return
		true
			All values in (LABEL_MIN:LABEL_MAX). Labels are valid.
		false
			Values outside above specified range. Labels are invalid.
=#
function read_labelvector( datafile::IOStream , vector::Array{Int64, 2} )
	for i = 1:size(vector,2)
		byte = read(datafile, UInt8)
		( byte in (LABEL_MIN:LABEL_MAX) ) ? vector[byte+1, i] = 1 : return false
	end
	return true
end


#=
	@function flip
		Reverse the byte order of a 32 bit unsigned integer.
		Returns reversed number.
=#
flip(x::UInt32) = ((x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24))

end