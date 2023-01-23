# Custom types
Params = Vector{Tuple{Array{Float64, 2}, Vector{Float64}}}

# Structure to define the neural network model
struct Model
    theta :: Params # Contains the weights/biases of the model
    forward :: Function # The forward function of the model
end

function fully_connected(theta::Params, x::Array{Float64, 2})
    # Generalised forward function for the network
    # theta : Contains the weights/biases of the network
    # x : Input to the network

    y = x
    for (w, b) in theta
        # println(size(w), size(b), size(x))
        y = y*w .+ b'
        y = sigmoid.(y)
    end
    return y
end

function initialize_params(shape::Vector{Int64}) :: Params
    # Initialize the weights and biases of the network
    # shape : Contains the number of neurons in each layer

    theta = Vector{Tuple{Array{Float64, 2}, Vector{Float64}}}()
    for i in 1:length(shape) - 1
        w = randn(shape[i], shape[i + 1])
        b = randn(shape[i + 1])
        push!(theta, (w, b))
    end
    return theta
end

# Activation functions
sigmoid(x) = 1 / (1 + exp(-x))
relu(x) = max.(0, x)
leaky_relu(x, alpha=0.01) = max.(0, x) + alpha * min.(0, x)