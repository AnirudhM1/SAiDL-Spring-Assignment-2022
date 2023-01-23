# Custom types
Data = Tuple{Array{Float64, 2}, Vector{Int64}}
Params = Vector{Tuple{Array{Float64, 2}, Vector{Float64}}}

# Generate data
function generate_data(n_samples::Int64)::Data
    # Generate data for classification
    # n_samples : Number of samples
    
    X = Array{Float64}(undef, 0, 2)
    y = Array{Int64}(undef, 0)

    for _ in 1:n_samples
        x1 = rand()
        x2 = rand()
        if x1>0.5 && x2>0.5
            X = vcat(X, [x1, x2]')
            push!(y, 1)
        elseif x1<0.5 && x2<0.5
            X = vcat(X, [x1, x2]')
            push!(y, 1)
        else
            X = vcat(X, [x1, x2]')
            push!(y, 0)
        end
    end
    return X, y
end

# Subsample from the Data
function sub_sample(data::Data, n_samples::Int64)::Data
    # Subsample the data
    # data : The data to be subsampled
    # n_samples : Number of samples to be subsampled
    
    X, y = data
    idx = rand(1:size(X, 1), n_samples)
    return X[idx, :], y[idx]
end

# Calculate the accuracy of the Model
function accuracy(samples, forward::Function, data::Data)
    # Calculate the accuracy of the model
    # samples : Samples from the posterior distribution
    # forward : Forward function of the model
    # data : The data to be used for calculating the accuracy

    X, y = data
    y_pred = Array{Float64, 2}(undef, length(y), 1)
    for theta in samples
        y_hat = forward(theta, X)
        y_pred += y_hat
    end

    y_pred /= length(samples)
    y_pred = y_pred .>= 0.5
    return sum(y_pred .== y) / length(y)
end