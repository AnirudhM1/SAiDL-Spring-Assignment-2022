include("nn.jl")
include("utils.jl")

# Custom types
Data = Tuple{Array{Float64, 2}, Vector{Int64}}

function Q(theta::Params)::Params
    # Transition function for the markov chain
    # The transition function is modelled as a guassian with mean as the previous sample and unit standard deviation
    
    # theta : Previous sample in the markov chain
    
    theta_dash = Vector{Tuple{Array{Float64, 2}, Vector{Float64}}}() # New sample in the markov chain
    
    for (w, b) in theta
        w_dash = w + randn(size(w))
        b_dash = b + randn(size(b))
        push!(theta_dash, (w_dash, b_dash))
    end
    
    return theta_dash
end

function log_prior(theta::Params)
    # Calculate the log of the prior probability
    # The prior probability is modelled as a guassian with mean 0 and unit standard deviation
    
    # Since the constant values get cancelled in the equation, 
    
    # theta : Sample in the markov chain
    
    std = 1
    
    log_prob = 0
    
    for (w, b) in theta
        log_prob += sum(w.^2)/(2*std) + sum(b.^2)/(2*std)
    end
    
    return -log_prob
end

function log_likelihood(theta::Params, forward::Function, data::Data)
    # Calculate the log of the likelihood
    
    # theta : Sample in the markov chain
    # forward : Forward function of the model
    # data : The training data
    
    X, y = data
    y_hat = forward(theta, X) # Predicted output
    
    log_prob = 0
    
    for (pred, target) in zip(y_hat, y)
        log_prob += target * log(pred) + (1 - target) * log(1 - pred)
    end
    
    return log_prob
end

function metrapolis_hastings(model::Model, data::Data, n_samples::Int64, n_burnin::Int64)
    # Sample from the posterior distribution
    # using Metrapolis-Hastings algorithm
    # model : The neural network model
    # data : The training data
    # n_samples : Number of samples to be generated
    # n_burnin : Number of burn-in samples

    forward = model.forward # Forward function of the model

    theta::Params = model.theta # Previous sample in the markov chain
    accepted = 0 # Number of accepted samples
    rejected = 0 # Number of rejected samples
    samples = [] # Samples generated

    while accepted < n_samples + n_burnin
        # Print the current data
        print("\u1b[1F")
        print("Accepted: ", accepted, "    Rejected: ", rejected)
        print("\u1b[0K")
        println()

        # Generate a new sample with the transition function and previous sample in the markov chain
        theta_dash = Q(theta)

        # Calculate the acceptance probability
        # We are dealing with logs here for numerical stability

        # Calculate the log of the prior probability
        log_prior_prob = log_prior(theta)
        log_prior_dash_prob = log_prior(theta_dash)

        # Calculate the log of the likelihood
        log_likelihood_prob = log_likelihood(theta, forward, data)
        log_likelihood_dash_prob = log_likelihood(theta_dash, forward, data)

        # Calculate the log of the acceptance probability
        log_prob = min(0, log_prior_dash_prob + log_likelihood_dash_prob - log_prior_prob - log_likelihood_prob)

        # Accept or reject the sample
        if log(rand()) < log_prob # Accept the sample
            theta = theta_dash
            accepted += 1
            push!(samples, theta)
        else
            rejected += 1 # Reject the sample
        end
    end

    samples = samples[n_burnin+1:end] # Remove the burn-in samples
    return samples, accepted, rejected
end
