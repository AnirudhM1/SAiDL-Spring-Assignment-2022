mutable struct Model
    inputs
    hidden
end

function metrapolis_hastings(model, data, N)
    theta = get_initial_weights(model.inputs, model.hidden)
    accepted = 0
    rejected = 0
    accepted_theta = []
    epoch=0
    while true
        epoch += 1
        theta_dash = Q(theta)
        ratio = log_f(model, theta_dash, data) - log_f(model, theta, data)
        p = min(0, ratio)
        k = sample_log_Bernoulli(p)
        if k == 1
            theta = theta_dash
            push!(accepted_theta, theta)
            accepted += 1
        else
            rejected += 1
        end
        
        if accepted == N
            break
        end
        if epoch%100 == 0
            acc = accuracy(model, accepted_theta, 1000)
            println("Epoch: ", epoch)
            println("Accuracy: ", acc)
        end
        if epoch == 30000
            break
        end
    end
    return accepted_theta, accepted, rejected
end

function Q(theta)
    theta_dash = [sample_normal(w, 1) for w in theta]
    return theta_dash
end

function f(model, theta, data)
    likelihood_value = likelihood(model, theta, data)
    prior_value = prior(theta)
    return likelihood_value * prior_value
end

function log_f(model, theta, data)
    likelihood_value = log_likelihood(model, theta, data)
    prior_value = log_prior(theta)
    return likelihood_value + prior_value
end

function prior(theta)
    p = 1
    for w in theta
        p *= pdf(w, 0, 1)
    end
    return p
end

function log_prior(theta)
    p = 0
    for w in theta
        p += log_pdf(w, 0, 1)
    end
    return p
end

function likelihood(model, theta, data)
    p = 1
    for d in data
        X, y = d
        y_pred = forward(model, theta, X)
        p *= loss(y_pred, y)
    end
    return p
end

function log_likelihood(model, theta, data)
    p = 0
    for d in data
        X, y = d
        y_pred = forward(model, theta, X)
        p += log_loss(y_pred, y)
    end
    return p
end

function loss(y_pred, y) # This function assumes that the output passes through the sigmoid function
    if y == 1
        return y_pred
    else
        return (1-y_pred)
    end
end

function log_loss(y_pred, y)
    if y == 1
        return log(y_pred)
    else
        return log(1-y_pred)
    end
end

average(Y) = sum(Y)/length(Y)

function pdf(d, u, sigma)
    a = ((d-u)^2)/(2*sigma*sigma)
    expo = exp(-a)
    a = 1/sigma
    return a*expo
end

function log_pdf(d, u, sigma)
    a = ((d-u)^2)/(2*sigma*sigma)
    return -a - log(sigma)
end

function sample_Bernoulli(p)
    if p == 1
        return 1
    end
    random_num = rand()
    if (random_num > p)
        return 0
    else 
        return 1
    end
end

function sample_log_Bernoulli(p)
    if p == 0
        return 1
    end
    random_num = log(rand())
    if (random_num > p)
        return 0
    else 
        return 1
    end
end

sample_normal(u, sigma) = u + sigma * randn()

sigmoid(x) = 1/(1+exp(-x))

function forward(model, theta, x)
    # Assigning weights to the respective weight matrices
    w_hidden = Matrix{Float64}(undef, model.inputs, model.hidden)
    b_hidden = Matrix{Float64}(undef, 1, model.hidden)
    w_output = Matrix{Float64}(undef, model.hidden, 1)
    b_output = Matrix{Float64}(undef, 1, 1)
    weights = theta

    index=1

    for i=1:model.inputs
        for j=1:model.hidden
            w_hidden[i,j] = weights[index]
            index += 1
        end
    end
    for i=1:model.hidden
        b_hidden[1,i] = weights[index]
        index += 1
    end
    for i=1:model.hidden
        w_output[i,1] = weights[index]
        index += 1
    end
    b_output[1,1] = weights[index]

    # Forward pass
    input = transpose([inp for inp in x])
    hidden = sigmoid.(input*w_hidden + b_hidden)
    output = sigmoid.(hidden*w_output + b_output)
    return output[1,1]
end


function generate_data(num_samples)
    data = []
    for _=1:num_samples
        a = rand()
        b = rand()
        x = (a, b)
        case_1 = a > 0.5
        case_2 = b > 0.5
        if ((case_1 && !case_2) || (!case_1 && case_2))
            y = 1
        else
            y = 0
        end
        d_i = (x, y)
        push!(data, d_i)
    end
    return data
end

function accuracy(model, theta_samples, num_samples)
    data = generate_data(num_samples)
    correct = 0
    for d in data
        X, y = d
        y_pred = []
        for theta in theta_samples
            y_pred_i = forward(model, theta, X)
            push!(y_pred, y_pred_i)
        end
        y_hat = (sum(y_pred)/length(y_pred)) > 0.5 ? 1 : 0
        if y_hat == y
            correct += 1
        end
    end
    acc = correct/length(data)
    return acc
end

function get_initial_weights(inputs, hidden)
    num_weights = inputs*hidden + hidden*1 + hidden + 1
    weights = rand(num_weights)
    return weights
end

function random_sampler(model, num_samples)
    theta_samples = []
    for _=1:num_samples
        weights = get_initial_weights(model.inputs, model.hidden)
        push!(theta_samples, weights)
    end
    return theta_samples
end

data = generate_data(500)
model = Model(2, 6)

println("Metrapolis Hastings Sampler")
accepted_theta, accepted, rejected = metrapolis_hastings(model, data, 1000)
# Printing accuracy with subsampled theta
acc = accuracy(model, accepted_theta, 120)
println("Accepted: ", accepted)
println("Rejected: ", rejected)
println("Accuracy: ", acc)
println("-----------------------")
println("Random Sampler")
accepted_theta = random_sampler(model, accepted)
acc = accuracy(model, accepted_theta, 120)
println("Accuracy: ", acc)