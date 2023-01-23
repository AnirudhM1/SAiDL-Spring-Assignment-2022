include("mcmc.jl")
include("utils.jl")

# HYPERPARAMETERS
TRAINING_DATA_SIZE = 5000
TESTING_DATA_SIZE = 1000
N_SAMPLES = 25 
N_BURNIN = 10
SHAPE = [2, 2, 1]

# Generate data
training_data = generate_data(TRAINING_DATA_SIZE)
testing_data = generate_data(TESTING_DATA_SIZE)

# Create the model
model = Model(initialize_params(SHAPE), fully_connected)

# Sample from the posterior distribution using metrapolis_hastings algorithm
println("Starting sampling...")
println("Accepted: 0    Rejected: 0")
samples, accepted, rejected = metrapolis_hastings(model, training_data, N_SAMPLES, N_BURNIN)
println("Sampling Complete!\n")

println("Number of accepted samples: ", accepted)
println("Number of rejected samples: ", rejected)

println()

# Calculate the accuracy of the model on the Training Data
train_accuracy = accuracy(samples, model.forward, training_data)
println("Training Accuracy: ", train_accuracy)

# Calculate the accuracy of the model on the Testing Data
test_accuracy = accuracy(samples, model.forward, testing_data)
println("Testing Accuracy: ", test_accuracy)
