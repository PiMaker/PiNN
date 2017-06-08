import numpy as np


step_size = 0.01
inputs = 4
data = [
    (np.matrix('0;0;0;0'), 0),
    (np.matrix('0;0;0;1'), 0),
    (np.matrix('0;0;1;0'), 0),
    (np.matrix('0;1;0;0'), 0),
    (np.matrix('1;0;0;0'), 0),
    (np.matrix('1;1;0;0'), 1),
    (np.matrix('0;1;1;0'), 1),
    (np.matrix('0;0;1;1'), 1),
    (np.matrix('1;0;0;1'), 1),
    (np.matrix('0;1;0;1'), 1),
    (np.matrix('1;0;1;0'), 1),
]


def buildNetwork():

    # Generates a default neural network

    # Network layout:
    # * One matrix per layer
    # * Every row is a neuron
    # * Every column is a connection to a specific node from the previous layer
    # => shape: (neurons in previous layer)x(neurons in layer)

    # For now: Hardcoded 4 inputs, 2 hidden layer (8 neurons each), 1 output
    # array[0] = First hidden layer, array[2] = Output layer

    return [np.random.rand(20, inputs), np.random.rand(20, 20), np.random.rand(1, 20)]


def run(net, input):

    # Runs the neural network on the given input

    out = np.array(input)

    weighted = []

    for layer in net:
        # Dot product with intermediate step
        temp = np.multiply(layer, out.transpose())
        weighted.append(temp)
        out = temp.sum(axis=1)

        # Map ReLU over every element
        out = vfunc(out)

    # What is left is the (1,1) output matrix
    # We also return the weighted input matrix per layer for use in backprop
    return (out.item(), weighted)


# ReLU Helper
# (Vectorize function for easy application to arrays)
vfunc = np.vectorize(lambda t: max(0, t), otypes=[np.float])
    

# z is the weighted input matrix for each layer
def trainOnce(net, z, direction):
    
    # Train the network using backpropagation

    dx = np.array([direction])

    # Reverse iteration over layers
    for l in range(len(z)-1, 0, -1):
        layer = z[l]

        # Multiply neurons with passed down value (cut off propagation if necessary)
        for neuron in range(0, len(layer)):
            layer[neuron] *= dx[neuron]

        # Apply new dx to weights in current layer
        net[l] = np.add(net[l], vlearn(layer))

        # Calculate new dx for next step
        dx = vnorm(np.transpose(np.sum(layer, axis=0)))


# Backprop Helpers
vnorm = np.vectorize(lambda t: -1 if t < 0 else (1 if t > 0 else 0), otypes=[np.float])
vlearn = np.vectorize(lambda t: step_size if t > 0 else (-step_size if t < 0 else 0), otypes=[np.float])


def loss(net):

    # Calculate loss

    total = 0.0
    for set in data:
        corr = set[1]
        pred, _ = run(net, set[0])
        temp = corr-pred
        total += temp*temp
        
    return total


def countCorrect(net):

    # Simply check how much of the test data can be classified correctly

    wrong = 0

    for set in data:
        corr = set[1]
        pred, _ = run(net, set[0])
        if (corr < 0.5 and pred > 0.5) or (corr > 0.5 and pred < 0.5):
            wrong += 1

    return len(data) - wrong


print('Beginning training')

net = buildNetwork()

for i in range(1000):
    # Output loss
    l = loss(net)
    print('Loss at iteration ' + str(i) + ': ' + str(l))

    # Break if loss small enough
    if l < 0.8:
        print('Reached loss delta!')
        break

    if l < 5 and step_size >= 0.01:
        print('Decreasing step size')
        step_size /= 10

    # Sample a data set
    set = data[i % len(data)]
    # Run the network
    out, z = run(net, set[0])

    # Select direction
    dir = 0
    if set[1] == 1 and out < 0.5:
        dir = 1
    elif set[1] == 0 and out > 0.5:
        dir = -1

    # Perform training
    trainOnce(net, z, dir)

print('Loss (end): ' + str(loss(net)))
print('Correct: ' + str(countCorrect(net)) + ' of ' + str(len(data)))
