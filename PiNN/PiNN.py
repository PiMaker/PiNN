import numpy as np
import matplotlib.pyplot as plt

#
#
#   PARSE INPUT
#
#

step_size = 0.005
#inputs = 4
# data = [
#     (np.matrix('0;0;0;0'), 0),
#     (np.matrix('0;0;0;1'), 0),
#     (np.matrix('0;0;1;0'), 0),
#     (np.matrix('0;1;0;0'), 0),
#     (np.matrix('1;0;0;0'), 0),
#     (np.matrix('1;1;0;0'), 1),
#     (np.matrix('0;1;1;0'), 1),
#     (np.matrix('0;0;1;1'), 1),
#     (np.matrix('1;0;0;1'), 1),
#     (np.matrix('0;1;0;1'), 1),
#     (np.matrix('1;0;1;0'), 1),
# ]

print('-----------------')
print(' Loading data... ')
print('-----------------')

data_table = [
    ['b', 'c', 'x', 'f', 'k', 's'],
    ['f', 'g', 'y', 's'],
    ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
    ['t', 'f'],
    ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
    ['a', 'd', 'f', 'n'],
    ['c', 'w', 'd'],
    ['b', 'n'],
    ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
    ['e', 't'],
    ['b', 'c', 'u', 'e', 'z', 'r', '?'],
    ['f', 'y', 'k', 's'],
    ['f', 'y', 'k', 's'],
    ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    ['p', 'u'],
    ['n', 'o', 'w', 'y'],
    ['n', 'o', 't'],
    ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
    ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
    ['a', 'c', 'n', 's', 'v', 'y'],
    ['g', 'l', 'm', 'p', 'u', 'w', 'd']
]


# Thanks stack overflow
def flatten(seq,container=None):
    if container is None:
        container = []
    for s in seq:
        if hasattr(s,'__iter__') and type(s) != type(''):
            flatten(s,container)
        else:
            container.append(s)
    return container


data = []
inputs = len(flatten(data_table))

with open('agaricus-lepiota.data') as f:
    for line in f:
        if line.strip() == '':
            continue

        split = line.strip().split(',')

        # [1, 0] = edible, [0, 1] = poisonous
        pred = np.array([1, 0]) if split[0] == 'e' else np.array([0, 1])

        m = np.zeros((inputs, 1))
        mcur = 0

        for c in range(0, len(data_table)):
            try:
                i = data_table[c].index(split[c + 1])
                m[mcur + i] = 1
            except ValueError:
                pass

            mcur += len(data_table[c])

        data.append((m, pred))

print(str(len(data)) + ' data rows loaded')


#
#
#   NEURAL NETWORK
#
#

def buildNetwork():
    # Generates a default neural network

    # Network layout:
    # * One matrix per layer
    # * Every row is a neuron
    # * Every column is a connection to a specific node from the previous layer
    # => shape: (neurons in previous layer)x(neurons in layer)

    # 2 hidden layers
    # array[0] = First hidden layer, array[2] = Output layer

    outputs = 2
    init_max = 0.02
    return [np.random.rand(70, inputs)*init_max-(init_max/2), np.random.rand(outputs, 70)*init_max-(init_max/2)]


def run(net, input):
    # Runs the neural network on the given input

    out = np.array(input)

    weighted = []

    for layer in net:
        # Dot product with intermediate step
        temp = np.multiply(layer, np.transpose(out))
        weighted.append(temp)
        out = np.sum(temp, axis=1)

        # Map ReLU over every element
        out = np.maximum(0, out)

    # What is left is the output matrix
    # We also return the weighted input matrix per layer for use in backprop
    return (out, weighted)


# z is the weighted input matrix for each layer
def trainOnce(net, z, direction):
    # Train the network using backpropagation

    dx = np.array(direction)

    # Reverse iteration over layers
    for l in range(len(z)-1, 0, -1):
        z_layer = z[l]

        # Reduce z_layer to either 1 (has been activated) or 0 (1/x)
        z_layer[z_layer != 0] = 1

        # Chain rule (cut off propagation if necessary)
        for neuron in range(0, z_layer.shape[0]):
            z_layer[neuron] *= dx[neuron]

        # Apply new dx to weights in current layer (step_size is multiplied to each weight accordingly)
        net[l] = np.add(net[l], np.multiply(z_layer, step_size))

        # Calculate new dx for next step
        dx = np.transpose(np.sum(z_layer, axis=0))

    return net


# Loss helpers
def softmax(arr):
    arr_exp = np.exp(arr)
    exp_sum = np.sum(arr_exp)
    return np.divide(arr_exp, exp_sum)

def crossEntropy(corr, pred):
    return -np.sum(np.multiply(corr, np.log(pred)))

def loss(corr, pred):
    # Cross-Entropy loss
    # Also calculates the gradient of the loss with respect to the input parameters
    p_max = softmax(pred)
    return (crossEntropy(corr, p_max), -np.subtract(p_max, corr))


def check(net):
    # Simply check how much of the test data can be classified correctly
    wrong = 0
    for set in data:
        corr = set[1]
        pred, _ = run(net, set[0])
        if np.argmax(corr) != np.argmax(pred):
            wrong += 1

    return len(data) - wrong


print()
print('-----------------------')
print(' Beginning training... ')
print('-----------------------')

net = buildNetwork()
losses = []

for i in range(0, int(len(data)*100)):
    # Sample a data set
    set = data[i % len(data)]

    # Run the network
    out, z = run(net, set[0])

    # Calculate loss
    l, grad = loss(set[1], out)
    losses.append(l)
    
    # DEBUG print
    # print(str(set[1]) + " -> " + str(out) + " -> " + str(grad))

    # Output for measly humans
    print('Loss at iteration ' + str(i) + ': ' + str(l))

    # Perform training
    net = trainOnce(net, z, grad)

print('Training done, calculating statistics...')
print('Correct: ' + str(check(net)) + ' of ' + str(len(data)))
print()

# Plot loss
plt.plot(losses)
plt.ylabel('Cross-Entropy Loss')
plt.xlabel('Iteration')
plt.title('Loss function')
plt.show()
