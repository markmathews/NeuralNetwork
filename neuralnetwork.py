import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    A feed-forward neural network

    Attributes
    -----
    nodes_per_layer : list
        Number of nodes at each layer of the network, including input and
        output layers

    num_layers : int
        Total number of layers in the network, including input and output

    learning_rate : float
        The scalar by which weight updates are multiplied to control learning

    weights : list
        List of weight matrices corresponding to transformations from the
        output of one layer to the input of the next

    initial_weights_copy : list
        A copy of the initialised weights

    Y_scale : float
        The scalar which maps outputs from the range [-1, 1] back to their
        original range

    Methods
    -----
    activation
        The activation function. Can be set to either self.sigmoid or self.relu

    activationDerv
        Derivative of the activation function. Can be set to either
        self.sigmoidDerv or self.reluDerv

    initialiseWeights
        Initialise weight matrices by sampling from a uniform distribution
        over [0, 1)

    resetWeights
        Reset all weights to self.initial_weights_copy

    calculateDeltas
        Calculate the vectors whose outer product with the outputs from a
        previous layer give the updates for weights between the two layers

    updateWeights
        Perform weight updates

    feedForward
        Propagate an input through the network, returning the inputs and
        outputs at every layer

    lastOutput
        Return the output from the last layer for an input vector

    preProcessData
        Format input data arrays to have shape (d,1) where d represents the
        dimensionality of each data point

    train
        Train the network

    test
        Test the network on some data and return resulting predictions

    show_plot
        Plot two input data arrays
    """
    def __init__(
        self,
        nodes_per_layer,  # 1D array w/ num. of nodes per layer
        learning_rate=1,
        activation='relu'
        # activation='sigmoid'
    ):
        self.nodes_per_layer = nodes_per_layer
        self.num_layers = len(nodes_per_layer)
        self.learning_rate = learning_rate  # by default
        if activation == 'relu':
            self.activation = np.vectorize(self.relu)
            self.activationDerv = np.vectorize(self.reluDerv)
        elif activation == 'sigmoid':
            self.activation = np.vectorize(self.sigmoid)
            self.activationDerv = np.vectorize(self.sigmoidDerv)
        else:
            raise ValueError('Activation type not understood - '
                             'only relu and sigmoid supported')
        self.initialiseWeights()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerv(self, x):
        """Derivative of the sigmoid function"""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        if x > 0:
            return x
        return 0

    def reluDerv(self, x):
        if x > 0:
            return 1
        return 0

    def initialiseWeights(self):
        """
        Initialise a list of weight matrices corresponding to the
        number of nodes in adjacent layers of the network
        """
        self.weights = []
        for index, nodes in enumerate(self.nodes_per_layer[:-1]):
            nodes_this_layer = nodes
            nodes_next_layer = self.nodes_per_layer[index + 1]
            # Note the dimensions of the weight matrix:
            weight_matrix = np.random.uniform(size=(nodes_next_layer,
                                                    nodes_this_layer))
            self.weights.append(weight_matrix)
        self.initial_weights_copy = self.weights

    def resetWeights(self):
        self.weights = self.initial_weights_copy

    def calculateDeltas(self, input_per_layer, output_per_layer, target):
        """Returns a list of delta vectors for each layer"""
        deltas = [np.array([])] * self.num_layers
        deltas[-1] = (output_per_layer[-1] - target) * \
            self.activationDerv(input_per_layer[-1])

        # Calculate deltas for all other layers, starting from second last
        # and working backwards up to the second layer (all but the input
        # layer will have deltas at the end of this loop):
        for index in range(len(deltas) - 2, 0, -1):
            deltas[index] = \
                np.multiply(self.activationDerv(input_per_layer[index]),
                            np.matmul(self.weights[index].T,
                                      deltas[index + 1]))
        return deltas

    def updateWeights(self, delta_per_layer, output_per_layer):
        for index in range(len(self.weights)):
            weight_delta = self.learning_rate * np.outer(
                delta_per_layer[index + 1],  # deltas of next layer
                output_per_layer[index]  # outputs of previous layer
            )
            self.weights[index] -= weight_delta

    def feedForward(self, x):
        """Feed one training example through the network and generate
        arrays of input and output vectors for each layer"""
        inputs_per_layer = [x]
        outputs_per_layer = [x]
        for index in range(self.num_layers - 1):
            input_vec = np.matmul(self.weights[index],
                                  outputs_per_layer[index])
            output_vec = self.activation(input_vec)
            inputs_per_layer.append(input_vec)
            outputs_per_layer.append(output_vec)
        return inputs_per_layer, outputs_per_layer

    def lastOutput(self, x):
        """Get the output of the last layer for input vector x"""
        _, outputs = self.feedForward(x)
        return outputs[-1]

    def preProcessData(self, D, layer=None):
        # Format input dimensions
        if len(D[0].shape) == 0:
            D = np.array([np.reshape(x, (1, 1)) for x in D])
        elif len(D[0].shape) == 1:
            D = np.array([np.reshape(x, (x.shape[0], 1)) for x in D])
        else:
            raise ValueError('Input to network must be a column vector')

        # Check if data compatible with network
        if layer == 'input':
            num_nodes_compare = self.nodes_per_layer[0]
        elif layer == 'output':  # 'output'
            num_nodes_compare = self.nodes_per_layer[-1]
        else:
            raise ValueError('Pass either \'input\' or \'output\''
                             ' as layer parameter')
        if D[0].shape[0] != num_nodes_compare:
            raise ValueError('Data dimensions must match those of network')

        # Finally, normalise data to range [-1, 1]
        D = D / abs(D).max()

        return D

    def train(self, X, Y):
        """Train the network based on input data"""
        # Pre-processing
        self.Y_scale = abs(Y).max()
        X = self.preProcessData(X, layer='input')
        Y = self.preProcessData(Y, layer='output')

        # Training
        for x, y in zip(X, Y):
            inputs, outputs = self.feedForward(x)
            print('I:\n-----\n{}\nO:\n-----\n{}'
                  .format(inputs, outputs))
            deltas = self.calculateDeltas(inputs, outputs, y)
            self.updateWeights(deltas, outputs)
            print('Deltas:\n-----{}\nWeights:\n-----\n{}\n'
                  .format(deltas, self.weights))
            print('*-----------'*3 + '*\n')

    def test(self, X):
        """Make predictions on test data using weights learned during
        training"""
        X = self.preProcessData(X, layer='input')

        predictions = []
        for x in X:
            predictions.append(self.lastOutput(x))

        # Convert to np array and re-scale
        predictions = np.array(predictions) * self.Y_scale
        return predictions

    def show_plot(X, Y):
        plt.plot(X.flatten(), Y.flatten())
