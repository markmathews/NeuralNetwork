import numpy as np
import matplotlib as plt


class NeuralNetwork:
    """
    A single hidden layer (for now) feed-forward neural network
    """
    def __init__(
        self,
        nodes_per_layer,  # 1D array w/ num. of nodes per layer
        # activation='sigmoid'
    ):
        self.initialiseWeights(nodes_per_layer)
        self.nodes_per_layer = nodes_per_layer
        self.num_layers = len(nodes_per_layer)
        self.learning_rate = 0.01  # by default
        self.activation = self.sigmoid  # for now, only using sigmoid
        self.activationDerv = self.sigmoidDerv

    def sigmoid(self, x):  # TESTED
        return 1 / (1 + np.exp(-x))

    def sigmoidDerv(self, x):  # TESTED
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def initialiseWeights(self, nodes_per_layer):  # TESTED
        """
        Initialise a list of weight matrices corresponding to the
        connections between adjacent layers in the network
        """
        self.weights = []
        for index, nodes in enumerate(nodes_per_layer[:-1]):
            nodes_this_layer = nodes
            nodes_next_layer = nodes_per_layer[index + 1]
            # Note the dimensions of the weight matrix:
            weight_matrix = np.random.uniform(size=(nodes_next_layer,
                                                    nodes_this_layer))
            self.weights.append(weight_matrix)

    def calculateDeltas(self, input_per_layer, output_per_layer, target):
        """Returns a list of delta vectors for each layer"""
        deltas = [np.array([])] * self.num_layers
        deltas[-1] = (target - output_per_layer[-1]) * \
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
        for index in range(0, self.num_layers - 1):
            input_vec = np.matmul(self.weights[index],
                                  outputs_per_layer[index])
            output_vec = self.activation(input_vec)
            inputs_per_layer.append(input_vec)
            outputs_per_layer.append(output_vec)
        return inputs_per_layer, outputs_per_layer

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
        return D

    def train(self, X, Y):
        """Train the network based on input data"""
        X = self.preProcessData(X, layer='input')
        Y = self.preProcessData(Y, layer='output')
        for x, y in zip(X, Y):
            inputs, outputs = self.feedForward(x)
            deltas = self.calculateDeltas(inputs, outputs, y)
            print('Deltas:\n', deltas)
            self.updateWeights(deltas, outputs)
            print('Weights:\n', self.weights)
            print('-----------')

    def test(self, X):
        """Make predictions on test data using weights learned during
        training"""
        X = self.preProcessData(X, layer='input')
        predictions = []
        for x in X:
            _, outputs = self.feedForward(x)
            predictions.append(outputs[-1])  # output of last layer
        return np.array(predictions)
