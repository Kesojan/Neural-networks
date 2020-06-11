"""
Series of steps to build nerual network from scratch and to gain deeper comprehension
Step 5
Covers forward activation and hidden layer manipulation
Describes current status of a server at a single instance
Objective to pass batch of sample for efficiency
Dataset values should range from -1 to +1 due to neuron configuration
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# 100 feature sets of 3 classes
X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
# defining activation ReLu
activation1 = Activation_ReLU()

layer1.forward(X)

# print layer 1 output
activation1.forward(layer1.output)
print(activation1.output)
