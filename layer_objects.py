"""
Series of steps to build nerual network from scratch and to gain deeper comprehension
Step 3
Describes current status of a server at a single instance
Objective to pass batch of sample for efficiency
Dataset values should range from -1 to +1 due to neuron configuration
"""
import numpy as np

np.random.seed(0)
X = [[1, 0.4, 0.6, 0.5],
     [0.1, 0.3, -0.8, 0.20],
     [-0.5, -0.7, 0.3, -0.8]]
"""
2 ways to intilalize a layer
Initilize weights first
"""


class LayerDense:
    def init(self, n_inputs, n_neurons):
        # parameter is size of input and neuron quantity
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


"""
1st parameter is input size
2nd parameter is # of neurons(does not matter)
"""
layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 2)

layer1.forward(X)

# printing layer 1
layer2.forward(layer1.output)
print(layer2.output)
