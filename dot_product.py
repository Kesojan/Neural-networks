import numpy as np
"""
Series of steps to build nerual network from scratch and to gain deeper comprehension
Step 4
Computing using dot product with layer of neurons and numerous inputs
"""
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]
# simple calculation
output = np.dot(weights, inputs) + biases
print(output)
