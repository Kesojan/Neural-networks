""" 
Series of steps to build nerual network from scratch and to gain deeper comprehension
Step 1:
neuron properties, multilayer preception model
previous neuron nodes output values --> current input
"""
inputs = [4.1, 5.1, 2.1]
weights = [0.5, -0.1, 0.8]
bias = 2

# first step: input * weight + bias
ouput = inputs[0]*weights[0] + inputs[1]*weights[1]+inputs[2]*weights[2]
output = ouput+bias
print(output)
