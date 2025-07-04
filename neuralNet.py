import math


def sigmoid(num):
    return 1 / (1 + math.exp(-num))


class aNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def calculate(self, inputs):
        total = 0
        for w in range(len(self.weights)):
            total += self.weights[w] * inputs[w]
        total += self.bias
        return sigmoid(total)


weights = [0.5, 0.8]
bias = 2
neuron = aNeuron(weights, bias)

inputs = [3, 4]
output = neuron.calculate(inputs)
print(output)