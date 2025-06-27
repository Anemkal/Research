import math


def sigmoid(num):
    return 1 / (1 + math.exp(-num))


def sigmoidDer(sigOutput):
    return sigOutput * (1 - sigOutput)


class aNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def calculate(self, inputs):
        self.inputs = inputs
        total = 0
        for w in range(len(self.weights)):
            total += self.weights[w] * inputs[w]
        total += self.bias
        self.output = sigmoid(total)
        return self.output

    def updateWeights(self, gradient, learnRate):
        for w in range(len(self.weights)):
            self.weights[w] -= learnRate * gradient * self.inputs[w]
        self.bias -= learnRate * gradient

#weights = [0.5, 0.8]
#bias = 2
#neuron = aNeuron(weights, bias)

hidden1 = aNeuron([0.5, 0.8], 2)
hidden2 = aNeuron([-0.2,-0.4], 1.5)
outputNeuron = aNeuron([1.0, -1.0],0.5)

for epoch in range(1000):
    inputs = [3, 4]
    target = 1
    output1 = hidden1.calculate(inputs)
    output2 = hidden2.calculate(inputs)
    finalOutput = outputNeuron.calculate([output1, output2])

    # backpropagation
    outputGrad = finalOutput - target
    error = finalOutput - target
    #outputGrad = error * sigmoidDer(finalOutput)
    hidden1Grad = outputGrad * outputNeuron.weights[0] * sigmoidDer(output1)
    hidden2Grad = outputGrad * outputNeuron.weights[1] * sigmoidDer(output2)

    outputNeuron.updateWeights(outputGrad, learnRate=0.1)
    hidden1.updateWeights(hidden1Grad, learnRate=0.1)
    hidden2.updateWeights(hidden2Grad, learnRate=0.1)


out1 = hidden1.calculate(inputs)
out2 = hidden2.calculate(inputs)
finalOutput = outputNeuron.calculate([out1, out2])

print("final output:", finalOutput)
print("hidden neuron 1 weights:", hidden1.weights, "bias:", hidden1.bias)
print("hidden neuron 2 weights:", hidden2.weights, "bias:", hidden2.bias)
print("output neuron weights:", outputNeuron.weights, "bias:", outputNeuron.bias)