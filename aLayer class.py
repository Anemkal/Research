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


class aLayer:
    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate(inputs))
        return outputs

    def backward(self, gradients, learnRate):
        for i in range(len(self.neurons)):
            self.neurons[i].updateWeights(gradients[i], learnRate)


data = [
    ([3, 4], 1),
    ([1, 5], 1),
    ([0, 0], 0),
    ([2, -3], 0)
]

hidden1 = aNeuron([0.5, 0.8], 2)
hidden2 = aNeuron([-0.2, -0.4], 1.5)
hiddenLayer = aLayer([hidden1, hidden2])
outputNeuron = aNeuron([1.0, -1.0], 0.5)

# training loop
for epoch in range(1000):
    for inputs, target in data:
        output1, output2 = hiddenLayer.forward(inputs)
        finalOutput = outputNeuron.calculate([output1, output2])

        # backpropagation
        outputGrad = finalOutput - target
        hidden1Grad = outputGrad * outputNeuron.weights[0] * sigmoidDer(output1)
        hidden2Grad = outputGrad * outputNeuron.weights[1] * sigmoidDer(output2)

        outputNeuron.updateWeights(outputGrad, learnRate=0.1)
        hiddenLayer.backward([hidden1Grad, hidden2Grad], learnRate=0.1)


print("\nFinal outputs after training:")
for inputs, target in data:
    out1, out2 = hiddenLayer.forward(inputs)
    final = outputNeuron.calculate([out1, out2])
    print(f"Input: {inputs}, Predicted: {round(final, 3)} Target: {target}")

#print("final output:", finalOutput)
#print("hidden neuron 1 weights:", hidden1.weights, "bias:", hidden1.bias)
#print("hidden neuron 2 weights:", hidden2.weights, "bias:", hidden2.bias)
#print("output neuron weights:", outputNeuron.weights, "bias:", outputNeuron.bias)