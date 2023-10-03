import sys
import math


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.tmp = []
        self.z = 0
        self.output = 0
        self.loss = 0

    def addInputs(self, inputs):
        a = 0
        for i in self.weights:
            # print('i: ' + i)
            # print('inpt: ' + inputs[a])
            self.tmp.append(int(i) * int(inputs[a]))
            a = a + 1

    def calcZ(self):
        for i in self.tmp:
            self.z = self.z + i
        # print(self.z)
        self.z + self.bias

    def calcSigmoid(self):
        self.output = 1 / (1 + (math.exp(1) ** -self.z))

    def calcMax(self):
        max(0.1 * self.z, self.z)

    def calcLoss(self):
        # funcion de Loss, derivar J(a,y)
        self.loss = self.output - 0
