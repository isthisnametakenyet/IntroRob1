import neuron
import random

def SingleNeuron():
    inputs = input("Input the inputs: ")
    weights = input("Enter the weights: ")
    bias = int(input("Enter the bias: "))

    inputs = inputs.split(',')
    weights = weights.split(',')

    thisNeuron = neuron.Neuron(weights, bias)
    thisNeuron.addInputs(inputs)
    thisNeuron.calcZ()

    func = input("Use Sigmoid Function? ")
    if func == 'y':
        thisNeuron.calcSigmoid()
    else:
        thisNeuron.calcMax()
    print('Output: ' + str(thisNeuron.output))
    print('Loss: ' + str(thisNeuron.loss))

def ClasificadorNetwork():
    inputs = input("Input the inputs: ")
    inputs = inputs.split(',')

    ##LAYER 1

    weights1 = []
    weights2 = []
    for i in range(len(inputs)):
        weights1.append(random.uniform(0, 1))
        weights2.append(random.uniform(0, 1))
    #print(weights1)

    bias1 = random.uniform(0,1)
    bias2 = random.uniform(0,1)

    x1Neuron = neuron.Neuron(weights1, bias1)
    x2Neuron = neuron.Neuron(weights2, bias2)

    x1Neuron.addInputs(inputs)
    x1Neuron.calcZ()
    x1Neuron.calcSigmoid()

    x2Neuron.addInputs(inputs)
    x2Neuron.calcZ()
    x2Neuron.calcSigmoid()

    ##LAYER 2

    weights1 = []
    weights2 = []
    weights3 = []
    for i in range(2):
        weights1.append(random.uniform(0, 1))
        weights2.append(random.uniform(0, 1))
        weights3.append(random.uniform(0, 1))

    z1Neuron = neuron.Neuron(weights1, random.uniform(0,1))
    z2Neuron = neuron.Neuron(weights2, random.uniform(0,1))
    z3Neuron = neuron.Neuron(weights3, random.uniform(0,1))

    zInput = str(x1Neuron.output) + ',' + str(x2Neuron.output)
    zInput = zInput.split(',')

    z1Neuron.addInputs(zInput)
    z1Neuron.calcZ()
    z1Neuron.calcSigmoid()

    z2Neuron.addInputs(zInput)
    z2Neuron.calcZ()
    z2Neuron.calcSigmoid()

    z3Neuron.addInputs(zInput)
    z3Neuron.calcZ()
    z3Neuron.calcSigmoid()


    ##LAYER 3

    weights4 = []
    weights5 = []
    for i in range(2):
        weights4.append(random.uniform(0, 1))
        weights5.append(random.uniform(0, 1))

    z4Neuron = neuron.Neuron(weights4, random.uniform(0, 1))
    z5Neuron = neuron.Neuron(weights5, random.uniform(0, 1))

    zInput = str(z1Neuron.output) + ',' + str(z2Neuron.output) + ',' + str(z3Neuron.output)
    zInput = zInput.split(',')

    z4Neuron.addInputs(zInput)
    z4Neuron.calcZ()
    z4Neuron.calcSigmoid()

    z5Neuron.addInputs(zInput)
    z5Neuron.calcZ()
    z5Neuron.calcSigmoid()

    print("OUTPUT X1: " + str(x1Neuron.output))
    print("OUTPUT X2: " + str(x2Neuron.output))
    print("OUTPUT Z1: " + str(z1Neuron.output))
    print("OUTPUT Z2: " + str(z2Neuron.output))
    print("OUTPUT Z3: " + str(z3Neuron.output))
    print("OUTPUT Z4: " + str(z4Neuron.output))
    print("OUTPUT Z5: " + str(z5Neuron.output))

if __name__ == '__main__':
    #SingleNeuron()
    ClasificadorNetwork()

