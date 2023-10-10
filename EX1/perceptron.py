import neuron
import random
import numpy


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
    # print(weights1)

    bias1 = random.uniform(0, 1)
    bias2 = random.uniform(0, 1)

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

    z1Neuron = neuron.Neuron(weights1, random.uniform(0, 1))
    z2Neuron = neuron.Neuron(weights2, random.uniform(0, 1))
    z3Neuron = neuron.Neuron(weights3, random.uniform(0, 1))

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


def DynamicNetwork():
    inputs = input("Input the inputs: ")
    inputs = inputs.split(',')

    width = int(input("Num Layers: "))
    height = []

    i = 0
    while (i < width):
        height.append(int(input("Num Neurons Layer " + str(i) + ": ")))
        i = i + 1

    layers = [[]]

    ##LAYER 0
    j = 0
    tmp_layer = []
    while (j < height[0]):
        weights = []
        #print(j)
        for i in range(len(inputs)):
            weights.append(random.uniform(0, 1))
            #print(inputs)

        tmp_layer.append(neuron.Neuron(weights, random.uniform(0, 1)))
        j = j + 1

    layers[0] = tmp_layer

    #print(str(layers[0][0].weights))
    #print(str(layers[0][1].weights))
    #print(str(layers[0][2].weights))

    ##OTHER LAYERS
    j = 1
    while (j < width):  # Por capa
        tmp_layer = []

        for n in range(height[j]):  # Por neurona de esta capa
            weights = []
            for i in range(height[j - 1]):  # Por input de la capa anterior
                weights.append(random.uniform(0, 1))

            tmp_layer.append(neuron.Neuron(weights, random.uniform(0, 1)))

        layers.append(tmp_layer)
        j = j + 1

    #print(layers)
    #print('Weights de 0-0 : ' + str(layers[0][0].weights) + '  Inputs:' + str(inputs))


    ##CALC OUTPUTS
    for neur in layers[0]: # Usamos el input de teclado para capa 0
        neur.addInputs(inputs)
        neur.calcZ()
        neur.calcSigmoid()
        #print("   Layer 0 output: " + str(neur.output))

    l = 1
    while (l < width):  # Por capa
        n = 0
        while (n < height[l]): # Por neurona de esta capa
            newInput = ''
            for i in layers[l-1]:
                newInput += str(i.output) + ','

            print("Layer " + str(l) + " inputs: " + str(newInput))

            layers[l][n].addInputs(newInput.split(','))
            layers[l][n].calcZ()
            layers[l][n].calcSigmoid()

            #print("   Layer " + str(l) + " output: " + str(layers[l][n].output))
            n = n + 1
        l += 1

    ##FINAL OUTPUT
    j = 0
    result = []
    for i in layers[width - 1]:
        print("Final Output Neuron " + str(j) + ": " + str(i.output))
        result.append(i.output)
        j += 1

    return result

if __name__ == '__main__':
    expectation = input("Expected OUtput: ")
    expectation = expectation.split(',')

    # SingleNeuron()
    # ClasificadorNetwork()
    result = DynamicNetwork()

    err = []
    i = 0
    for r in result:
        err.append(r - float(expectation[i]))
        i += 1

    print("Error margin: " + str(err))