import neuron

if __name__ == '__main__':
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
