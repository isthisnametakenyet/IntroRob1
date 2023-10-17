import neuron
import random


class NetworkDyn:
    def __init__(self, init_inputs, num_layers, height_map):
        self.layers = [[]]

        ##LAYER 0
        j = 0
        tmp_layer = []
        while j < height_map[0]:
            weights = []
            for i in range(len(init_inputs)):
                weights.append(random.uniform(0, 1))
                #weights.append(1) #DEBUG

            tmp_layer.append(neuron.Neuron(weights, random.uniform(0, 1)))
            #tmp_layer.append(neuron.Neuron(weights, 1)) #DEBUG
            j = j + 1

        self.layers[0] = tmp_layer

        ##OTHER LAYERS
        j = 1
        while j < num_layers:  # Por capa
            tmp_layer = []

            for n in range(height_map[j]):  # Por neurona de esta capa
                weights = []
                for i in range(height_map[j - 1]):  # Por input de la capa anterior
                    weights.append(random.uniform(0, 1))
                    #weights.append(1) #DEBUG

                tmp_layer.append(neuron.Neuron(weights, random.uniform(0, 1)))
                #tmp_layer.append(neuron.Neuron(weights, 1)) #DEBUG

            self.layers.append(tmp_layer)
            j = j + 1

        #print(self.layers)

    def calc_outputs(self, init_inputs):
        for neur in self.layers[0]:  # Usamos el input de teclado para capa 0
            neur.addInputs(init_inputs)
            neur.calcZ()
            neur.calcSigmoid()
            print("   Layer 0 output: " + str(neur.output))

        l = 1
        while l < (len(self.layers)):  # Por capa
            n = 0
            while n < (len(self.layers[l])):  # Por neurona de esta capa
                newInput = ''
                for i in self.layers[l - 1]:
                    newInput += str(i.output) + ','

                print("Layer " + str(l) + " inputs: " + str(newInput))

                self.layers[l][n].addInputs(newInput.split(','))
                self.layers[l][n].calcZ()
                self.layers[l][n].calcSigmoid()

                n = n + 1
            l += 1

    def calc_error(self, expectation):
        err = []
        i = 0
        for exp in expectation:
            err.append(self.layers[len(self.layers) - 1][i].output - float(exp))
            i += 1

        return err

    def backward_propagation(self, error, learn_rate):
        print("Error: " + str(error))

        l = len(self.layers) - 2
        while l >= 0: #Por cada layer menos la ultima

            for n in self.layers[l]: #Por cada neurona de la layer
                i = 0
                for w in n.weights: #Por cada weight de la neurona
                    #print("W: " + str(w) + " i: " + str(i))
                    n.weights[i] = n.output * float(learn_rate) * float(error[i])
                    i += 1

            l -= 1