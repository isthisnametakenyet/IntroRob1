#!/usr/bin/env python3

import neuron
import random


class NetworkDyn:
    def __init__(self, init_inputs, num_layers, height_map, expectation):
        self.layers = [[]]

        ##LAYER 0
        j = 0
        tmp_layer = []
        while j < height_map[0]:
            weights = []
            for i in range(len(init_inputs)):
                weights.append(random.uniform(0, 1))
                # weights.append(1) #DEBUG

            tmp_layer.append(neuron.Neuron(weights, random.uniform(0, 1)))
            # tmp_layer.append(neuron.Neuron(weights, 1)) #DEBUG
            j = j + 1

        self.layers[0] = tmp_layer


        ##OTHER LAYERS
        j = 1
        while j < num_layers:  # Por capa
            print("j: " + str(j) + " num_layers: " + str(num_layers))
            tmp_layer = []
            
            for n in range(height_map[j]):  # Por neurona de esta capa
                weights = []
                for i in range(height_map[j - 1]):  # Por input de la capa anterior
                    weights.append(random.uniform(0, 1))
                    # weights.append(1) #DEBUG

                tmp_layer.append(neuron.Neuron(weights, random.uniform(0, 1)))
                # tmp_layer.append(neuron.Neuron(weights, 1)) #DEBUG

            self.layers.append(tmp_layer)
            print("Height: " + str(height_map[j]) + " ")

            j = j + 1
        

        ##ULTIMA LAYER
        j = num_layers
        tmp_layer = []
        
        for n in range(len(expectation)):  # Por neurona de esta capa
            weights = []
            for i in range(height_map[j-1]):  # Por input de la capa anterior
                weights.append(random.uniform(0, 1))
                # weights.append(1) #DEBUG

            tmp_layer.append(neuron.Neuron(weights, random.uniform(0, 1)))
            # tmp_layer.append(neuron.Neuron(weights, 1)) #DEBUG

        self.layers.append(tmp_layer)

        # print(self.layers)

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

                # print("Layer " + str(l) + " inputs: " + str(newInput))

                self.layers[l][n].addInputs(newInput.split(','))
                self.layers[l][n].calcZ()
                self.layers[l][n].calcSigmoid()
                print("   Layer " + str(l) +  " output: " + str(self.layers[l][n].output))

                n = n + 1
            l += 1

    def calc_error(self, expectation):
        err = []
        i = 0
        print("Output: ")
        for exp in expectation:
            print(str(self.layers[len(self.layers) - 1][i].output))
            error_tmp = self.layers[len(self.layers) - 1][i].output - float(exp)
            err.append(error_tmp)
            i += 1
            # print("Exp: " + str(exp) + " Err: " + str(error_tmp))

        return err

    def backward_propagation(self, error, learn_rate):
        print("Error: " + str(error))

        #print("L LEN : " + str(len(self.layers) - 1))

        l = len(self.layers) - 1
        print("NumL: " + str(len(self.layers)) + " Weigths L 0: " + str(len(self.layers[0][0].weights)) + " Weigths L 1: " + str(len(self.layers[1][0].weights)))
        while l >= 0:
            numW = len(self.layers[l][0].weights) # Numero de weights de cada neurona de la layer
            print("NumW: " + str(numW))
            for w in range(0, numW):  # Por cada Weight de cada Neurona de la layer
                numN = 0
                for n in self.layers[l]:  # Por cada neurona de la layer
                    print("Layer: " + str(l) + " Weight: " + str(w) + " Neuron: " + str(n))
                    if l == len(self.layers) - 1:  # Si es la ultima layer
                        # print("Neu Weights: " + str(len(n.weights)) + " Errors: " + str(len(error)))
                        n.weights[w] = n.weights[w] - (float(learn_rate) * float(error[numN]) * n.output)
                        n.z = float(error[numN])
                        n.bias = float(n.bias) - (float(learn_rate) * float(n.z))
                        numN += 1
                    else:
                        z2 = 0
                        i = 0
                        for n2 in self.layers[l+1]:
                            z2 += n2.z * n.weights[i]
                            i += 1
                        n.z = z2 * (1 - n.output) * n.output
                        n.bias = float(n.bias) - (float(learn_rate) * float(n.z))
                        n.weights[w] = n.weights[w] - (n.z * float(learn_rate) * n.output)
                        print("W: " + str(n.weights[w]) + " Z: " + str(n.z) + " B: " + str(n.bias))
            l -= 1

