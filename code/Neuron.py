import numpy as np
import random
import Axon
import time


class Neuron():
    def __init__(self, coordinate, base_space, output_neuron = False, name = "not_set"):
        super(Neuron, self).__init__()
        #        self.name = name
        self.parent_connections = {} # closer to input
        self.children_connections = {} # closer to output
        self.output = 0
        self.activated = False
        self.output_neuron = output_neuron

        self.coordinate = coordinate
        self.name = ",".join([str(self.coordinate.x), str(self.coordinate.y), str(self.coordinate.z)]) + str(time.time_ns())
        self.hash_val = int(''.join(c for c in self.name if c.isdigit()))
        self.base_space = base_space
        print("Hey, im a neuron!")

    def reset_neuron(self):
        self.output = 0
        self.activated = False
        for p in self.parent_connections.keys():
            self.parent_connections[p].new_weights = []
        for c in self.children_connections.keys():
            self.children_connections[c].new_weights = []


    def wire(self):
        return
        #weight = 0.8
        pkeys = list(self.parent_connections.keys())
        for p in pkeys:
            weight = round(random.uniform(0, 1), 2)
            parent_connection = self.parent_connections[p]
            parent = parent_connection.parent
            axon_name = parent.name + "," + self.name
            axon = Axon.Axon(parent,self, axon_name, self.base_space, weight, [])
            parent.children_connections[self.__hash__()] = axon
            self.parent_connections[axon_name] = axon
        for p in pkeys:
            self.parent_connections.pop(p)


    def gradient_normalisation(self, gradient):
        #return gradient
        return max(min(0.5,gradient),-0.5)
        #return ((1. / (1 + np.exp(-gradient)))-0.5) * 1

    def change_weight(self):
        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            parent_connection.weight = parent_connection.get_weight() - self.gradient_normalisation(sum(parent_connection.new_weights))
            self.parent_connections[p] = parent_connection

    #            parent = parent_connection.parent
    #            new_weight_to_parent = parent_connection.get_weight() - sum(parent_connection.new_weights)
    #            parent.children_connections[self.__hash__()] = [self, new_weight_to_parent, [new_weight_to_parent]]
    #            self.parent_connections[p] = [parent, new_weight_to_parent, [new_weight_to_parent]]


    def get_weight(self, parent):
        return self.parent_connections[parent + self].get_weight()

    def gradient_descent(self, bis_hier, learning_rate):

        ab_hier = self.a_null_a_eins() * bis_hier
        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            self.base_space.axon_line_dict[p + self.name][0][0].set_color("red")

            self.parent_connections[p].parent.gradient_descent(ab_hier, learning_rate)

            self.base_space.axon_line_dict[p + self.name][0][0].set_color("gray")
            error_durch_w = self.a_null_w_parent(parent_connection.parent) * bis_hier

            self.parent_connections[p].new_weights.append(self.gradient_normalisation(learning_rate * error_durch_w))

            print("weight: ", round(parent_connection.get_weight(),2), " adjust: ", round(sum(parent_connection.new_weights),2))
        self.change_weight()
    #        self.reset_neuron()


    def activation_function(self, z):
        return z

    #        return 1. / (1 + np.exp(-z))

    def deri_activation_function(self, z):
        return z

    def activation(self):
        if self.activated:
            return self.output

        summation = 0
        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            summation += parent_connection.parent.activation() * parent_connection.get_weight()
        self.output = self.activation_function(summation)
        self.activated = True
        return self.output

    def deri_activation(self):
        summation = 0
        for p in self.parent_connections.keys():
            summation += self.parent_connections[p].deri_activation()
        return self.deri_activation_function(summation)

    def a_null_w_parent(self, parent):
        return self.deri_activation_function(parent.activation())

    def a_null_a_eins(self):
        summation = 0
        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            summation += parent_connection.get_weight()
        return self.deri_activation_function(summation)

    def __hash__(self):
        return self.hash_val

#    def color_me(self,color="red"):



class Input_Neuron():
    def __init__(self, coordinate, base_space, name = "not set"):
        super(Input_Neuron, self).__init__()
        self.children_connections = {} # closer to output
        self.output = 0
        self.activated = False
#        self.name = name

        self.coordinate = coordinate
        self.name = ",".join([str(self.coordinate.x), str(self.coordinate.y), str(self.coordinate.z)]) + str(time.time_ns())
        self.hash_val = int(''.join(c for c in self.name if c.isdigit()))
        self.base_space = base_space

    def reset_neuron(self):
        pass

    def wire(self):
        pass

    def gradient_descent(self, a, b):
        pass

    def set_input(self, input):
        self.activated = True
        self.output = input

    def activation_function(z):
        return 1. / (1 + np.exp(-z))

    def activation(self):
        return self.output

    def deri_activation(self):
        return self.output

    def reset_neuron(self):
        self.output = 0
        self.activated = False

    def __hash__(self):
        return self.hash_val