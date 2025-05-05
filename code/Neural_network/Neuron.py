import numpy as np
import random
random.seed(1)
import Neural_network.Axon
import time
import math


class Neuron():
    def __init__(self, coordinate, base_space, output_neuron = False, name = "not_set", bias = 0):
        super(Neuron, self).__init__()
        self.name = name
        self.parent_connections = {} # closer to input
        self.children_connections = {} # closer to output
        self.output = 0
        self.activated = False
        self.output_neuron = output_neuron
        self.calculated_gradient = False
        self.bias = bias
        self.started = False
        self.error_for_output_neuron = 0

        self.coordinate = coordinate
        if name == "not_set":
            self.name = ",".join([str(self.coordinate.x), str(self.coordinate.y), str(self.coordinate.z)]) + str(time.time_ns())
        # self.hash_val = int(''.join(c for c in self.name if c.isdigit()))
        self.base_space = base_space
#        print("Hey, im a neuron!")


    def reset_neuron(self):
        self.output = 0
        self.started = False
        self.activated = False
        self.error_for_output_neuron = 0
        self.delta_error_through_delta_neuron_output = 0
        self.delta_error_through_delta_neuron_net = 0
        self.delta_out_through_delta_net = 0
        self.calculated_gradient = False
        for p in self.parent_connections.keys():
            self.parent_connections[p].new_weights = []
        for c in self.children_connections.keys():
            self.children_connections[c].new_weights = []
        pass

    def reset_neuron_gradient_calculations(self):
        self.started = False
        self.error_for_output_neuron = 0
        self.delta_error_through_delta_neuron_output = 0
        self.delta_error_through_delta_neuron_net = 0
        self.delta_out_through_delta_net = 0
        self.calculated_gradient = False
        for p in self.parent_connections.keys():
            self.parent_connections[p].new_weights = []
        for c in self.children_connections.keys():
            self.children_connections[c].new_weights = []


    def wire(self):
        return
        pkeys = list(self.parent_connections.keys())
        for p in pkeys:
            weight = round(random.uniform(0, 1), 2)
            parent_connection = self.parent_connections[p]
            parent = parent_connection.parent
            axon_name = parent.name + "," + self.name
            axon = Axon.Axon(parent, self, axon_name, self.base_space, weight, [])
            parent.children_connections[self.__hash__()] = axon
            self.parent_connections[axon_name] = axon
        for p in pkeys:
            self.parent_connections.pop(p)


    def gradient_normalisation(self, gradient):
        return gradient
        return max(min(0.5,gradient),-0.5)
        #return ((1. / (1 + np.exp(-gradient)))-0.5)

    def change_weight(self):

        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            if parent_connection.new_weights != []:

                gradient = self.gradient_normalisation(sum(parent_connection.new_weights))

                # print("weight: ", round(parent_connection.get_weight(), 3), " adjust by: ", round(-gradient, 4))
                # if not self.base_space.fast:
                #     #                print("from ", self.name, " to ", parent_connection.parent.name)
                #     print("weight: ", round(parent_connection.get_weight(), 3), " adjust by: ", round(-gradient, 4))
                new_weight = max(min(10,parent_connection.get_weight() - gradient),-10)
                if math.isnan(new_weight):
                    print("pause")

                # print("Axon: ", parent_connection.name, " weight: ", round(parent_connection.get_weight(), 3), " adjust by: ", round(-gradient, 4))
                parent_connection.weight = new_weight
                parent_connection.new_weights = []
                self.parent_connections[p] = parent_connection


    def get_weight(self, parent):
        return self.parent_connections[parent + self].get_weight()

    def gradient_descent(self, learning_rate, depth_counter):
        if self.base_space.verbal:
            print("Start gradient descent to neuron", self.name)

        self.started = True
        # if self.name == "o21":
        #     print("stop")

        #self.delta_error_through_delta_neuron_output = 0

        for c in self.children_connections.keys():
            children_connection = self.children_connections[c]

            if not children_connection.child.calculated_gradient:
                children_connection.child.gradient_descent(learning_rate, depth_counter=depth_counter-1)

            self.delta_error_through_delta_neuron_output += children_connection.child.delta_error_through_delta_neuron_net

        if self.output_neuron:
            self.delta_error_through_delta_neuron_output = self.error_for_output_neuron
        # if self.name == "o21":
        #     print("stop")
        self.calculated_gradient = True

        self.delta_out_through_delta_net = self.deri_activation_function()
        self.delta_error_through_delta_neuron_net = self.delta_error_through_delta_neuron_output * self.delta_out_through_delta_net
        # if self.name == "o22":
        #     pass
            #print("stop")
        #print("computed error/net in neuron: ", self.name, ": ", round(self.delta_error_through_delta_neuron_net,3))



        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            #            if self.name == "h11":
            #                print("stop")
            if self.base_space.Visualization:
                pass
                #self.base_space.axon_line_dict[p + self.name][0][0].set_color("red")

                #error_through_w = self.a_null_w_parent(parent_connection.parent) * self.delta_error_through_delta_neuron_output

            delta_net_through_delta_w = parent_connection.parent.activation()

            # TODO delta_net_through_delta_w is 0 in second output neuron because output is deleted in neuron_reset
            # TODO reset_neuron_all and reset_gradient_calculations
            gradient = self.delta_error_through_delta_neuron_net * delta_net_through_delta_w


            # Appending the gradient
            if gradient != 0:
                if self.base_space.verbal:
                    print("Gradient descent to neuron: ", self.name, " gradient: ", gradient)
                if math.isnan(learning_rate*gradient*depth_counter):
                    print("pause")
                self.parent_connections[p].new_weights.append(learning_rate * gradient * depth_counter)
            #   self.parent_connections[p].new_weights.append(self.gradient_normalisation(learning_rate * gradient))
            if not parent_connection.parent.started:
                parent_connection.parent.gradient_descent(learning_rate = learning_rate, depth_counter = depth_counter + 1)

            if self.base_space.Visualization:
                self.base_space.axon_line_dict[p + self.name][0][0].set_color("gray")

        if not self.output_neuron:
            # pass
            self.bias = max(-1, min(1, self.bias - round((learning_rate * self.delta_error_through_delta_neuron_net),4)))
            if self.base_space.fast:
                print("bias: ", self.bias)

        self.change_weight()
    #        self.reset_neuron()


    def activation_function(self, z):
        if math.isnan(z):
            print("pause")
        return z # linear
        return (1. / (1 + np.exp(-z))) # sigmoid

    def deri_activation_function(self):
        if math.isnan(self.activation()):
            print("pause")
        return 1 # linear
        return self.activation() * (1 - self.activation()) # sigmoid

    def activation(self):
        if self.activated:
            return self.output

        summation = 0
        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            summation += parent_connection.parent.activation() * parent_connection.get_weight()
        self.output = self.activation_function(summation + self.bias)
        self.activated = True
#        if not self.base_space.fast:
#            print(self.name, " activation: ", self.output)
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

    # def __hash__(self):
    #     return self.hash_val

    def color_me(self,color="black"):
        neuron_dict_entry = self.base_space.neuron_dot_dict[self.name]
        neuron_dict_entry[0].set_color(color)
        print("My bias is: ", self.bias)



class Input_Neuron():
    def __init__(self, coordinate, base_space, name = "not_set"):
        super(Input_Neuron, self).__init__()
        self.children_connections = {} # closer to output
        self.output = 0
        self.activated = False
        self.started = False
        self.name = name
        self.bias = 0
        self.input = 0

        self.coordinate = coordinate
        if name == "not_set":
            self.name = ",".join([str(self.coordinate.x), str(self.coordinate.y), str(self.coordinate.z)]) + str(time.time_ns())
        # self.hash_val = int(''.join(c for c in self.name if c.isdigit()))
        self.base_space = base_space

    def reset_neuron(self):
        self.activated = False
        self.output = 0
        self.started = False
        self.input = 0

    def reset_neuron_gradient_calculations(self):

        pass

    def wire(self):
        pass

    def gradient_descent(self, learning_rate, error_for_output_neuron = 0, depth_counter=0):
        pass

    def set_input(self, input):
        if math.isnan(input):
            print("pause")
        self.input = input

    def activation_function(self, z):
        return z
        #return 1. / (1 + np.exp(-z))

    def activation(self):
#        if not self.base_space.fast:
#            print(self.name, " activation: ", self.output)
        self.output = self.activation_function(self.input)
        self.activated = True
        return self.output

    def deri_activation(self):
        return self.output

    # def __hash__(self):
    #     return self.hash_val

    def change_weight(self):
        pass

    def color_me(self,color="black"):
        neuron_dict_entry = self.base_space.neuron_dot_dict[self.name]
        neuron_dict_entry[0].set_color(color)
        print("My bias is: ", self.bias)