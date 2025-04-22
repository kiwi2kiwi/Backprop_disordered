# import sys
#
# sys.path.append("..") #Morphogen_simulation_v2

import sys
import os
# /Morphogen_simulation_v2
sys.path.append(os.path.abspath('..'))
import Morphogen_simulation_v2.Cell_space
# sys.path.append(os.path.abspath('../Neural_network'))
import Neural_network.Neuron_space
import Neural_network.nn_execution as nn_exe

# from Morphogen_simulation_v2 import Cell_space

# from Morphogen_simulation_v2 import
class Individual:
    def __init__(self):
        super(Individual, self).__init__()
        print("creation of input and output cells")
        c = Morphogen_simulation_v2.Cell_space.Cell_space()
        print("debug 1 manually written rule. Connect from the first input to the first output neuron")
        c.input_to_output_debug()
        print("neurogenesis")
        c.neurogenesis()
        c.start_vis()
        c.draw_image()
        print("Creating neural network backbone and importing structure")
        n = Neural_network.Neuron_space.NeuronSpace()
        n.import_network(c)
        n.start_vis()
        n.draw_brain()
        print("Training the network")
        nn_exe.running_the_network(n)
        print("trained the network")


Individual()



