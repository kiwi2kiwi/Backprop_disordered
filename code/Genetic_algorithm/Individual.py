# import sys
#
# sys.path.append("..") #Morphogen_simulation_v2

import sys
import os

sys.path.append(os.path.abspath('../Morphogen_simulation_v2'))
import Cell_space
sys.path.append(os.path.abspath('../Neural_network'))
import Neuron_space

# from Morphogen_simulation_v2 import Cell_space

# from Morphogen_simulation_v2 import
class Individual:
    def __init__(self):
        super(Individual, self).__init__()
        print("creation of input and output cells")
        c = Cell_space.Cell_space()
        print("debug 1 manually written rule. Connect from the first input to the first output neuron")
        print("neurogenesis")
        c.neurogenesis()
        print("Creating neural network backbone and importing structure")
        n = Neuron_space.NeuronSpace()
        n.import_network(c)
        print("done")


Individual()



