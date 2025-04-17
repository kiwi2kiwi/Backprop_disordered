# import sys
#
# sys.path.append("..") #Morphogen_simulation_v2

import sys
import os

sys.path.append(os.path.abspath('../Morphogen_simulation_v2'))
import Cell_space

# from Morphogen_simulation_v2 import Cell_space

# from Morphogen_simulation_v2 import
class Individual:
    def __init__(self):
        super(Individual, self).__init__()
        print("creation of input and output cells")
        c = Cell_space.Cell_space()
        print("neurogenesis")
        c.neurogenesis()
        print("done")

Individual()



