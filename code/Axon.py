import random
import numpy as np
import Coordinates

class Axon():
    # simulates the axon and its synapse
    def __init__(self, parent, child, name, base_space, weight):
        super(Axon, self).__init__()
        self.parent = parent
        self.child = child
        self.name = name
        self.base_space = base_space
        self.weight = weight

    def get_weight(self):
        return self.weight

    def color_me(self, color="black"):
        value = self.base_space.axon_line_dict[self.name]
        if value[1].active:
            value[0][0].set_color(color)