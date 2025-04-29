import random
import numpy as np
import Neural_network.Coordinates
import math

class Axon():
    # simulates the axon and its synapse
    def __init__(self, parent, child, name, base_space, weight, new_weights):
        super(Axon, self).__init__()
        self.parent = parent
        self.child = child
        self.name = name
        self.base_space = base_space
        self.weight = weight
        self.new_weights = new_weights
        if math.isnan(self.weight):
            print("pause")

    def get_weight(self):
        if math.isnan(self.weight):
            print("pause")
        return self.weight

    def color_me(self, color="black"):
        value = self.base_space.axon_line_dict[self.name]
        value[0][0].set_color(color)