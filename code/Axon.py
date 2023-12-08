import random
import numpy as np
import Coordinates

class Axon():
    # simulates the axon and its synapse
    def __init__(self, neuron1, neuron2, name, base_space):
        super(Axon, self).__init__()
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        self.name = name
        self.base_space = base_space


    def color_me(self, color="black"):
        value = self.base_space.axon_line_dict[self.name]
        if value[1].active:
            value[0][0].set_color(color)