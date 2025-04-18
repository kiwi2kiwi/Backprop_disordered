import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from random import randrange
from Morphogen_simulation_v2 import *
import Morphogen_simulation_v2.Coordinates
from Morphogen_simulation_v2.Cell_v2 import Cell
import Morphogen_simulation_v2.Rules
import math
import numpy as np
import Morphogen_simulation_v2.Morphogens_v2
import copy
import pickle

size = 100

class Cell_space():

    def __init__(self):
        self.Cells = {}
        self.Cell_counter = 0
        self.Axons = {}
        self.Axon_counter = 0
        self.Morphogens = {}
        self.Morphogen_counter = 0
        self.Rules = {}
        self.Rule_counter = 0
        self.input_cells = []
        self.output_cells = []

        input_coords = self.ordered_input_neurons(2,2)
        output_coords = self.ordered_output_neurons(1,3)

        for i in input_coords:
            new_input_cell = Cell(self, i, input=True)
            self.input_cells.append(new_input_cell)
            self.Cells[new_input_cell.name] = new_input_cell

        for o in output_coords:
            new_output_cell = Cell(self, o, output=True)
            self.output_cells.append(new_output_cell)
            self.Cells[new_output_cell.name] = new_output_cell

        for i in np.arange(5):
            Morphogen_simulation_v2.Morphogens_v2.Morphogens_v2(1, self, cell_unique=False)
            Morphogen_simulation_v2.Rules.Rule(self)


    def neurogenesis(self):
        print("neurogenesis")
        for cell in self.Cells.values():
            cell.develop()

    def input_to_output_debug(self):
        # TODO create a connection from first input to first output
        # get morpho address from first input
        in_morpho = self.input_cells[0].address
        # get morpho address from first output
        out_morpho = self.output_cells[0].address


        demo_rule = Morphogen_simulation_v2.Rules.Rule(self)
        demo_rule.threshold = 0
        demo_rule.morphogen = in_morpho
        demo_rule.target_morphogen = out_morpho
        demo_rule.rule_type = 4



    def ordered_input_neurons(self, height, width):
        global size
        V = []
        area = size - 20
        y_distance = area / height
        z_distance = area / width
        Y = np.arange(-(size / 2) + 10, (size / 2) - 10, y_distance)
        Z = np.arange(-(size / 2) + 10, (size / 2) - 10, z_distance)
        for y in Y:
            for z in Z:
                V.append(Morphogen_simulation_v2.Coordinates.Coordinate(-(size/2), y, z))
        return V

    def ordered_output_neurons(self, height, width):
        global size
        V = []
        area = size - 20
        y_distance = area / height
        z_distance = area / width
        Y = np.arange(-(size/2)+10,(size/2)-10,y_distance)
        Z = np.arange(-(size/2)+10,((size/2)-10),z_distance)
        for y in Y:
            for z in Z:
                V.append(Morphogen_simulation_v2.Coordinates.Coordinate(size/2, y, z))
        return V


    def spawn_cell(self, old_Coordinate, replicate_vector, morphogens):
        self.cell_count += 1
        new_cell = Cell(self,Coordinates.Coordinate(old_Coordinate.x + replicate_vector.x, old_Coordinate.y + replicate_vector.y, old_Coordinate.z + replicate_vector.z), morphogens,0,self.cell_count)
        for key in new_cell.morphogens:
            new_cell.morphogens[key][1] = new_cell.morphogens[key][0].deteriorate(new_cell.morphogens[key][1], new_cell)

        new_cell.coordinate.x = new_cell.coordinate.x + randrange(10) * 0.01 - 0.05
        new_cell.coordinate.y = new_cell.coordinate.y + randrange(10) * 0.01 - 0.05
        new_cell.coordinate.z = new_cell.coordinate.z + randrange(10) * 0.01 - 0.05
        for c2 in self.Cells:
            distance = Coordinates.distance_finder(new_cell.coordinate, c2.coordinate)
            if distance < 0.5:
                vector = [new_cell.coordinate.x - c2.coordinate.x, new_cell.coordinate.y - c2.coordinate.y, new_cell.coordinate.z - c2.coordinate.z]
                norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
                direction = [vector[0] / (norm * 2), vector[1] / (norm * 2), vector[2] / (norm * 2)]
                new_cell.coordinate.x = new_cell.coordinate.x - direction[0]
                new_cell.coordinate.y = new_cell.coordinate.y - direction[1]
                new_cell.coordinate.z = new_cell.coordinate.z - direction[2]
                c2.coordinate.x = c2.coordinate.x + direction[0]
                c2.coordinate.y = c2.coordinate.y + direction[1]
                c2.coordinate.z = c2.coordinate.z + direction[2]
        # self.new_cells.append(new_cell)
        self.Cells.append(new_cell)
        return new_cell


    def start_vis(self):
        self.cells_in_plot = {}
        self.axon_line_dict = {}  # name: (axon, linie auf plot)
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.size = 120
        self.ax.set_xlim(-(self.size / 2), self.size / 2)
        self.ax.set_ylim(-(self.size / 2), self.size / 2)
        self.ax.set_zlim(-(self.size / 2), self.size / 2)
        for c in self.Cells.values():
            self.cells_in_plot[c.name] = [(self.ax.scatter(c.coordinate.x, c.coordinate.y, c.coordinate.z, c="grey",
                                                           s=10)), c]

        for a in self.Axons.values():
            self.axon_line_dict[a.name] = [(self.ax.plot3D([a.parent.coordinate.x, a.child.coordinate.x],
                                                           [a.parent.coordinate.y, a.child.coordinate.y],
                                                           [a.parent.coordinate.z, a.child.coordinate.z], linewidth=1,
                                                           c='grey')), a]

    def draw_image(self):
        for c in self.cells_in_plot.keys():
            self.cells_in_plot[c][0].axes.cla()
        for c in self.Cells.values():  # create cells
            if c.name not in self.cells_in_plot.keys():
                self.cells_in_plot[c.name] = [(self.ax.scatter(c.coordinate.x, c.coordinate.y, c.coordinate.z, c="grey",
                                                               s=30)), c]
            else:
                self.cells_in_plot[c.name][0] = self.ax.scatter(c.coordinate.x, c.coordinate.y, c.coordinate.z,
                                                                c="grey",
                                                                s=30)
        for a in self.Axons.values():  # create cells
            if a.name not in self.axon_line_dict.keys():
                self.axon_line_dict[a.name] = [(self.ax.plot3D([a.parent.coordinate.x, a.child.coordinate.x],
                                                           [a.parent.coordinate.y, a.child.coordinate.y],
                                                           [a.parent.coordinate.z, a.child.coordinate.z], linewidth=1,
                                                           c='grey')), a]
            else:
                self.cells_in_plot[a.name][0] = self.ax.plot3D([a.parent.coordinate.x, a.child.coordinate.x],
                                                           [a.parent.coordinate.y, a.child.coordinate.y],
                                                           [a.parent.coordinate.z, a.child.coordinate.z], linewidth=1,
                                                           c='grey')

        self.ax.set_xlim(-(self.size / 2), self.size / 2)
        self.ax.set_ylim(-(self.size / 2), self.size / 2)
        self.ax.set_zlim(-(self.size / 2), self.size / 2)
        plt.ion()
        # self.fig.savefig('..//Bilder//temp' + str(self.cell_count) + '.png', dpi=self.fig.dpi)

print("hi")
# Cell_space()