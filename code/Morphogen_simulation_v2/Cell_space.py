import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from random import randrange
import Coordinates
from Cell_v2 import Cell
import Rules
import math
import numpy as np
import Morphogens_v2
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
            new_input_cell = Cell(self, i)
            self.input_cells.append(new_input_cell)
            self.Cells[new_input_cell.name] = new_input_cell

        for o in output_coords:
            new_output_cell = Cell(self, o)
            self.output_cells.append(new_output_cell)
            self.Cells[new_output_cell.name] = new_output_cell

        self.start_vis()
        self.draw_image()

    def neurogenesis(self):
        for cell in self.Cells.values():
            cell.develop()

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
                V.append(Coordinates.Coordinate(-(size/2), y, z))
        return V

    def ordered_output_neurons(self, height, width):
        global size
        V = []
        area = size - 20
        y_distance = area / height
        z_distance = area / width
        Y = np.arange(-(size/2)+10,(size/2)-10,y_distance)
        Z = np.arange(-(size/2)+10,(size/2)-10,z_distance)
        for y in Y:
            for z in Z:
                V.append(Coordinates.Coordinate(size/2, y, 0))
        return V


    def spawn_cell(self, old_Coordinate, replicate_vector, morphogens):
        self.cell_count += 1
        new_cell = Cell(self,Coordinates.Coordinate(old_Coordinate.x + replicate_vector.x, old_Coordinate.y + replicate_vector.y, old_Coordinate.z + replicate_vector.z), morphogens,0,self.cell_count)
        for key in new_cell.morphogens:
            new_cell.morphogens[key][1] = new_cell.morphogens[key][0].deteriorate(new_cell.morphogens[key][1], new_cell)

        new_cell.Coordinate.x = new_cell.Coordinate.x + randrange(10) * 0.01 - 0.05
        new_cell.Coordinate.y = new_cell.Coordinate.y + randrange(10) * 0.01 - 0.05
        new_cell.Coordinate.z = new_cell.Coordinate.z + randrange(10) * 0.01 - 0.05
        for c2 in self.Cells:
            distance = Coordinates.distance_finder(new_cell.Coordinate, c2.Coordinate)
            if distance < 0.5:
                vector = [new_cell.Coordinate.x - c2.Coordinate.x, new_cell.Coordinate.y - c2.Coordinate.y, new_cell.Coordinate.z - c2.Coordinate.z]
                norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
                direction = [vector[0] / (norm * 2), vector[1] / (norm * 2), vector[2] / (norm * 2)]
                new_cell.Coordinate.x = new_cell.Coordinate.x - direction[0]
                new_cell.Coordinate.y = new_cell.Coordinate.y - direction[1]
                new_cell.Coordinate.z = new_cell.Coordinate.z - direction[2]
                c2.Coordinate.x = c2.Coordinate.x + direction[0]
                c2.Coordinate.y = c2.Coordinate.y + direction[1]
                c2.Coordinate.z = c2.Coordinate.z + direction[2]
        # self.new_cells.append(new_cell)
        self.Cells.append(new_cell)
        return new_cell


    def start_vis(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.size = 10
        self.ax.set_xlim(-(self.size / 2), self.size / 2)
        self.ax.set_ylim(-(self.size / 2), self.size / 2)
        self.ax.set_zlim(-(self.size / 2), self.size / 2)
        for c in self.Cells:
            self.cells_in_plot[c.name] = [(self.ax.scatter(c.Coordinate.x, c.Coordinate.y, c.Coordinate.z, c="grey",
                                                             s=10)), c]

    def draw_image(self):
        for c in self.cells_in_plot.keys():
            self.cells_in_plot[c][0].axes.cla()
        for c in self.Cells:  # create cells
            if c.name not in self.cells_in_plot.keys():
                self.cells_in_plot[c.name] = [(self.ax.scatter(c.Coordinate.x, c.Coordinate.y, c.Coordinate.z, c="grey",
                                                             s=30)), c]
            else:
                self.cells_in_plot[c.name][0] = self.ax.scatter(c.Coordinate.x, c.Coordinate.y, c.Coordinate.z, c="grey",
                                                             s=30)

        self.ax.set_xlim(-(self.size / 2), self.size / 2)
        self.ax.set_ylim(-(self.size / 2), self.size / 2)
        self.ax.set_zlim(-(self.size / 2), self.size / 2)
        plt.ion()
        self.fig.savefig('..//Bilder//temp' + str(self.cell_count) + '.png', dpi=self.fig.dpi)

Cell_space()