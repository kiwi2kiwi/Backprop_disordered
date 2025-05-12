import Morphogen_simulation_v2.Coordinates
import copy
import Morphogen_simulation_v2.Morphogens_v2
import numpy as np

class Cell():
    def __init__(self, cell_space, Coordinate, output = False, input = False):
        self.name = cell_space.Cell_counter # len(Cell_space.Cells.keys())
        cell_space.Cell_counter += 1
        self.cell_space = cell_space
        self.cell_space.Cells[self.name] = self
        self.coordinate = Coordinate
        self.children = {}
        self.parents = {}
        self.Axons = {}
        self.address = Morphogen_simulation_v2.Morphogens_v2.Morphogens_v2(1, self.cell_space, cell_unique=True) # the adress is a unique morphogen that each cell always expresses
        self.address.add_cell(self.name) # TODO do this after the cell is added to the cell_space cells
        self.morphogens = {}
        self.output = output
        self.input = input
        self.replication_counter = 0
        self.integrated = False
        self.integrated_checking = False
        # self.replicate_vector = Coordinates.Coordinate(0, 0, -1)

    def new_morphogens(self, new_morphogen_name):
        self.morphogens[new_morphogen_name] = self.cell_space.Morphogens[new_morphogen_name]

    def del_morphogen(self, morphogen_name): # dont remove morphogens that are unique cell addresses
        if not self.cell_space.Morphogens[morphogen_name].cell_unique and morphogen_name in self.morphogens.keys():
            try:
                self.morphogens[morphogen_name].cells.pop(self.name)
                self.morphogens.pop(morphogen_name)
            except:
                print(morphogen_name in self.morphogens.keys())
                temp_test = self.morphogens[morphogen_name]
                print("stop")

    def calc_morphogen(self, morphogen_name):
        # get distance to all other cells and calculate the morphogen * distance
        concentration = 0
        morphogen = self.cell_space.Morphogens[morphogen_name]
        for cell_name in morphogen.cells.keys():
            cell = self.cell_space.Cells[cell_name]
            if cell.name != self.name:
                distance = max(1, Morphogen_simulation_v2.Coordinates.distance_finder(self.coordinate, cell.coordinate))
                calculated = (morphogen.amount/np.log(distance)) # morphogen with distance falloff
                concentration += calculated
        return concentration

    def develop(self):
        for rule in self.cell_space.Rules.values():
            rule.rule(self)

    # this markes all cells that are reachable by the neural network, so that we don't consider useless neurons in the computation
    def check_integration(self):
        self.integrated_checking = True
        if self.input == True:
            self.integrated = True
            return True
        else:
            for cell in self.parents.values():
                if not cell.integrated_checking:
                    if cell.check_integration():
                        self.integrated = True
            return self.integrated