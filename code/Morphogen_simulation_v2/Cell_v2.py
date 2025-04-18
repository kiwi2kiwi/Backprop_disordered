import Morphogen_simulation_v2.Coordinates
import copy
import Morphogen_simulation_v2.Morphogens_v2

class Cell():
    def __init__(self, Cell_space, Coordinate, output = False, input = False):
        self.Cell_space = Cell_space
        self.coordinate = Coordinate
        self.children = {}
        self.parents = {}
        self.Axons = {}
        self.address = Morphogen_simulation_v2.Morphogens_v2.Morphogens_v2(1, self.Cell_space, cell_unique=True) # the adress is a unique morphogen that each cell always expresses
        self.address.cells.append(self) # add cell ----------------TODO  to morphogen
        self.name = Cell_space.Cell_counter
        Cell_space.Cell_counter += 1
        self.morphogens = {}
        self.morphogen_counter = 0
        # self.replicate_vector = Coordinates.Coordinate(0, 0, -1)

    def new_morphogens(self, new_morphogen):
        self.morphogens[new_morphogen.name] = new_morphogen
        self.morphogen_counter += 1

    def del_morphogens(self, morphogen): # dont remove morphogens that are unique cell addresses
        if not morphogen.cell_unique:
            self.morphogens.pop(morphogen.name)

    def calc_morphogen(self, morphogen):
        # get distance to all other cells and calculate the morphogen * distance
        # for cell in cell_space.Morphogens[morphogen.name].cells:
        concentration = 0
        for cell in morphogen.cells:
            # if cell.name != self.name:
            distance = max(1, Morphogen_simulation_v2.Coordinates.distance_finder(self.coordinate, cell.coordinate))
            calculated = morphogen.amount/distance # morphogen with distance falloff
            concentration += calculated
        return concentration

    def develop(self):
        for rule in self.Cell_space.Rules.values():
            rule.rule(self)


    # def step(self):
    #
    #
    #     if not self.executed_morphogens:
    #         for key in self.morphogens:
    #             self.morphogens[key][0].rule(self, self.morphogens[key][1])
    #         self.executed_morphogens = True
    #     self.replicate()
    #
    #
    #
    # def replicate(self):
    #     # print("Cell: " + str(self.name))
    #     if self.name == 0:
    #         print("pause")
    #     if self.mitosis_counter != 0:
    #         print("Cell: " + str(self.name) + " replicating")
    #         print(self.morphogens)
    #         self.mitosis_counter -= 1
    #         c = self.Cell_space.spawn_cell(self.Coordinate, self.replicate_vector,copy.deepcopy(self.morphogens))
    #         self.children.append(c)
    #     if self.mitosis_counter == 0:
    #         self.executed_morphogens = True
