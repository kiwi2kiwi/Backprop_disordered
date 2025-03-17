import Coordinates
import copy

class Cell():
    def __init__(self, Cell_space, Coordinate, mitosis_counter, name, excite_inhibit):
        self.Cell_space = Cell_space
        self.Coordinate = Coordinate
        self.children = []
        self.parents = []
        self.address = ""
        self.rules = []
        self.excite = excite_inhibit
        self.name = name
        self.morphogens = []


        self.replicate_vector = Coordinates.Coordinate(0, 0, -1)
        self.mitosis_counter = mitosis_counter
        self.executed_morphogens = False


    def step(self):
        #TODO gather all morphogens from surrounding are and divide them by distance

        if not self.executed_morphogens:
            for key in self.morphogens:
                self.morphogens[key][0].rule(self, self.morphogens[key][1])
            self.executed_morphogens = True
        self.replicate()

    def connect(self, child):

        c = self.Cell_space.spawn_cell(self.Coordinate, self.replicate_vector, copy.deepcopy(self.morphogens))
        self.children.append(c)

    def replicate(self):
        # print("Cell: " + str(self.name))
        if self.name == 0:
            print("pause")
        if self.mitosis_counter != 0:
            print("Cell: " + str(self.name) + " replicating")
            print(self.morphogens)
            self.mitosis_counter -= 1
            c = self.Cell_space.spawn_cell(self.Coordinate, self.replicate_vector,copy.deepcopy(self.morphogens))
            self.children.append(c)
        if self.mitosis_counter == 0:
            self.executed_morphogens = True
