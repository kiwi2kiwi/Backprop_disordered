def make_two_children_below(cell, amount):
    if amount >= 0.2:
        cell.mitosis_counter = 2
        if cell.name == 0:
            print("2 mitosis for cell 0")

    if amount < 0.2 and amount > 0:
        cell.morphogens["leg"][1] = 0.5



def elongate_down(cell, amount):
    if amount >= 0.2:
        cell.mitosis_counter = 1

import Coordinates
import Cell_v2
import Axon
import random
class Rule():
    def __init__(self, address, executing_cell, cell_space):
        self.address = address
        self.executing_cell = executing_cell
        self.cell_space = cell_space


        # These variables can be mutated
        self.new_coordinate_shift = Coordinates.Coordinate(1,0,0)
        self.threshold = 0.5
        self.morphogen = random.choice(self.cell_space.Morphogens)# pick first morphogen from all morphogens
        self.inhibit_excite_type = 1 # 1 = excite, 0 = inhibit
        self.child_limit = 0
        self.rule_type = 3


    def mutate(self, mutation_rate):
        # TODO mutate one of the mutate-able variables by the mutation rate
        """
                Mutates one of the mutable variables by a small random amount based on mutation_rate.
                """
        mutation_targets = [
            "new_coordinate_shift",
            "threshold",
            "morphogen",
            "inhibit_excite_type",
            "child_limit",
            "rule_type",
            "delete_rule"
        ]

        # Pick a random attribute to mutate
        target = random.choice(mutation_targets)

        if target == "new_coordinate_shift":
            # Mutate coordinate shift with a small random change
            self.new_coordinate_shift.x += random.uniform(-mutation_rate, mutation_rate)
            self.new_coordinate_shift.y += random.uniform(-mutation_rate, mutation_rate)
            self.new_coordinate_shift.z += random.uniform(-mutation_rate, mutation_rate)

        elif target == "threshold":
            # Mutate threshold but keep it between 0 and 1
            self.threshold = max(0, min(1, self.threshold + random.uniform(-mutation_rate, mutation_rate)))

        elif target == "morphogen":
            # Pick a new morphogen randomly
            self.morphogen = random.choice(self.cell_space.Morphogens)

        elif target == "inhibit_excite_type":
            # Flip between 0 and 1 with some probability
            if random.random() < mutation_rate:
                self.inhibit_excite_type = 1 - self.inhibit_excite_type

        elif target == "child_limit":
            # Change child limit within a reasonable range (0 to 10)
            self.child_limit = max(0, self.child_limit + random.randint(-1, 1))

        elif target == "rule_type":
            # Change rule type within a range (1 to 5, for example)
            self.rule_type = max(1, min(5, self.rule_type + random.randint(-1, 1)))

        elif target == "delete_rule":
            # Small chance of deleting the rule
            if random.random() < mutation_rate:
                self.is_deleted = True

    # i tried to make a function that is customizable by the genetic algorithm
    # this function should be modifyable by the genetic algorithm and perform various tasks such as creating an action when a morphogen threshold is reached.
    def rule(self):
            # 2 types of actions
            # 1. send out new morphogen
            # 2. if child cells above threshold, stop creating new ones
            # 3. create Axon to cell with morphogen x
            #

        if self.rule_type == 1:
            self.executing_cell.morphogens.append()
        if self.rule_type == 2:
            if self.morphogen >= self.threshold: # TODO dont to this via child number but via summed morphogens or nearby cells
                if self.child_limit < len(self.executing_cell.children):
                    self.create_new_cell()
        if self.rule_type == 3:
            if self.morphogen < self.threshold: # TODO dont to this via child number but via summed morphogens or nearby cells
                if self.child_limit < len(self.executing_cell.children):
                    self.create_new_cell()
        if self.rule_type == 4:
            if self.morphogen >= self.threshold:
                self.connect(self.morphogen)

    def create_new_cell(self):
        # create new cell at parameters
        new_cell_coordinates = Coordinates.change_coords(self.executing_cell.Coordinate, self.new_coordinate_shift)
        new_cell_rules = self.executing_cell.rules
        new_cell = Cell_v2.Cell(Cell_space = self.cell_space, Coordinate=new_cell_coordinates, name="", rules = new_cell_rules)
        self.cell_space.Cells.append(new_cell)

        # when divided, put the child cell address here and create a rule that connects to the address marker of the child cell

    def connect(self, morphogen):
        for i in self.morphogen.cells:
            self.cell_space.addAxons(Axon(self.executing_cell, i, self.inhibit_excite_type))
        # TODO prevent connections to itself


    # Example function
    def make_two_children_below(cell, amount):
        if amount >= 0.2:
            cell.mitosis_counter = 2
            if cell.name == 0:
                print("2 mitosis for cell 0")

        if amount < 0.2 and amount > 0:
            cell.morphogens["leg"][1] = 0.5

    # Example function
    def elongate_down(cell, amount):
        if amount >= 0.2:
            cell.mitosis_counter = 1