import Morphogen_simulation_v2.Coordinates
import Morphogen_simulation_v2.Cell_v2
import Morphogen_simulation_v2.Axon
import random
import Morphogen_simulation_v2.Morphogens_v2
from Morphogen_simulation_v2 import *

import math

class Rule():
    def __init__(self, cell_space):
        self.cell_space = cell_space
        self.name = self.cell_space.Rule_counter
        self.cell_space.Rule_counter += 1
        self.cell_space.Rules[self.name] = self


        # These variables can be mutated
        self.new_coordinate_shift = Coordinates.Coordinate(5,0,0) # default is to the right
        self.threshold = 0.5
        self.morphogen = random.choice(self.cell_space.Morphogens)# pick random morphogen from all morphogens
        self.inhibit_excite_type = 1 # 1 = excite, 0 = inhibit
        self.child_limit = 0
        self.rule_type = 3
        self.target_morphogen = random.choice(self.cell_space.Morphogens)


    def mutate(self, mutation_rate=1):
        """
                Mutates one of the mutate-able variables by a small random amount based on mutation_rate.
                """
        mutation_targets = [
            "new_coordinate_shift",
            "threshold",
            "morphogen",
            "inhibit_excite_type",
            "child_limit",
            "rule_type",
            "delete_rule",
            "create_rule"
        ]

        # Pick a random attribute to mutate
        target = random.choice(mutation_targets)

        if target == "new_coordinate_shift":
            # Mutate coordinate shift with a small random change

            # Generate a random shift vector
            dx = random.uniform(-5, 5)
            dy = random.uniform(-5, 5)
            dz = random.uniform(-5, 5)

            # Compute the magnitude of the vector
            magnitude = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # Ensure the shift is at least distance 1
            if magnitude < 1.0:
                scale = 1.0 / magnitude  # Scale factor to make magnitude 1
                dx *= scale
                dy *= scale
                dz *= scale

            # Apply the shift
            self.new_coordinate_shift = Coordinates.Coordinate(dx, dy, dz)

        elif target == "threshold":
            # Mutate threshold but keep it between 0 and 10
            self.threshold = max(0, min(10, self.threshold + random.uniform(-1, 1)))

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
                self.cell_space.Rules.pop(self.name)

        elif target == "create_rule":
            # Small chance of creating a new rule
            if random.random() < mutation_rate:
                new_rule = Rule(self.cell_space)
                new_rule.new_coordinate_shift = self.new_coordinate_shift
                new_rule.threshold = self.threshold
                new_rule.morphogen = self.morphogen
                new_rule.inhibit_excite_type = self.inhibit_excite_type
                new_rule.child_limit = self.child_limit
                new_rule.rule_type = self.rule_type


    # i tried to make a function that is customizable by the genetic algorithm
    # this function should be modifyable by the genetic algorithm and perform various tasks such as creating an action when a morphogen threshold is reached.
    def rule(self, executing_cell):
            # 2 types of actions
            # 1. send out new morphogen
            # 2. if child cells above threshold, stop creating new ones
            # 3. create Axon to cell with morphogen x
            #
        # TODO LET RULES ACTIVATE EACH OTHER so the morphogen removal rule 2 only happens after a condition is met

        morphogen_concentration = executing_cell.calc_morphogen(self.morphogen)
        if self.rule_type == 1: # create a new morphogen and add it to the cell expression
            # TODO this should be done by the mutation, not with a rule
            new_morpho = Morphogens_v2(1, cell = executing_cell)
            executing_cell.new_morphogens(new_morpho)
        if self.rule_type == 2: # remove a morphogen from a cell
            executing_cell.del_morphogens(self.morphogen)
        if self.rule_type == 3: # create a new cell
            if morphogen_concentration >= self.threshold: # TODO dont to this via child number but via summed morphogen density of nearby cells
                if self.child_limit < len(executing_cell.children):
                    self.create_new_cell(executing_cell)
        if self.rule_type == 4:   # connect to target cell
            # if there is enough concentration of a morphogen
            if morphogen_concentration >= self.threshold:
                self.connect(executing_cell)
        if self.rule_type == 5: # add a morphogen to the cell expression

            executing_cell.new_morphogens(new_morpho)

    def create_new_cell(self, executing_cell):
        # create new cell at parameters
        new_cell_coordinates = Coordinates.change_coords(executing_cell.coordinate, self.new_coordinate_shift)
        new_cell = Cell_v2.Cell(Cell_space = self.cell_space, Coordinate=new_cell_coordinates)
        self.cell_space.Cells[new_cell.name] = new_cell

        # when divided, put the child cell address here and create a rule that connects to the address marker of the child cell

    def connect(self, executing_cell):
        for child in self.target_morphogen.cells: # TODO morphogens associated with each cell are not updated properly yet
            if executing_cell.name != child.name:
                print("Rule type 4 activated, connect cell ", executing_cell.name, " to cell ", child.name, " Rule name:", self.name)
                Axon.Axon(executing_cell, child, self.inhibit_excite_type, self.cell_space)


