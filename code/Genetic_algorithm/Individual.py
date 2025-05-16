import sys
import os
import numpy as np
import Morphogen_simulation_v2.Cell_space
import Neural_network.Neuron_space
import Neural_network.nn_execution as nn_exe

class Individual:
    def __init__(self, environment):
        super(Individual, self).__init__()
        self.viz = False # visualization
        self.environment = environment
        self.c = Morphogen_simulation_v2.Cell_space.Cell_space()


    def morphogenesis_individual(self):
        # the rules should be executed a few times to allow for recursive structure building
        self.morphogenesis()
        self.morphogenesis()
        if self.viz:
            self.c.start_vis()
            self.c.draw_image()

    def running_the_network(self):
        self.n = Neural_network.Neuron_space.NeuronSpace(Visualization=self.viz)
        self.n.import_network(self.c)
        # rules_list = list(self.c.Rules.values())
        # print("type:", rules_list[0].rule_type, "logic:", rules_list[0].logic_morphogen, " target:", rules_list[0].target_morphogen)
        if self.viz:
            self.n.start_vis()
            self.n.draw_brain()
        self.fitness_scores = nn_exe.running_the_network(individual=self, n=self.n, viz = False)

    # connect input cells to x output cells
    def input_to_output_debug(self):
        demo_rule_0 = Morphogen_simulation_v2.Rules.Rule(self.c)
        demo_rule_0.threshold = 0
        demo_rule_0.logic_morphogen = self.c.input_cells[0].address.name
        demo_rule_0.target_morphogen = self.c.output_cells[0].address.name
        demo_rule_0.rule_type = 4

        demo_rule_1 = Morphogen_simulation_v2.Rules.Rule(self.c)
        demo_rule_1.threshold = 0
        demo_rule_1.logic_morphogen = self.c.input_cells[0].address.name
        demo_rule_1.target_morphogen = self.c.output_cells[1].address.name
        demo_rule_1.rule_type = 4

        demo_rule_2 = Morphogen_simulation_v2.Rules.Rule(self.c)
        demo_rule_2.threshold = 0
        demo_rule_2.logic_morphogen = self.c.input_cells[0].address.name
        demo_rule_2.target_morphogen = self.c.output_cells[2].address.name
        demo_rule_2.rule_type = 4

        # demo_rule_3 = Morphogen_simulation_v2.Rules.Rule(self.c)
        # demo_rule_3.threshold = 0
        # demo_rule_3.logic_morphogen = self.c.input_cells[0].address.name
        # demo_rule_3.target_morphogen = self.c.output_cells[3].address.name
        # demo_rule_3.rule_type = 4
        #
        # demo_rule_4 = Morphogen_simulation_v2.Rules.Rule(self.c)
        # demo_rule_4.threshold = 0
        # demo_rule_4.logic_morphogen = self.c.input_cells[0].address.name
        # demo_rule_4.target_morphogen = self.c.output_cells[4].address.name
        # demo_rule_4.rule_type = 4
        #
        # demo_rule_5 = Morphogen_simulation_v2.Rules.Rule(self.c)
        # demo_rule_5.threshold = 0
        # demo_rule_5.logic_morphogen = self.c.input_cells[0].address.name
        # demo_rule_5.target_morphogen = self.c.output_cells[5].address.name
        # demo_rule_5.rule_type = 4
        #
        # demo_rule_6 = Morphogen_simulation_v2.Rules.Rule(self.c)
        # demo_rule_6.threshold = 0
        # demo_rule_6.logic_morphogen = self.c.input_cells[0].address.name
        # demo_rule_6.target_morphogen = self.c.output_cells[6].address.name
        # demo_rule_6.rule_type = 4
        #
        # demo_rule_7 = Morphogen_simulation_v2.Rules.Rule(self.c)
        # demo_rule_7.threshold = 0
        # demo_rule_7.logic_morphogen = self.c.input_cells[0].address.name
        # demo_rule_7.target_morphogen = self.c.output_cells[7].address.name
        # demo_rule_7.rule_type = 4
        #
        # demo_rule_8 = Morphogen_simulation_v2.Rules.Rule(self.c)
        # demo_rule_8.threshold = 0
        # demo_rule_8.logic_morphogen = self.c.input_cells[0].address.name
        # demo_rule_8.target_morphogen = self.c.output_cells[8].address.name
        # demo_rule_8.rule_type = 4
        #
        # demo_rule_9 = Morphogen_simulation_v2.Rules.Rule(self.c)
        # demo_rule_9.threshold = 0
        # demo_rule_9.logic_morphogen = self.c.input_cells[0].address.name
        # demo_rule_9.target_morphogen = self.c.output_cells[9].address.name
        # demo_rule_9.rule_type = 4



    def create_random_rules(self, x):
        for i in np.arange(0,x):
            demo_rule_3 = Morphogen_simulation_v2.Rules.Rule(self.c)
            # demo_rule_3.threshold = 0

    def morphogenesis(self):
        self.c.neurogenesis()

    def get_data(self):
        return [self.environment.X_train, self.environment.X_val, self.environment.y_train, self.environment.y_val]

    def copy_rules_to(self, individual):
        # TODO also copy morphogens
        new_cell_space = individual.c
        new_rules = {}
        for r in self.c.Rules.values():
            new_rule = Morphogen_simulation_v2.Rules.Rule(new_cell_space)
            new_rule.name = r.name
            new_rule.new_coordinate_shift = r.new_coordinate_shift
            new_rule.threshold = r.threshold
            new_rule.logic_morphogen = r.logic_morphogen
            new_rule.inhibit_excite_type = r.inhibit_excite_type
            new_rule.child_limit = r.child_limit
            new_rule.rule_type = r.rule_type
            new_rule.target_morphogen = r.target_morphogen
            new_rule.mutation_counter = r.mutation_counter
            new_rules[new_rule.name] = new_rule
        new_cell_space.Rule_counter = self.c.Rule_counter
