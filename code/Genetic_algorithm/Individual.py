import sys
import os
# /Morphogen_simulation_v2
# sys.path.append(os.path.abspath('..'))
import Morphogen_simulation_v2.Cell_space
# sys.path.append(os.path.abspath('../Neural_network'))
import Neural_network.Neuron_space
import Neural_network.nn_execution as nn_exe

# from Morphogen_simulation_v2 import Cell_space

# from Morphogen_simulation_v2 import
class Individual:
    def __init__(self, environment):
        super(Individual, self).__init__()
        self.viz = False # visualization
        self.environment = environment
        # print("creation of input and output cells")
        self.c = Morphogen_simulation_v2.Cell_space.Cell_space()


    def morphogenesis_individual(self):
        # print("morphogenesis")
        self.morphogenesis()
        if self.viz:
            self.c.start_vis()
            self.c.draw_image()

    def running_the_network(self):
        # print("Creating neural network backbone and importing structure")
        self.n = Neural_network.Neuron_space.NeuronSpace(Visualization=self.viz)
        self.n.import_network(self.c)
        if self.viz:
            self.n.start_vis()
            self.n.draw_brain()
        # print("Training the network")
        self.fitness_scores = nn_exe.running_the_network(individual=self, n=self.n)
        # print("trained the network")

    def input_to_output_debug(self):
        demo_rule_1 = Morphogen_simulation_v2.Rules.Rule(self.c)
        demo_rule_1.threshold = 0
        demo_rule_1.logic_morphogen = self.c.input_cells[0].address.name
        demo_rule_1.target_morphogen = self.c.output_cells[0].address.name
        demo_rule_1.rule_type = 4

        demo_rule_2 = Morphogen_simulation_v2.Rules.Rule(self.c)
        demo_rule_2.threshold = 0
        demo_rule_2.logic_morphogen = self.c.input_cells[0].address.name
        demo_rule_2.target_morphogen = self.c.output_cells[1].address.name
        demo_rule_2.rule_type = 4

        demo_rule_3 = Morphogen_simulation_v2.Rules.Rule(self.c)
        demo_rule_3.threshold = 0
        demo_rule_3.logic_morphogen = self.c.input_cells[0].address.name
        demo_rule_3.target_morphogen = self.c.output_cells[2].address.name
        demo_rule_3.rule_type = 4

    def morphogenesis(self):
        self.c.neurogenesis()
        # print("morphogenesis")

    def get_data(self):
        # print("import data")
        return [self.environment.X_train, self.environment.X_val, self.environment.y_train, self.environment.y_val,]

    # def import_morphogens(self):
    #     print("import morphogens")
    #
    # def import_morphogens(self):
    #     print("import morphogens")

    def copy_rules_to(self, individual):
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


# Individual()



