print("imported Population")
import sys
sys.path.append('..')
import Genetic_algorithm.Individual
import numpy as np
import pandas as pd

class Population:
    def __init__(self, environment):
        super(Population, self).__init__()
        self.population_size = 10
        # TODO
        #  this is a collection of all the learners in a generation
        #  give access to all the learners
        #  1. initialize an individual - done
        #  2. give individual a morphogen rule set
        #  3. Morphogenesis - done
        #  4. Give the data to the individual - done
        #  5. Individual trains and predicts - done
        #  6. Individual returns metrics as fitness score - done
        #  7. Individual returns its morphogen rule set

        # TODO
        #  for the first generation use the default debug morphogen set and mutate all of them - done
        #  receive set of individuals from the last population - done
        #  remove bottom 50% of individuals - done
        #  repopulate with the top 50% -----
        #  for the clones mutate the morphogen rule set of an individual
        self.environment = environment
        generation_population = self.first_generation()
        generation_population = pd.DataFrame(generation_population, columns=["individual", "fitness"])
        best_fitness = generation_population.iloc[np.argmax(generation_population.iloc[:, 1]),:]
        # sorting by fitness
        generation_population = generation_population.sort_values("fitness", ascending=False)
        selection_pressured = generation_population.iloc[:int(self.population_size/3),:]

        # repopulation
        # inheritable: Rules
        morpho_rule_set = []#pd.Series(name="morpho_rules")
        for i in selection_pressured.iloc[:, 0]:
            # morpho_rule_set[morpho_rule_set.shape[0]] = i.c.Rules
            morpho_rule_set.append(i.c.Rules)

        selection_pressured["Morpho_rules"]=morpho_rule_set

        # take the morpho rules and give them to the new generation


        print("stop")

    def first_generation(self):
        generation = []
        for counter in np.arange(0, self.population_size):

            individual = Genetic_algorithm.Individual.Individual(environment=self.environment)
            # directly mutate the morphogens, not the individual
            print("debug 1 manually written rule. Connect from the first input to the first output neuron")
            individual.input_to_output_debug()
            rule_keys = list(individual.c.Rules.keys())
            for rule in rule_keys:
                individual.c.Rules[rule].mutate()
            individual.morphogenesis_individual()
            individual.running_the_network()
            generation.append([individual, individual.fitness_score])
        return generation

    # take the morpho rules from the previous generation for the next one
    def generation(self, morphogens_prev_generation):
        generation = []
        for counter in np.arange(0, self.population_size):
            individual = Genetic_algorithm.Individual.Individual(environment=self.environment)
            individual.
            generation.append([individual, individual.fitness_score])
        return generation


