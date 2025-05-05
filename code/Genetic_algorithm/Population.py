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
        #  problem: the surviving individuals from the last generation get mutated
        #  only the repopulation should get mutated
        #  dont run the surviving individuals, just copy their fitness score
        #  repopulate with the top 50% ----- working on it
        #  for the clones mutate the morphogen rule set of an individual
        self.environment = environment
        self.survivors = 5
        generations = []
        generation = self.first_generation()
        generations.append(generation)
        indiv_fit = self.selection(generation)

        for timestep in np.arange(0,20):
            print("Generation:", timestep)
            generation = self.generation(indiv_fit)
            generations.append(generation)
            indiv_fit = self.selection(pd.concat([generation, indiv_fit]))

        print("stop")

    def first_generation(self):
        generation = []
        for counter in np.arange(0, self.population_size-5):
            print("Individual:", counter, "/", self.population_size)
            individual = Genetic_algorithm.Individual.Individual(environment=self.environment)
            # directly mutate the morphogens, not the individual
            # print("debug 1 manually written rule. Connect from the first input to the first output neuron")
            individual.input_to_output_debug()
            rule_keys = list(individual.c.Rules.keys())
            for rule in rule_keys:
                individual.c.Rules[rule].mutate()
            individual.morphogenesis_individual()
            individual.running_the_network()
            generation.append([individual, individual.fitness_score])
        return pd.DataFrame(generation, columns=["individual", "fitness"])

    # take the morpho rules from the previous generation for the next one
    def generation(self, individuals_prev_generation):
        generation = []
        for counter in np.arange(0, self.population_size-self.survivors):
            print("Individual:", counter,"/",self.population_size)
            individual = Genetic_algorithm.Individual.Individual(environment=self.environment)

            # take the morpho rules and give them to the new generation
            # TODO
            #  properly copy rules and also the rule counter of the individual cell_space
            individuals_prev_generation.iloc[counter % len(individuals_prev_generation), 0].copy_rules_to(individual)
            # individual.c.Rules = morphogens_prev_generation.iloc[counter%len(morphogens_prev_generation),0].c.Rules

            rule_keys = list(individual.c.Rules.keys())
            for rule in rule_keys:
                individual.c.Rules[rule].mutate()
            individual.morphogenesis_individual()
            individual.running_the_network()

            generation.append([individual, individual.fitness_score])
        return pd.concat([individuals_prev_generation,pd.DataFrame(generation, columns=["individual", "fitness"])])

    def selection(self, generation_population):
        # generation_population = pd.DataFrame(generation_population, columns=["individual", "fitness"])
        # best_fitness = generation_population.iloc[np.argmax(generation_population.iloc[:, 1]), :]
        # sorting by fitness
        generation_population = generation_population.sort_values("fitness", ascending=False)
        selection_pressured = generation_population.iloc[:int(self.population_size / 3), :]
        return selection_pressured

        # # inheritable: Rules
        # morpho_rule_set = []  # pd.Series(name="morpho_rules")
        # for i in selection_pressured.iloc[:, 0]:
        #     # morpho_rule_set[morpho_rule_set.shape[0]] = i.c.Rules
        #     morpho_rule_set.append([i.c.Rules, i.fitness_score])
        #
        # selection_pressured["Morpho_rules"] = morpho_rule_set
        # return morpho_rule_set