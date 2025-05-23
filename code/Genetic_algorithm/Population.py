import sys
from fileinput import filename

sys.path.append('..')
import json
import Genetic_algorithm.Individual
import Neural_network.Neuron_space
import Neural_network.nn_execution as nn_exe
import Morphogen_simulation_v2.Cell_space
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.setrecursionlimit(300)

class Population:
    def __init__(self, environment):
        super(Population, self).__init__()
        self.population_size = 15
        self.environment = environment
        self.survivors = 5
        self.generations = []

    def run_simulation_from_start(self):
        generation = self.first_generation()
        self.generations.append(generation)
        indiv_fit = self.selection(generation)

        for timestep in np.arange(0,5):
            print("Generation:", timestep)
            generation = self.generation(indiv_fit, timestep)
            self.generations.append(generation)
            indiv_fit = self.selection(pd.concat([generation, indiv_fit]))

        self.save(filename="../../Morphogen_rule_saves/current.genes", individuals=indiv_fit["individual"], timestep=timestep)

        print("stop")

    def continue_simulation_from_file(self, filepath_to_rules):
        # load all individuals from the file
        individuals, generation_timestep = self.load(filename=filepath_to_rules)

        generation = self.generation(individuals, generation_timestep, loaded = True)
        self.generations.append(generation)
        indiv_fit = self.selection(generation)

        for timestep in np.arange(0,5):
            print("Generation:", timestep + generation_timestep)
            generation = self.generation(indiv_fit, timestep)
            self.generations.append(generation)
            indiv_fit = self.selection(pd.concat([generation, indiv_fit]))

        pass


    def small_generation_debug(self):
        individual = Genetic_algorithm.Individual.Individual(environment=self.environment)


    def first_generation(self):
        generation = []
        for counter in np.arange(0, self.population_size):
            individual = Genetic_algorithm.Individual.Individual(environment=self.environment)

            # print("debug 1 manually written rule. Connect from the first input to the first output neuron")
            # individual.input_to_output_debug()

            individual.c.Morphogen_addresses_of_previous_generation = list(individual.c.Morphogens.keys())
            individual.create_random_rules(400)

            rule_keys = list(individual.c.Rules.keys())
            # directly mutate the morphogens, not the individual
            for rule in rule_keys:
                individual.c.Rules[rule].mutate()
            individual.morphogenesis_individual()
            individual.running_the_network()
            generation.append([individual, individual.fitness_scores])
            print("Individual:", counter, "/", self.population_size, "score", np.mean(individual.fitness_scores))
        return pd.DataFrame(generation, columns=["individual", "fitness"])

    # take the morpho rules from the previous generation for the next one
    def generation(self, individuals_prev_generation, timestep, loaded = False):
        generation = []
        for counter in np.arange(0, self.population_size-self.survivors):

            individual = Genetic_algorithm.Individual.Individual(environment=self.environment)

            # take the morpho rules and give them to the new generation
            if loaded:
                individuals_prev_generation[counter % len(individuals_prev_generation)].copy_rules_and_morpho_addresses_to(individual)
            else:
                individuals_prev_generation.iloc[counter % len(individuals_prev_generation), 0].copy_rules_and_morpho_addresses_to(individual)

            rule_keys = list(individual.c.Rules.keys())
            for rule in rule_keys:
                if rule in individual.c.Rules.keys():
                    individual.c.Rules[rule].mutate()
            # print("available morphogen addresses of previous:", individuals_prev_generation.iloc[counter % len(individuals_prev_generation), 0].c.Morphogens.keys())
            # print("available cells of previous:",
            #       individuals_prev_generation.iloc[counter % len(individuals_prev_generation), 0].c.Cells.keys())
            # print("available morphogen addresses of current:", individual.c.Morphogens.keys())
            # print("available cells of current:", individual.c.Cells.keys())
            individual.morphogenesis_individual()
            individual.running_the_network()

            generation.append([individual, individual.fitness_scores])
            print("Individual:", counter,"/",self.population_size, "\tGeneration", timestep, "neurons:", len(individual.c.Cells.keys()), "score", np.mean(individual.fitness_scores))
        if loaded:
            return pd.DataFrame(generation, columns=["individual", "fitness"])
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



    # ../../Morphogen_rule_saves
    def save(self, filename, individuals, timestep = 0):
        """
        individuals: list of objects, each with a .cell_space attribute
        """
        all_data = []

        for individual in individuals:
            cs = individual.c
            data = {
                'Morphogen_addresses_of_previous_generation': cs.Morphogen_addresses_of_previous_generation,
                'Rules': [rule.to_dict() for rule in cs.Rules.values()],
                # 'Rule_counter': cell_space.Rule_counter
            }
            all_data.append(data)

        with open(filename, 'w') as f:
            json.dump({'individuals': all_data, "generation" : timestep}, f, indent=2)

    def load(self, filename):
        with open(filename, 'r') as f:
            file_data = json.load(f)

        individuals = []
        for data in file_data['individuals']:
            individual = Genetic_algorithm.Individual.Individual(environment=self.environment)
            cs = individual.c
            cs.Morphogen_addresses_of_previous_generation = data['Morphogen_addresses_of_previous_generation']
            # cs.Rule_counter = data['Rule_counter']
            # cs.Rules = {}

            for rule_data in data['Rules']:
                Morphogen_simulation_v2.Rules.Rule(cs, from_data=rule_data)

            individuals.append(individual)

        return individuals, file_data['generation']

    def plot_metrics_over_generations(self):
        self.generations[-1].iloc[1, 0].c.start_vis()

        data = self.generations[1:]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        generations = list(range(len(data)))

        # Initialize structures for all metrics
        all_metrics = {name: [] for name in metric_names}

        for df in data:
            # Unpack fitness column: shape (num_individuals, 4)
            metrics_array = np.stack(df['fitness'].values)  # shape: (10, 4)

            for i, name in enumerate(metric_names):
                all_metrics[name].append(metrics_array[:, i])  # list of arrays per generation

        # Plot each metric
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, name in enumerate(metric_names):
            metric_data = all_metrics[name]
            means = [np.mean(gen) for gen in metric_data]
            stds = [np.std(gen) for gen in metric_data]
            mins = [np.min(gen) for gen in metric_data]
            maxs = [np.max(gen) for gen in metric_data]

            ax = axes[i]
            ax.plot(generations, means, label=f'Mean {name}', color='tab:blue')
            ax.fill_between(generations,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color='tab:blue', alpha=0.2, label='±1 Std Dev')
            ax.fill_between(generations, mins, maxs, color='gray', alpha=0.1, label='Min–Max Range')

            ax.set_title(name)
            ax.set_xlabel('Generation')
            ax.set_ylabel(name)
            ax.grid(True)
            ax.legend()

        plt.suptitle('Performance Metrics Across Generations', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # retrain an individual and plots its training metrics
    def plot_training_of_individual(self):

        n = Neural_network.Neuron_space.NeuronSpace(Visualization=False)
        n.import_network(self.generations[-1].iloc[1,0].c)
        nn_exe.running_the_network(individual=self.generations[-1].iloc[1,0], n=n, viz=True, epochs = 50)

        self.generations[-1].iloc[-1,0].c.start_vis()
