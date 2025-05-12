import sys
sys.path.append('..')
import Genetic_algorithm.Individual
import Neural_network.Neuron_space
import Neural_network.nn_execution as nn_exe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.setrecursionlimit(300)

class Population:
    def __init__(self, environment):
        super(Population, self).__init__()
        self.population_size = 20
        # TODO
        #  this is a collection of all the learners in a generation
        #  give access to all the learners
        #  1. initialize an individual - done
        #  2. give individual a morphogen rule set - done
        #  3. Morphogenesis - done
        #  4. Give the data to the individual - done
        #  5. Individual trains and predicts - done
        #  6. Individual returns metrics as fitness score - done
        #  7. Individual returns its morphogen rule set - done

        # TODO
        #  for the first generation use the default debug morphogen set and mutate all of them - done
        #  receive set of individuals from the last population - done
        #  remove bottom 50% of individuals - done
        #  problem: the surviving individuals from the last generation get mutated - fixed
        #  only the repopulation should get mutated - done
        #  dont run the surviving individuals, just copy their fitness score - done
        #  repopulate with the top x% - done
        #  for the clones mutate the morphogen rule set of an individual - done
        self.environment = environment
        self.survivors = 5
        self.generations = []
        generation = self.first_generation()
        self.generations.append(generation)
        indiv_fit = self.selection(generation)

        for timestep in np.arange(0,1):
            print("Generation:", timestep)
            generation = self.generation(indiv_fit, timestep)
            self.generations.append(generation)
            indiv_fit = self.selection(pd.concat([generation, indiv_fit]))



        print("stop")

    def first_generation(self):
        generation = []
        for counter in np.arange(0, self.population_size):
            print("Individual:", counter, "/", self.population_size)
            individual = Genetic_algorithm.Individual.Individual(environment=self.environment)

            # print("debug 1 manually written rule. Connect from the first input to the first output neuron")
            individual.input_to_output_debug()

            rule_keys = list(individual.c.Rules.keys())
            # directly mutate the morphogens, not the individual
            for rule in rule_keys:
                individual.c.Rules[rule].mutate()
            individual.morphogenesis_individual()
            individual.running_the_network()
            generation.append([individual, individual.fitness_scores])
        return pd.DataFrame(generation, columns=["individual", "fitness"])

    # take the morpho rules from the previous generation for the next one
    def generation(self, individuals_prev_generation, timestep):
        generation = []
        for counter in np.arange(0, self.population_size-self.survivors):

            individual = Genetic_algorithm.Individual.Individual(environment=self.environment)

            # take the morpho rules and give them to the new generation
            individuals_prev_generation.iloc[counter % len(individuals_prev_generation), 0].copy_rules_to(individual)
            # individual.c.Rules = morphogens_prev_generation.iloc[counter%len(morphogens_prev_generation),0].c.Rules

            rule_keys = list(individual.c.Rules.keys())
            for rule in rule_keys:
                if rule in individual.c.Rules.keys():
                    individual.c.Rules[rule].mutate()
            individual.morphogenesis_individual()
            print("Individual:", counter,"/",self.population_size, "\tGeneration", timestep, "neurons:", len(individual.c.Cells.keys()))
            individual.running_the_network()

            generation.append([individual, individual.fitness_scores])
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
        nn_exe.running_the_network(individual=self.generations[-1].iloc[1,0], n=n, viz=True)
        
        self.generations[-1].iloc[-1,0].c.start_vis()
