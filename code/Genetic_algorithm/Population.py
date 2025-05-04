print("imported Population")
import sys
sys.path.append('..')
import Genetic_algorithm.Individual
import numpy as np

class Population:
    def __init__(self, environment):
        super(Population, self).__init__()
        # TODO
        #  this is a collection of all the learners in a generation
        #  give access to all the learners
        #  1. initialize an individual - done
        #  2. give individual a morphogen rule set
        #  3. Morphogenesis - done
        #  4. Give the data to the individual - done
        #  5. Individual trains and predicts
        #  6. Individual returns metrics as fitness score
        #  7. Individual returns its morphogen rule set

        # TODO
        #  receive set of individuals from the last population
        #  remove bottom 50% of individuals
        #  repopulate with the top 50%
        #  for the clones mutate the morphogen rule set of an individual
        self.environment = environment

    def generation(self):
        self.generation = []
        for counter in np.arange(0,50):
            individual = Genetic_algorithm.Individual.Individual(environment=self.environment)
            self.generation.append([individual, individual.fitness_score])


