# Runs the whole simulation

class Environment:
    def __init__(self):
        super(Environment, self).__init__()
        self.population = {}


    def running_all(self):

        # TODO call the population
        #  Run the simulation for one generation
        #  Evaluate the fitness of all learners
        #  Eliminate two thirds of the most unfit learners
        #  Repopulate the population with the remaining third
        #  Mutate the morphogens of the new learners
        print("generation run")

    def run_generation(self):
        for entity in self.population:
            # TODO entity.run() returns the learning speed
            entity.run
        print("generation run")

    def evaluation(self, population):
        print("evaluating all learners of the population")

    def eliminate_unfit_learners(self):
        print("eliminating unfit learners")

    def repopulate(self):
        print("repopulating")

    def mutate(self):
        print("mutation")

