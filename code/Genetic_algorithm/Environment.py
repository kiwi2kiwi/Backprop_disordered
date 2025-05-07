import os
import sys
sys.path.append('..')
import Genetic_algorithm.Population

import numpy as np
import math
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# Runs the whole simulation
class Environment:
    def __init__(self):
        super(Environment, self).__init__()
        self.population = {}
        self.running_all()


    def running_all(self):

        # TODO call the population
        #  Run the simulation for one generation
        #  Evaluate the fitness of all learners
        #  Eliminate two thirds of the most unfit learners
        #  Repopulate the population with the remaining third
        #  Mutate the morphogens of the new learners
        self.data_loading()
        self.population = Genetic_algorithm.Population.Population(environment = self)

        print("generation run")

    def run_generation(self):
        self.population.generation()
        self.populatio
        print("generation run")

    def evaluation(self, population):
        print("evaluating all learners of the population")

    def eliminate_unfit_learners(self):
        print("eliminating unfit learners")

    def repopulate(self):
        print("repopulating")

    def mutate(self):
        print("mutation")

    def data_loading(self):
        iris = datasets.load_iris()
        X = np.array(iris.data)
        y = np.array(iris.target)
        y_oh = np.eye(3)[y]
        X, y = shuffle(X, y_oh, random_state=42)
        self.X_train = X[:100]
        self.X_val = X[100:]
        self.y_train = np.array(y[:100])
        self.y_val = np.array(y[100:])
        # X_train = X[:1]
        # X_val = X[1:]
        # y_train = np.array(y[:1])
        # y_val = np.array(y[1:])

        std_slc = StandardScaler()
        std_slc.fit(self.X_train)
        self.X_train = std_slc.transform(self.X_train)
        self.X_val = std_slc.transform(self.X_val)

import trace

tracer = trace.Trace(
    ignoredirs=[sys.prefix, sys.exec_prefix],
    trace=100,
    count=0
)
tracer.run('Environment()')

Environment()