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
        self.dataset_type = "iris"
        self.data_loading()



    def running_all(self):

        self.population = Genetic_algorithm.Population.Population(environment = self)
        self.population.run_simulation_from_start()
        self.population.continue_simulation_from_file(filepath_to_rules="../../Morphogen_rule_saves/current.genes")

        print("generation run")



    def data_loading(self):
        if self.dataset_type == "mnist":
            mnist = datasets.load_digits()
            X = np.array(mnist.data)
            y = np.array(mnist.target)

            X, y = shuffle(X, y, random_state=1)

            # only select the datapoints with the labels 1,2,3
            X = X[np.isin(y, [0, 1, 2, 3, 4, 5])]
            y = y[np.isin(y, [0, 1, 2, 3, 4, 5])]

            # normalize the 0-255 scale to 0-1
            X = X / 255
            y = np.eye(10)[y]

            self.X_train = X[:100]
            self.X_val = X[100:200]
            self.y_train = np.array(y[:100])
            self.y_val = np.array(y[100:200])

            # X_train = X[:1]
            # X_val = X[1:]
            # y_train = np.array(y[:1])
            # y_val = np.array(y[1:])

            # std_slc = StandardScaler()
            # std_slc.fit(self.X_train)
            # self.X_train = std_slc.transform(self.X_train)
            # self.X_val = std_slc.transform(self.X_val)
        if self.dataset_type == "iris":
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

# import trace
#
# tracer = trace.Trace(
#     ignoredirs=[sys.prefix, sys.exec_prefix],
#     trace=100,
#     count=0
# )
# tracer.run('Environment()')

import cProfile, pstats

profiler = cProfile.Profile()
profiler.enable()
env = Environment()
env.running_all()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('ncalls')
stats.print_stats()


print("stop debugger")