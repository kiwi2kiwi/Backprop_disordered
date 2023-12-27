import Neuron_space
import Backprop

import numpy as np

# Do you want visualization? Do you want the learning to me fast?

n = Neuron_space.NeuronSpace(fast = True, neuron_number = 2)
n.spawn_neurons_axons()


bp = Backprop.Backpropagation(n)


from sklearn import datasets
iris = datasets.load_iris()
from sklearn.utils import shuffle

X = np.array(iris.data)
y = np.array(iris.target)
X, y = shuffle(X, y)
X_train = X[:100]
X_test = X[100:]
y_train = np.array([y[:100]])
y_test = np.array([y[100:]])

train_data = np.concatenate((X_train, y_train.T), axis=1)
test_data = np.concatenate((X_test, y_test.T), axis=1)



bp.train(train_data)
bp.evaluation(test_data)

print("Simulation done")