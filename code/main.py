import Neuron_space
import Backprop

import numpy as np

n = Neuron_space.NeuronSpace()
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

def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
