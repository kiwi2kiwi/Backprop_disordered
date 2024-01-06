import Neuron_space
import Backprop

import numpy as np

# Do you want visualization? Do you want the learning to me fast?

n = Neuron_space.NeuronSpace(fast = False, Visualization=False, neuron_number = 1)
n.spawn_neurons_axons()


bp = Backprop.Backpropagation(n)


from sklearn import datasets
iris = datasets.load_iris()
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

X = np.array(iris.data)
y = np.array(iris.target)
y = y
X, y = shuffle(X, y)
X_train = X[:100]
X_test = X[100:]
y_train = np.array([y[:100]])
y_test = np.array([y[100:]])

std_slc = StandardScaler()
std_slc.fit(X_train)
X_train = std_slc.transform(X_train)
X_test = std_slc.transform(X_test)


train_data = np.concatenate((X_train, y_train.T), axis=1)
test_data = np.concatenate((X_test, y_test.T), axis=1)


losses = np.array([])
for i in np.arange(0,20):
    losses = np.concatenate((losses, bp.train(X_train, y_train.T)))



import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(np.arange(len(losses)), losses)
fig.show()
bp.evaluation(X_test, y_test.T)

print("Simulation done")