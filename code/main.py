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
y = y/2
X, y = shuffle(X, y)
X_train = X[:100]
X_test = X[100:]
y_train = np.array([y[:100]])
y_test = np.array([y[100:]])

std_slc = StandardScaler()
std_slc.fit(X_train)
#X_train = std_slc.transform(X_train)
#X_test = std_slc.transform(X_test)


train_data = np.concatenate((X_train, y_train.T), axis=1)
test_data = np.concatenate((X_test, y_test.T), axis=1)


losses = np.array([])
validation_losses = np.array([])
for i in np.arange(0,2):
    losses = np.concatenate((losses, bp.train(X_train, y_train.T, learning_rate = 1)))
    validation_losses = np.concatenate((validation_losses, bp.train(X_test, y_test.T, learning_rate=0)))

import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
# smoothing loss curve
#losses = [np.average(i) for i in np.array_split(losses, round(len(losses)/3,0))]
ax1.plot(np.arange(len(losses)), losses, label='train losses')
ax2.plot(np.arange(len(validation_losses)), validation_losses, label='test losses')
ax1.set_title("error")
ax2.set_title("error")
fig1.show()
#fig2.show()
bp.evaluation(X_test, y_test.T*2)

# neurons coloured by their bias
# axons coloured by their weight
n.start_vis()
n.draw_brain()

print("Simulation done")