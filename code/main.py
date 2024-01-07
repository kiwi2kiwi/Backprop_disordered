import Neuron_space
import Backprop

import numpy as np

# Do you want visualization? Do you want the learning to be fast?

n = Neuron_space.NeuronSpace(fast = True, Visualization=False, neuron_number = 50)
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

epochs = 100
losses = np.array([])
epoch_losses = []
validation_losses = np.array([])
epoch_validation_losses = []

for i in np.arange(0,epochs):
    loss = bp.train(X_train, y_train.T, learning_rate = 1)
    epoch_losses.append(np.average(loss))
    losses = np.concatenate((losses, loss))

    validation_loss = bp.train(X_test, y_test.T, learning_rate=0)
    epoch_validation_losses.append(np.average(validation_loss))
    validation_losses = np.concatenate((validation_losses, validation_loss))

import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

#fig3, ax3 = plt.subplots()
#ax3.plot(np.arange(len(losses)), losses, label='2 train losses')

ax1.plot(np.arange(len(epoch_losses)), epoch_losses, label='train losses')
ax2.plot(np.arange(len(epoch_validation_losses)), epoch_validation_losses, label='test losses')
ax1.set_title("training error")
ax1.set_xlabel("epochs")
ax2.set_title("testing error")
ax2.set_xlabel("epochs")
fig1.show()
fig2.show()
bp.evaluation(X_test, y_test.T*2)

# neurons coloured by their bias
# axons coloured by their weight
n.start_vis()
n.draw_brain()

print("Simulation done")