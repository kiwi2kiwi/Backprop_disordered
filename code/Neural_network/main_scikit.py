import Neuron_space
import Backprop

import numpy as np


from sklearn import datasets
iris = datasets.load_iris()
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


X = np.array(iris.data)
y = np.array(iris.target)
#y = y/2
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

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(6,),
                    random_state=1)
clf.fit(X_train, y_train.T)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

ax.plot(np.arange(len(epoch_losses)), epoch_losses, label='train losses')
ax.plot(np.arange(len(epoch_validation_losses)), epoch_validation_losses, label='test losses')
ax1.set_title("Iris dataset losses")
ax.set_xlabel("epochs")
fig.legend()
fig.show()



print("Simulation done")