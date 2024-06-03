import Neuron_space
import Backprop

import numpy as np
np.random.seed(1)

# Do you want visualization? Do you want the learning to be fast?

n = Neuron_space.NeuronSpace(fast = True, Visualization=False, neuron_number = 20)
n.spawn_neurons_axons(input_number=64, output_number=10)


bp = Backprop.Backpropagation(n)


from sklearn import datasets
mnist = datasets.load_digits()
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


X = np.array(mnist.data)
y = np.array(mnist.target)

f = np.zeros([len(y),10])
for idx, i in enumerate(y):
    f[idx, i] = 1

y = f

X, y = shuffle(X, y, random_state=1)
X_train = X[:100]
X_validation = X[100:150]
X_test = X[150:200]
y_train = np.array(y[:100])
y_validation = np.array(y[100:150])
y_test = np.array(y[150:200])

std_slc = StandardScaler()
std_slc.fit(X_train)
X_train = std_slc.transform(X_train)
X_validation = std_slc.transform(X_validation)
X_test = std_slc.transform(X_test)


#train_data = np.concatenate((X_train, y_train.T), axis=1)
#test_data = np.concatenate((X_test, y_test.T), axis=1)

epochs = 10
train_acc = []
losses = np.array([])
epoch_losses = []

test_acc = []
validation_losses = np.array([])
epoch_validation_losses = []

for idx,i in enumerate(np.arange(0,epochs)):
    validation_loss = bp.train(X_validation, y_validation, learning_rate=0)
    epoch_validation_losses.append(np.average(validation_loss))
    validation_losses = np.vstack([validation_losses, validation_loss]) if validation_losses.size else validation_loss
    test_acc.append(bp.evaluation(X_validation, y_validation))

    loss = bp.train(X_train, y_train, learning_rate = 1)
    epoch_losses.append(np.average(loss))
    losses = np.vstack([losses, loss]) if losses.size else loss
    train_acc.append(bp.evaluation(X_train, y_train))

    print("epoch: ", (idx+1), "/", epochs)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()

ax.plot(np.arange(len(epoch_losses)), epoch_losses, label='train losses')
ax.plot(np.arange(len(epoch_validation_losses)), epoch_validation_losses, label='test losses')
ax.set_title("MNIST dataset losses")
ax.set_xlabel("epochs")
fig.legend()
fig.show()

ax1.plot(np.arange(len(train_acc)), train_acc, label='train accuracy')
ax1.plot(np.arange(len(test_acc)), test_acc, label='test accuracy')
ax1.set_title("MNIST dataset accuracy")
ax1.set_xlabel("epochs")
fig1.legend()
fig1.show()

for i in np.arange(0, y_test.shape[1]):
    print(i, " accuraccy: ", bp.evaluation(X_test, y_test))



# neurons coloured by their bias
# axons coloured by their weight
n.start_vis()
n.draw_brain()

for i in np.arange(0, y_test.shape[1]):
    print(i, " accuraccy: ", bp.evaluation(X_test, y_test))


print("Simulation done")