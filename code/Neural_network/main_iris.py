import Neuron_space
import Backprop

import numpy as np
np.random.seed(1)

# Do you want visualization? Do you want the learning to be fast?

n = Neuron_space.NeuronSpace(fast = True, Visualization=False, neuron_number = 4)
n.spawn_neurons_axons(input_number=4, output_number=1)


bp = Backprop.Backpropagation(n)


from sklearn import datasets
iris = datasets.load_iris()
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


X = np.array(iris.data)
#X[:,0] = 0
#X[:,1] = 0
#X[:,2] = 0
#X[:,3] = 0
y = np.array(iris.target)
y = y/2
X, y = shuffle(X, y, random_state=10)
X_train = X[:10]
X_val = X[140:]
y_train = np.array([y[:10]])
y_val = np.array([y[140:]])

# X_train = X
# y_train = np.array([y])

# X_train = X[:100]
# X_val = X[100:]
# y_train = np.array([y[:100]])
# y_val = np.array([y[100:]])



std_slc = StandardScaler()
std_slc.fit(X_train)
X_train = std_slc.transform(X_train)
#X_test = std_slc.transform(X_test)


#train_data = np.concatenate((X_train, y_train.T), axis=1)
#test_data = np.concatenate((X_val, y_val.T), axis=1)

epochs = 1000
train_acc = []
losses = np.array([])
epoch_losses = []

test_acc = [0.1]
validation_losses = np.array([0.5])
epoch_validation_losses = [0.5]

for idx,i in enumerate(np.arange(0,epochs)):
    #n.print_states()
    validation_loss = bp.get_loss(X_val, y_val.T)
    epoch_validation_losses.append(np.average(validation_loss))
    validation_losses = np.vstack([validation_losses, validation_loss]) if validation_losses.size else validation_loss
    test_acc.append(bp.iris_evaluation(X_val, y_val.T))

    #n.print_states()
    loss = bp.train(X_train, y_train.T, learning_rate = 0.5)
    epoch_losses.append(np.average(loss))
    losses = np.vstack([losses, loss]) if losses.size else loss
    train_acc.append(bp.iris_evaluation(X_train, y_train.T))
    #n.print_states()

    print("epoch: ", (idx+1), "/", epochs)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()

ax.plot(np.arange(len(epoch_losses)), epoch_losses, label='train losses')
ax.plot(np.arange(len(epoch_validation_losses)), epoch_validation_losses, label='validation losses')
ax.set_title("Iris dataset losses")
ax.set_xlabel("epochs")
fig.legend()
fig.show()

ax1.plot(np.arange(len(train_acc)), train_acc, label='train accuracy')
ax1.plot(np.arange(len(test_acc)), test_acc, label='validation accuracy')
ax1.set_title("Iris dataset accuracy")
ax1.set_xlabel("epochs")
fig1.legend()
fig1.show()

#print("accuraccy: ", bp.evaluation(X_val, y_val.T))

# neurons coloured by their bias
# axons coloured by their weight
n.start_vis()
n.draw_brain()

n.print_states()
pred = []
for ds in X_train:
    pred.append(bp.predict(ds))
print("pred: ", pred)
print("targ: ", y_train)

print("Simulation done")