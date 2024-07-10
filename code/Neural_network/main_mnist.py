import Neuron_space
import Backprop

import numpy as np
np.random.seed(1)

# Do you want visualization? Do you want the learning to be fast?

n = Neuron_space.NeuronSpace(fast = True, Visualization=False, neuron_number = 100)
n.spawn_neurons_axons(input_number=64, output_number=10)


bp = Backprop.Backpropagation(n)
n.bp = bp

from sklearn import datasets
mnist = datasets.load_digits()
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def plot_metrics(train_acc,train_rec,train_pre,train_f1,epoch_losses,validation_acc,epoch_validation_losses):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.plot(np.arange(len(epoch_losses)), epoch_losses, label='train losses')
    ax1.plot(np.arange(len(epoch_validation_losses)), epoch_validation_losses, label='validation losses')
    ax1.set_title("MNIST dataset losses")
    ax1.set_xlabel("epochs")
    ax2.plot(np.arange(len(train_acc)), train_acc, label='train accuracy')
    ax2.plot(np.arange(len(validation_acc)), validation_acc, label='validation accuracy')
    ax2.set_title("MNIST dataset accuracy")
    ax2.set_xlabel("epochs")
    ax3.plot(np.arange(len(train_rec)), train_rec, label='train recall')
    ax3.plot(np.arange(len(train_f1)), train_f1, label='train F1')
    ax3.plot(np.arange(len(train_pre)), train_pre, label='train precision')
    ax3.set_title("MNIST dataset metrics")
    ax3.set_xlabel("epochs")
    fig.legend()
    fig.show()

X = np.array(mnist.data)
y = np.array(mnist.target)

f = np.zeros([len(y),10])
for idx, i in enumerate(y):
    f[idx, i] = 1

y = f

train_len = 50
val_len = 55
test_len = 85
X, y = shuffle(X, y, random_state=1)
X_train = X[[0,2,4,5]]
#X_train = X[:train_len]
X_validation = X[train_len:val_len]
X_test = X[val_len:test_len]
y_train = y[[0,2,4,5]]
#y_train = np.array(y[:train_len])
y_validation = np.array(y[train_len:val_len])
y_test = np.array(y[val_len:test_len])

std_slc = StandardScaler()
std_slc.fit(X_train)
X_train = std_slc.transform(X_train)
X_validation = std_slc.transform(X_validation)
X_test = std_slc.transform(X_test)


#train_data = np.concatenate((X_train, y_train.T), axis=1)
#test_data = np.concatenate((X_test, y_test.T), axis=1)

epochs = 200
train_acc = []
train_rec = []
train_pre = []
train_f1 = []
losses = np.array([])
epoch_losses = []

validation_acc = []
validation_losses = np.array([])
epoch_validation_losses = []

for idx,i in enumerate(np.arange(0,epochs)):
    validation_loss = bp.get_loss(X_validation, y_validation)
    epoch_validation_losses.append(np.average(validation_loss))
    validation_losses = np.vstack([validation_losses, validation_loss]) if validation_losses.size else validation_loss
    validation_acc.append(bp.evaluation(X_validation, y_validation))

    loss = bp.train(X_train, y_train, learning_rate = 1) #* 0.98**idx)
    epoch_losses.append(np.average(loss))
    losses = np.vstack([losses, loss]) if losses.size else loss
    train_acc.append(bp.evaluation(X_train, y_train, "accuracy"))
    train_rec.append(bp.evaluation(X_train, y_train, "recall"))
    train_pre.append(bp.evaluation(X_train, y_train, "precision"))
    train_f1.append(bp.evaluation(X_train, y_train, "f1"))


    print("epoch: ", (idx+1), "/", epochs)

plot_metrics(train_acc,train_rec,train_pre,train_f1,epoch_losses,validation_acc,epoch_validation_losses)


# for i in np.arange(0, y_test.shape[1]):
#     print(i, " accuraccy: ", bp.evaluation(X_test, y_test))

print(train_acc)

# neurons coloured by their bias
# axons coloured by their weight
n.start_vis()
n.draw_brain()

# for i in np.arange(0, y_test.shape[1]):
#     print(i, " accuraccy: ", bp.evaluation(X_test, y_test))

for i in np.arange(0,len(X_train)):
    print(np.argmax(bp.predict(X_train[i])), " ", np.argmax(y_train[i]))

for i in np.arange(0,len(X_train)):
    print([ '%.2f' % elem for elem in bp.predict(X_train[i])], " ", y_train[i])

print("Simulation done")