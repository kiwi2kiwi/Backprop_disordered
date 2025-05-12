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
y_ori = np.array(mnist.target)

# turn y into one hot encoding
y_oh = np.eye(10)[y_ori]
# select only 0, 1 and 2
# mask = np.logical_or(np.logical_or(y_oh[:, 3] == 1, y_oh[:, 1] == 1), y_oh[:, 2] == 1)
# mask = np.logical_or(y_oh[:, 2] == 1, y_oh[:, 1] == 1)
X = X#[mask, :]
y = y_oh[:,:]#[mask]
# y = [y[i:i+1] for i in range(0,len(y),1)]


train_len = 1000
val_len = 1200
test_len = 2200
# train_len = 30
# val_len = 50
# test_len = 80
# X, y = shuffle(X, y, random_state=1)
# X_train = X[[0,2,4,5]]
X_train = X[:train_len]
X_validation = X[train_len:val_len]
X_test = X[val_len:test_len]
# y_train = y[[0,2,4,5]]
y_train = np.array(y[:train_len])
y_validation = np.array(y[train_len:val_len])
y_test = np.array(y[val_len:test_len])

std_slc = StandardScaler()
std_slc.fit(X_train)
X_train = std_slc.transform(X_train)
X_validation = std_slc.transform(X_validation)
X_test = std_slc.transform(X_test)


#train_data = np.concatenate((X_train, y_train.T), axis=1)
#test_data = np.concatenate((X_test, y_test.T), axis=1)

epochs = 10
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
    validation_acc.append(bp.old_evaluation(X_validation, y_validation))

    loss = bp.train(X_train, y_train, learning_rate = 0.5) #* 0.98**idx)
    epoch_losses.append(np.average(loss))
    losses = np.vstack([losses, loss]) if losses.size else loss
    train_acc.append(bp.old_evaluation(X_train, y_train, "accuracy"))
    train_rec.append(bp.old_evaluation(X_train, y_train, "recall"))
    train_pre.append(bp.old_evaluation(X_train, y_train, "precision"))
    train_f1.append(bp.old_evaluation(X_train, y_train, "f1"))


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

preds = []
trues = []
for i in np.arange(0,len(X_test)):
    preds.append(np.argmax(bp.predict(X_test[i])))
    trues.append(np.argmax(y_test[i]))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(trues, preds)

import seaborn as sns
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



for i in np.arange(0,len(X_train)):
    print(np.argmax(bp.predict(X_train[i])), " ", np.argmax(y_train[i]))

for i in np.arange(0,len(X_train)):
    print([ '%.2f' % elem for elem in bp.predict(X_train[i])], " ", y_train[i])

pred = np.empty((0, 10))
for ds in X_train:
    prediction = bp.predict(ds)
    #pred.append([int(round(i,0)) for i in bp.predict(ds)])
    filled = [0]*10
    filled[prediction.index(max(prediction))] = 1
    filled = np.array(filled)
    pred = np.vstack((pred,filled))
    bp.reset_neurons()



print("Simulation done")