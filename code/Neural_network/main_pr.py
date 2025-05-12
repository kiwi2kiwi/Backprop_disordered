import numpy as np
import scipy
from scipy.io import loadmat
import pandas as pd
test = scipy.io.loadmat("C:/Users/yanni/Desktop/6. Semester/ML/pycharms/data/TestingData_N1000_d2_M4-v7.mat")
train = scipy.io.loadmat("C:/Users/yanni/Desktop/6. Semester/ML/pycharms/data/TrainingData_N800_d2_M4-v7.mat")
# conversion to pandas DataFrame for easier handling
df_train = pd.DataFrame(
    data=np.array([train["Labels"].flatten(), train["DataVecs"][:, 0], train["DataVecs"][:, 1]]).T,
    columns=["Label", "X_Coordinate", "Y_Coordinate"]
)
df_test = pd.DataFrame(
    data=np.array([test["TestLabels"].flatten(), test["TestVecs"][:, 0], test["TestVecs"][:, 1]]).T,
    columns=["Label", "X_Coordinate", "Y_Coordinate"]
)

import Neuron_space
import Backprop

np.random.seed(1)

# Do you want visualization? Do you want the learning to be fast?

n = Neuron_space.NeuronSpace(fast = True, Visualization=False, neuron_number = 2)
n.spawn_neurons_axons(input_number=2, output_number=4)


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

# X = df_train[["X_Coordinate","Y_Coordinate"]]
# y = df_train["Label"]
# X = np.array(mnist.data)
# y = np.array(mnist.target)

X_train = np.array(df_train[["X_Coordinate","Y_Coordinate"]].sample(frac=1))
y_train = np.array(df_train["Label"].sample(frac=1))

X_validation = np.array(df_test[["X_Coordinate","Y_Coordinate"]].sample(frac=1))
y_validation = np.array(df_test["Label"].sample(frac=1))

f = np.zeros([len(y_train),4])
for idx, i in enumerate(y_train):
    f[idx, int(i-1)] = 1

y_train = f

f = np.zeros([len(y_validation),4])
for idx, i in enumerate(y_validation):
    f[idx, int(i-1)] = 1

y_validation = f

# train_len = 50
# val_len = 55
# test_len = 85
# X, y = shuffle(X, y, random_state=1)
# X_train = X[[0,2,4,5]]
# #X_train = X[:train_len]
# X_validation = X[train_len:val_len]
# X_test = X[val_len:test_len]
# y_train = y[[0,2,4,5]]
# #y_train = np.array(y[:train_len])
# y_validation = np.array(y[train_len:val_len])
# y_test = np.array(y[val_len:test_len])
#
# std_slc = StandardScaler()
# std_slc.fit(X_train)
# X_train = std_slc.transform(X_train)
# X_validation = std_slc.transform(X_validation)
# X_test = std_slc.transform(X_test)



#train_data = np.concatenate((X_train, y_train.T), axis=1)
#test_data = np.concatenate((X_test, y_test.T), axis=1)

epochs = 100
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

    loss = bp.train(X_train, y_train, learning_rate = 1) #* 0.98**idx)
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