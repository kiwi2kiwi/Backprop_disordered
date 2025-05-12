import Neuron_space
import Backprop
import numpy as np
np.random.seed(1)

# Do you want visualization? Do you want the learning to be fast?

n = Neuron_space.NeuronSpace(fast = True, Visualization=False, neuron_number = 5)
n.spawn_neurons_axons(input_number=4, output_number=3)


bp = Backprop.Backpropagation(n)


from sklearn import datasets
iris = datasets.load_iris()
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def plot_metrics(train_acc,train_rec,train_pre,train_f1,epoch_losses,validation_acc,epoch_validation_losses):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.plot(np.arange(len(epoch_losses)), epoch_losses, label='train losses')
    ax1.plot(np.arange(len(epoch_validation_losses)), epoch_validation_losses, label='validation losses')
    ax1.set_title("Iris dataset losses")
    ax1.set_xlabel("epochs")
    ax2.plot(np.arange(len(train_acc)), train_acc, label='train accuracy')
    ax2.plot(np.arange(len(validation_acc)), validation_acc, label='validation accuracy')
    ax2.set_title("Iris dataset accuracy")
    ax2.set_xlabel("epochs")
    ax3.plot(np.arange(len(train_rec)), train_rec, label='train recall')
    ax3.plot(np.arange(len(train_f1)), train_f1, label='train F1')
    ax3.plot(np.arange(len(train_pre)), train_pre, label='train precision')
    ax3.set_title("Iris dataset metrics")
    ax3.set_xlabel("epochs")
    fig.legend()
    fig.show()

X = np.array(iris.data)
y = np.array(iris.target)
#X[:,0] = 0
#X[:,1] = 0
#X[:,2] = 0
#X[:,3] = 0


# y = y/2

y_oh = np.eye(3)[y]


X, y = shuffle(X, y_oh, random_state=10)
X_train = X[:100]
X_val = X[100:]
y_train = np.array(y[:100])
y_val = np.array(y[100:])

# X_train = X
# y_train = np.array([y])

# X_train = X[:100]
# X_val = X[100:]
# y_train = np.array([y[:100]])
# y_val = np.array([y[100:]])



std_slc = StandardScaler()
std_slc.fit(X_train)
X_train = std_slc.transform(X_train)
X_val = std_slc.transform(X_val)


#train_data = np.concatenate((X_train, y_train.T), axis=1)
#test_data = np.concatenate((X_val, y_val.T), axis=1)

epochs = 50
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
    #n.print_states()
    validation_loss = bp.get_loss(X_val, y_val)
    epoch_validation_losses.append(np.average(validation_loss))
    validation_losses = np.vstack([validation_losses, validation_loss]) if validation_losses.size else validation_loss
    validation_acc.append(bp.evaluation(X_val, y_val))

    #n.print_states()
    loss = bp.train(X_train, y_train, learning_rate = 0.5)
    epoch_losses.append(np.average(loss))
    losses = np.vstack([losses, loss]) if losses.size else loss
    # train_acc.append(bp.iris_evaluation(X_train, y_train))
    train_acc.append(bp.evaluation(X_train, y_train, "accuracy"))
    train_rec.append(bp.evaluation(X_train, y_train, "recall"))
    train_pre.append(bp.evaluation(X_train, y_train, "precision"))
    train_f1.append(bp.evaluation(X_train, y_train, "f1"))
    #n.print_states()

    print("epoch: ", (idx+1), "/", epochs)

plot_metrics(train_acc,train_rec,train_pre,train_f1,epoch_losses,validation_acc,epoch_validation_losses)

preds = []
trues = []
for i in np.arange(0,len(X_val)):
    preds.append(np.argmax(bp.predict(X_val[i])))
    trues.append(np.argmax(y_val[i]))

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
    print([ '%.2f' % elem for elem in bp.predict(X_train[i])], " ", y_train[i])

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
ax1.plot(np.arange(len(validation_acc)), validation_acc, label='validation accuracy')
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