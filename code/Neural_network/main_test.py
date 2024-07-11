import Neuron_space_predifined
import Backprop

import numpy as np

# Do you want visualization? Do you want the learning to be fast?


n = Neuron_space_predifined.NeuronSpace(fast = True)
n.spawn_neurons_axons()


bp = Backprop.Backpropagation(n)


# x = np.array([[1.0,	2.0], [2.0,	3.0], [1.5,	1.0], [2.5,	2.5], [0.5,	1.5], [2.0,	0.5], [1.0,	3.0], [3.0,	2.0], [0.5,	2.5], [1.5,	0.5], [3.0,	1.5], [2.0,	1.0], [1.5,	2.0], [0.5,	3.0], [2.5,	0.5], [3.0,	0.5], [1.0,	1.5], [0.5,	0.5], [2.0,	2.5], [1.5,	3.0]])
# y = np.array([[1,	0],[0,	1],[1,	0],[0,	1],[1,	0],[0,	1],[1,	0],[0,	1],[1,	0],[1,	0],[0,	1],[0,	1],[1,	0],[1,	0],[0,	1],[0,	1],[1,	0],[1,	0],[0,	1],[1,	0]])
# x_train = x[[1,2]]#,4,6]]
# y_train = y[[1,2]]#,4,6]]
# x_train = [[0.05,0.1],[0.1,0.05]]
# y_train = [[0.01,0.99],[0.99,0.01]]

x_train = [[0.05,0.1],[0.05,0.1]]
y_train = [[0.01,0.99],[0.01,0.99]]
train_losses = []
train_acc = []



for i in np.arange(1,100):
    train_loss = bp.train(x_train, y_train, learning_rate=0.5)
    train_losses.append(np.average(train_loss))

    # for i in n.Axon_dict.values():
    #     print(i.name," ", i.weight)
    # for i in n.Neuron_dict.values():
    #     print(i.name," ", i.bias)
    print("prediction: ", bp.predict(x_train[0]))
    # print("prediction: ", bp.predict(x_train[1]))
    #train_acc.append(bp.evaluation([x_train], [y_train]))
    print("epoch: ", i, " loss: ",train_loss)






bp.evaluation(x, y)

for i in np.arange(0,len(x_train)):
    print([ '%.2f' % elem for elem in bp.predict(x_train[i])], " ", y_train[i])


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()

ax.plot(np.arange(len(train_losses)), train_losses, label='train losses')
ax.set_title("TEST dataset losses")
ax.set_xlabel("epochs")
fig.legend()
fig.show()

ax1.plot(np.arange(len(train_acc)), train_acc, label='train accuracy')
ax1.set_title("TEST dataset accuracy")
ax1.set_xlabel("epochs")
fig1.legend()
fig1.show()

#bp.backprop([y])

#bp.train(x,y)
#bp.evaluation(test_data)

print("Simulation done")