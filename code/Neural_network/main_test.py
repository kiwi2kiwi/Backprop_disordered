import Neuron_space_predifined
import Backprop

import numpy as np

# Do you want visualization? Do you want the learning to be fast?


n = Neuron_space_predifined.NeuronSpace(fast = False)
n.spawn_neurons_axons()


bp = Backprop.Backpropagation(n)

x = [[0.1, 0.5],[0.6, 0.1]]
y = [[0, 1],[1, 0]]


x = [[1.0,	2.0], [2.0,	3.0], [1.5,	1.0], [2.5,	2.5], [0.5,	1.5], [2.0,	0.5], [1.0,	3.0], [3.0,	2.0], [0.5,	2.5], [1.5,	0.5], [3.0,	1.5], [2.0,	1.0], [1.5,	2.0], [0.5,	3.0], [2.5,	0.5], [3.0,	0.5], [1.0,	1.5], [0.5,	0.5], [2.0,	2.5], [1.5,	3.0]]

y = [[1,	0],[0,	1],[1,	0],[0,	1],[1,	0],[0,	1],[1,	0],[0,	1],[1,	0],[1,	0],[0,	1],[0,	1],[1,	0],[1,	0],[0,	1],[0,	1],[1,	0],[1,	0],[0,	1],[1,	0]]


train_losses = []
train_acc = []

for i in np.arange(1,100):
    train_loss = bp.train(x, y)
    train_losses.append(np.average(train_loss))
    #train_losses = np.vstack([train_losses, train_loss]) if train_losses.size else train_loss
    #train_losses.append(np.average(train_loss))
    #train_losses = np.vstack([train_losses, train_loss]) if train_losses.size else train_loss
    train_acc.append(bp.evaluation(x, y))
#    bp.predict(x[0])
#    bp.compute_error(y[0])
#    bp.backprop(y[0])
#    for ns in n.neurons:
#        ns.reset_neuron()



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