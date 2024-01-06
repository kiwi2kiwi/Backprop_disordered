import Neuron_space
import Neuron_space_predifined
import Backprop

import numpy as np

# Do you want visualization? Do you want the learning to be fast?


n = Neuron_space_predifined.NeuronSpace(fast = False)
n.spawn_neurons_axons()


bp = Backprop.Backpropagation(n)

x = [[0.1, 0.5]]
y = [[0.05, 0.95]]

for i in np.arange(1,10):
    bp.predict(x[0])
    bp.compute_error(y[0])
    bp.backprop(y[0])
    for ns in n.neurons:
        ns.reset_neuron()










bp.backprop([y])

bp.train(x,y)
#bp.evaluation(test_data)

print("Simulation done")