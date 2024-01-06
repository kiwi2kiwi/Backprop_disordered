# Backprop_disordered
Deep feedforward neural network without layer structure\
Since layer structure is broken i name this BrokeNN.\
Runs on the Iris dataset.\
Practical application of Backpropagation_diy project.


Only fancy libraries needed are numpy for array convenience, sklearn for dataset, matplotlib for visualization


run main.py \
This line is important:\
n = Neuron_space.NeuronSpace(fast = True, neuron_number = 10)

fast means that you dont want to visualize learning progress of the nn in a 3D space.\
neuron_number is the total number of hidden neurons.\
Anything above 10 will hardly converge due to exploding gradients.\
Above 10 will take very very long to learn due to exploding learning parameters.\

This was not done to create a new fancy advanced network. It's just a funny lil project.

You're not allowed to use this for commercial purposes.
