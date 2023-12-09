from Neuron import *
import numpy as np

from sklearn import datasets
iris = datasets.load_iris()
from sklearn.utils import shuffle

X = np.array(iris.data)
y = np.array(iris.target)
X, y = shuffle(X, y)
X_train = X[:100]
X_test = X[100:]
y_train = np.array([y[:100]])
y_test = np.array([y[100:]])

train_data = np.concatenate((X_train, y_train.T), axis=1)
test_data = np.concatenate((X_test, y_test.T), axis=1)

input_layer = [Input_Neuron(0), Input_Neuron(1), Input_Neuron(2), Input_Neuron(3)]
neuron_number = 4

weights = []

layer_layout = [[1]]

layers = []
active_neurons = []
for idx, l in enumerate(layer_layout):
    layer = []
    for neur in l:
        n = Neuron(neuron_number)
        neuron_number += 1

        if idx == 0:
            for i in input_layer:
                n.parent_connections[i] = [i, 100, []]
        else:
            for i in layers[idx-1]:
                n.parent_connections[i] = [i, 100, []]
        layer.append(n)
        active_neurons.append(n)
    layers.append(layer)

for layer in layers:
    for n in layer:
        n.wire()



def error_function(pre,tar):
    return round((pre - tar)**2,3)

def deriv_error_function(pre,tar):
    return 2*(pre - tar)

def compute_error(target):
    for idx, n in enumerate(layers[-1]):
        pred = n.activation()
        print("error: ", error_function(pred, target[idx]))

def predict(slice_of_data):
    for idx, inp in enumerate(input_layer):
        inp.set_input(slice_of_data[idx])
    prediction = []
    for i in layers[-1]:
        prediction.append(i.activation())
    return prediction


def backprop(target):
    compute_error(target)
    learning_rate = 0.005
    for idx, n in enumerate(layers[-1]):
        error_through_a_zero = deriv_error_function(n.activation(), target[idx])
        n.gradient_descent(error_through_a_zero, learning_rate)


def train(data):

    for ds in data:
        predict(ds[:-1])
        backprop([ds[-1]])


train(train_data)

def do_test(data):
    pred = []
    for ds in data:
        pred.append(round(predict(ds[:-1])[0],0))
        for a in active_neurons:
            a.reset_neuron()
    target = data[:,-1]

    from sklearn.metrics import accuracy_score
    print("accuraccy: ", accuracy_score(target, pred))
    visual = np.concatenate((np.array([pred]), [target]), axis=0)
    print("done")



do_test(test_data)


print("end")