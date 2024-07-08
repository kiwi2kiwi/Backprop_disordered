from Neuron import *
import numpy as np
from sklearn.metrics import accuracy_score


class Backpropagation:
    def __init__(self, base_space):
        super(Backpropagation, self).__init__()
        self.base_space = base_space

        for neuron in self.base_space.Neuron_dict.values():
            neuron.wire()
# TUM ai medicine
# 


    def error_function(self, pre,tar):
        return (pre - tar)**2

    def deriv_error_function(self, pre,tar):
        #return ((1. / (1 + np.exp(-(pre - tar))))-0.5)
        return 2*(pre - tar)

    def compute_error(self, target):
        errors = []
        for idx, n in enumerate(self.base_space.output_set):
            pred = n.activation()
            if not self.base_space.fast:
                print("error: ", round(self.error_function(pred, target[idx]),4), " pred: ", pred, " targ: ", target[idx])
            errors.append(self.error_function(pred, target[idx]))
        return errors

    def predict(self, slice_of_data):
        self.reset_neurons()
        for idx, input_neuron in enumerate(self.base_space.input_set):
            input_neuron.set_input(slice_of_data[idx])
        prediction = []
        for o in self.base_space.output_set:
            prediction.append(o.activation())

        return prediction


    def backprop(self, target, learning_rate):
#        self.compute_error(target)
        for idx, n in enumerate(self.base_space.output_set):
            error_through_a_zero = self.deriv_error_function(n.activation(), target[idx])
            n.error_for_output_neuron = error_through_a_zero
            n.gradient_descent(learning_rate)
            self.reset_neurons()

        #for idx, n in enumerate(self.base_space.output_set):
            #n.gradient_descent(learning_rate)


    def train(self, x, y, learning_rate = 0.1):
        loss_array = None
        for idx, ds in enumerate(x):
            self.predict(ds)

            self.backprop(y[idx], learning_rate)
            if loss_array is None:
                loss_array = self.compute_error(y[idx])
            else:
                loss_array = np.vstack([loss_array, self.compute_error(y[idx])])
            if self.base_space.Visualization:
                self.base_space.draw_brain()
            self.reset_neurons()

        for neuron in self.base_space.neurons:
            neuron.change_weight()
        return loss_array

    def get_loss(self, x, y):
        loss_array = None
        for idx, ds in enumerate(x):
            self.predict(ds)
            if loss_array is None:
                loss_array = self.compute_error(y[idx])
            else:
                loss_array = np.vstack([loss_array, self.compute_error(y[idx])])
            if self.base_space.Visualization:
                self.base_space.draw_brain()
            self.reset_neurons()

        return loss_array

    def evaluation(self, x, y):
        pred = []
        for ds in x:
            pred.append([round(i,0) for i in self.predict(ds)])
            self.reset_neurons()
        target = y

        #print("accuraccy: ", accuracy_score(target, pred))
#        visual = np.concatenate((np.array([pred]), [target]), axis=0)
        return accuracy_score(target, pred)

    def reset_neurons(self):
        for n in self.base_space.neurons:
            n.reset_neuron()

    def reset_gradient_caches(self):
        for n in self.base_space.neurons:
            n.reset_gradient_cache()

#do_test(test_data)
#train(train_data)
#print("end")
