import numpy
from Neural_network.Neuron import *
import numpy as np
import sklearn.metrics


class Backpropagation:
    def __init__(self, base_space):
        super(Backpropagation, self).__init__()
        self.base_space = base_space
        self.avg_gradient_update = []

        for neuron in self.base_space.Neuron_dict.values():
            neuron.wire()



    def error_function(self, pre,tar):
        return ((tar - pre) **2)

    def deriv_error_function(self, pre,tar):
        return (2 * (pre - tar))


    def compute_error(self, target):
        errors = []
        for idx, n in enumerate(self.base_space.output_neuron_dict.values()):
            pred = n.activation()
            if not self.base_space.fast:
                print("error: ", round(self.error_function(pred, target[idx]),4), " pred: ", pred, " targ: ", target[idx])
            errors.append(self.error_function(pred, target[idx]))
        return errors

    def predict(self, slice_of_data):
        self.reset_neurons()
        for idx, input_neuron in enumerate(self.base_space.input_neuron_dict.values()):
            input_neuron.set_input(slice_of_data[idx])
        prediction = []
        for o in self.base_space.output_neuron_dict.values():
            prediction.append(o.activation())

        return prediction


    def backprop(self, target, learning_rate, slice_of_data):

        for idx, n in enumerate(self.base_space.output_neuron_dict.values()):
            if self.base_space.verbal:
                print("Backprop from output neuron: ", n.name)

            #            unique, counts = np.unique(target, return_counts=True)
            #            target_dict = dict(zip(unique, counts))
            #            class_balancer = sum(target_dict.values()) / target_dict[target[idx]]
            #            class_balancer = 1 / (1 + np.exp(-class_balancer))
            self.reset_neurons()
            self.reset_neuron_gradients()
            for input_idx, input_neuron in enumerate(self.base_space.input_neuron_dict.values()):
                input_neuron.set_input(slice_of_data[input_idx])
            n_out = n.activation()
            y_true = target[idx]
            error_through_net_out = self.deriv_error_function(n_out, y_true)
            n.error_for_output_neuron = error_through_net_out
            n.gradient_descent(learning_rate, depth_counter=1)# * class_balancer)


        # for idx, n in enumerate(self.base_space.output_neuron_dict.values()):


        #for idx, n in enumerate(self.base_space.output_set):
        #n.gradient_descent(learning_rate)


    def train(self, x, y, learning_rate = 0.1):
        loss_array = None
        for idx, ds in enumerate(x):
            # print("sample: ",idx)
            self.predict(ds)
            self.backprop(y[idx], learning_rate, ds)
            if loss_array is None:
                loss_array = self.compute_error(y[idx])
            else:
                loss_array = np.vstack([loss_array, self.compute_error(y[idx])])
            if self.base_space.Visualization:
                self.base_space.draw_brain()
            self.reset_neuron_gradients()

        self.avg_gradient_update=[]
        return loss_array

    def get_loss(self, x, y):
        loss_array = None
        for idx, ds in enumerate(x):
            self.predict(ds)
            if loss_array is None:
                loss_array = self.compute_error(y[idx])
            else:
                loss_array = np.vstack([loss_array, self.compute_error(y[idx])])
            if not self.base_space.fast:
                self.base_space.draw_brain()
            self.reset_neurons()

        return loss_array

    def evaluation(self, x, y, metric = "accuracy"):
        pred = []
        for ds in x:
            pred.append([int(round(i,0)) for i in self.predict(ds)])
            self.reset_neurons()

        target = np.asmatrix(y)
        pred = np.asmatrix(pred)
        if metric == "accuracy":
            accs = []
            for f in np.arange(0, target.shape[1]):
                accs.append(sklearn.metrics.accuracy_score(target[:, f], pred[:, f]))

            return np.mean(accs)
        if metric == "recall":
            recalls = []
            for f in np.arange(0, target.shape[1]):
                try:
                    recalls.append(sklearn.metrics.recall_score(target[:, f], pred[:, f], zero_division=1, average='weighted'))
                except:
                    print("stop")
            return np.mean(recalls)
        if metric == "precision":
            precisions = []
            for f in np.arange(0, target.shape[1]):
                precisions.append(sklearn.metrics.precision_score(target[:, f], pred[:, f], zero_division=1, average='weighted'))
            return np.mean(precisions)
        if metric == "f1":
            f1s = []
            for f in np.arange(0, target.shape[1]):
                f1s.append(sklearn.metrics.f1_score(target[:, f], pred[:, f], zero_division=1, average='weighted'))
            return np.mean(f1s)

    def reset_neurons(self):
        for n in self.base_space.Neuron_dict.values():
            n.reset_neuron()

    def reset_neuron_gradients(self):
        for n in self.base_space.Neuron_dict.values():
            n.reset_neuron_gradient_calculations()

    def iris_evaluation(self, x, y, metric = "acc"):
        pred = []
        for ds in x:
            try:
                prediction = self.predict(ds)

                pred_temp = numpy.zeros_like(prediction)
                pred_temp[numpy.argmax(prediction)] = 1
                pred.append(pred_temp)
                # pred.append([int(round(i,0)) for i in self.predict(ds)])
            except:
                print("stop")
                [int(round(i * 1, 0)) for i in self.predict(ds)]
                [int(round(i * 1, 0)) for i in self.predict(ds)]
                [int(round(i * 1, 0)) for i in self.predict(ds)]
                [int(round(i * 1, 0)) for i in self.predict(ds)]
            self.reset_neurons()

        target = y
        # target = np.argmax(target, axis=1)
        # pred = np.argmax(pred, axis=1)
        try:
            if metric == "acc":
                acc = sklearn.metrics.accuracy_score(target, pred)
                return acc
            if metric == "recall":
                recall = sklearn.metrics.recall_score(target, pred, zero_division=1, average=None)
                return recall
            if metric == "precision":
                precision = sklearn.metrics.precision_score(target, pred, zero_division=1, average=None)
                return precision
            if metric == "f1":
                f1 = sklearn.metrics.f1_score(target, pred, zero_division=1, average=None)
                return f1
        except:
            return 0
