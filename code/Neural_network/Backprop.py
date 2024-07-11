from Neuron import *
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
        return 0.5* ((tar - pre)**2)

    def deriv_error_function(self, pre,tar):
        return 0.5* (2*(pre - tar))


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

        for idx, n in enumerate(self.base_space.output_set):
            # print("Backprop from output neuron: ", n.name)

#            unique, counts = np.unique(target, return_counts=True)
#            target_dict = dict(zip(unique, counts))
#            class_balancer = sum(target_dict.values()) / target_dict[target[idx]]
#            class_balancer = 1 / (1 + np.exp(-class_balancer))

            n_out = n.activation()
            y_true = target[idx]
            error_through_net_out = self.deriv_error_function(n_out, y_true)
            n.error_for_output_neuron = error_through_net_out
        n.gradient_descent(learning_rate)# * class_balancer)
        self.reset_neuron_gradients()

        #for idx, n in enumerate(self.base_space.output_set):
            #n.gradient_descent(learning_rate)


    def train(self, x, y, learning_rate = 0.1):
        loss_array = None
        for idx, ds in enumerate(x):
            # print("sample: ",idx)
            self.predict(ds)
            self.backprop(y[idx], learning_rate)
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
            if self.base_space.Visualization:
                self.base_space.draw_brain()
            self.reset_neurons()

        return loss_array

    def evaluation(self, x, y, metric = "acc"):
        pred = []
        for ds in x:
            pred.append([int(round(i,0)) for i in self.predict(ds)])
            self.reset_neurons()

        target = np.asmatrix(y)
        pred = np.asmatrix(pred)
        if metric == "acc":
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
        for n in self.base_space.neurons:
            n.reset_neuron()

    def reset_neuron_gradients(self):
        for n in self.base_space.neurons:
            n.reset_neuron_gradient_calculations()

    # def iris_evaluation(self, x, y):
    #     pred = []
    #     for ds in x:
    #         pred.append([int(round(i*2,0)) for i in self.predict(ds)])
    #         self.reset_neurons()
    #     target = y*2
    #
    #     target = np.asmatrix(target)
    #     pred = np.asmatrix(pred)
    #     accs = []
    #     for f in np.arange(0, target.shape[1]):
    #         accs.append(sklearn.metrics.accuracy_score(target[:, f], pred[:, f]))
    #
    #     return np.mean(accs)
