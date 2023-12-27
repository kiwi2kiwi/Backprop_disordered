from Neuron import *
import numpy as np



class Backpropagation:
    def __init__(self, base_space):
        super(Backpropagation, self).__init__()
        self.base_space = base_space

        for neuron in self.base_space.Neuron_dict.values():
            neuron.wire()



    def error_function(self, pre,tar):
        return round((pre - tar)**2,3)

    def deriv_error_function(self, pre,tar):
        return 2*(pre - tar)

    def compute_error(self, target):
        for idx, n in enumerate(self.base_space.output_set):
            pred = n.activation()
            print("error: ", self.error_function(pred, target[idx]))

    def predict(self, slice_of_data):
        for idx, input_neuron in enumerate(self.base_space.input_set):
            input_neuron.set_input(slice_of_data[idx])
        prediction = []
        for i in self.base_space.output_set:
            prediction.append(i.activation())
        return prediction


    def backprop(self, target):
        self.compute_error(target)
        learning_rate = 0.001
        for idx, n in enumerate(self.base_space.output_set):

            error_through_a_zero = self.deriv_error_function(n.activation(), target[idx])
            n.gradient_descent(error_through_a_zero, learning_rate)


    def train(self, x, y):

        for ds in np.arange(x.len):
            self.predict(x[ds])
            self.backprop([y[ds]])
            if not self.base_space.fast:
                self.base_space.draw_brain()
            for n in self.base_space.neurons:
                n.reset_neuron()


    def evaluation(self, data):
        pred = []
        for ds in data:
            pred.append(round(self.predict(ds[:-1])[0],0))
            for a in self.base_space.neurons:
                a.reset_neuron()
        target = data[:,-1]

        from sklearn.metrics import accuracy_score
        print("accuraccy: ", accuracy_score(target, pred))
        visual = np.concatenate((np.array([pred]), [target]), axis=0)
        print("done")



#do_test(test_data)
#train(train_data)
#print("end")
