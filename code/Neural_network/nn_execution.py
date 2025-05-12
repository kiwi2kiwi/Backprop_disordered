import Neural_network.Neuron_space
import Neural_network.Backprop
import numpy as np
import math
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(2)


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

def running_the_network(individual, n, viz = False):
    bp = Neural_network.Backprop.Backpropagation(n)

    data_import = individual.get_data()
    X_train = data_import[0]
    X_val = data_import[1]
    y_train = data_import[2]
    y_val = data_import[3]

    epochs = 10
    train_acc = []
    train_rec = []
    train_pre = []
    train_f1 = []
    losses = np.array([])
    epoch_losses = []

    validation_acc = []
    validation_rec = []
    validation_pre = []
    validation_f1 = []
    validation_losses = np.array([])
    epoch_validation_losses = []

    for idx,i in enumerate(np.arange(0,epochs)):
        #n.print_states()


        validation_loss = bp.get_loss(X_val, y_val)
        epoch_validation_losses.append(np.average(validation_loss))
        validation_losses = np.vstack([validation_losses, validation_loss]) if validation_losses.size else validation_loss
        validation_acc.append(bp.evaluation(X_val, y_val, "accuracy"))
        validation_rec.append(bp.evaluation(X_val, y_val, "recall"))
        validation_pre.append(bp.evaluation(X_val, y_val, "precision"))
        validation_f1.append(bp.evaluation(X_val, y_val, "f1"))

        #n.print_states()
        for i in n.Axon_dict.values():
            if math.isnan(i.get_weight()):
                print("pause")
        loss = bp.train(X_train, y_train, learning_rate = 0.001)
        epoch_losses.append(np.average(loss))
        losses = np.vstack([losses, loss]) if len(losses) else loss
        train_acc.append(bp.evaluation(X_train, y_train, "accuracy"))
        train_rec.append(bp.evaluation(X_train, y_train, "recall"))
        train_pre.append(bp.evaluation(X_train, y_train, "precision"))
        train_f1.append(bp.evaluation(X_train, y_train, "f1"))
        #n.print_states()

        # print("epoch: ", (idx+1), "/", epochs)

    if viz:
        plot_metrics(train_acc,train_rec,train_pre,train_f1,epoch_losses,validation_acc,epoch_validation_losses)

    preds = []
    trues = []
    for i in np.arange(0,len(X_val)):
        preds.append(np.argmax(bp.predict(X_val[i])))
        trues.append(np.argmax(y_val[i]))

    if viz:
        cm = confusion_matrix(trues, preds)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()



        for i in np.arange(0,len(X_train)):
            print([ '%.2f' % elem for elem in bp.predict(X_train[i])], " ", y_train[i])

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
        print("stop")

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

    # print("Simulation done")
    return [np.average(validation_acc, axis=1)[-1],np.average(validation_pre, axis=1)[-1],np.average(validation_rec, axis=1)[-1],np.average(validation_f1, axis=1)[-1]]
