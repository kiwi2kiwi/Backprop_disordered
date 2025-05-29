import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



class CustomNetwork(nn.Module):
    def __init__(self, adjacency_matrix, steps):
        super().__init__()
        self.steps = steps
        self.adj_matrix = adjacency_matrix
        size = self.adj_matrix.shape[0]
        self.layer = nn.Linear(size, size, bias=False)
        nn.init.kaiming_uniform_(self.layer.weight, a=np.sqrt(5))
        # nn.init.constant_(self.layer.weight, 0.1)
        # nn.init.constant_(self.layer.bias, 0.0)

        # Apply pruning mask
        self.apply_mask()

    def apply_mask(self):
        # Custom pruning from mask (transpose mask to match Linear's weight shape [out, in])
        prune.custom_from_mask(self.layer, name="weight", mask=self.adj_matrix.T)

    def forward(self, x, vis=False):
        batch_size = x.shape[0]
        full_input = torch.zeros((batch_size, self.layer.in_features), device=x.device)
        full_input[:, :x.shape[1]] = x
        out = full_input
        if vis:
            out_vis = out.detach().numpy()
            out_vis = np.round(out_vis, 2)
            print(out_vis)
        for _ in range(self.steps):
            out = self.layer(out)
            # out = self.relu(out)
            if vis:
                out_vis = out.detach().numpy()
                out_vis = np.round(out_vis, 2)
                print(out_vis)
        return out[:, -3:]  # Output neuron slice

def compute_metrics(y_true, y_pred):
    """
    Computes accuracy, precision, recall, and F1 score for multi-class classification.
    Assumes y_true is one-hot encoded and y_pred contains raw scores or softmax probs.
    """
    y_true_idx = torch.argmax(y_true, dim=1).cpu().numpy()
    y_pred_idx = torch.argmax(y_pred, dim=1).cpu().numpy()

    return [accuracy_score(y_true_idx, y_pred_idx),precision_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
        recall_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)]

    return {
        'accuracy': accuracy_score(y_true_idx, y_pred_idx),
        'precision': precision_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
        'recall': recall_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
        'f1': f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
    }

def running_pytorch_network(individual, neuron_space, connectivity):
    data_import = individual.get_data()
    X_train = data_import[0]
    X_val = data_import[1]
    y_train = data_import[2]
    y_val = data_import[3]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # connectivity = torch.tensor(
    #     [[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    #      [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    #      [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    #      [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)

    conn_tensor = torch.tensor(connectivity)
    model = CustomNetwork(adjacency_matrix= conn_tensor, steps=conn_tensor.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=1.1)
    criterion = nn.MSELoss()
    losses = []

    # Training loop
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        # model.apply_mask()
        if epoch % 10 == 0:
            compute_metrics(y_train_tensor, output)
            acc = (output.argmax(1) == y_train_tensor.argmax(1)).float().mean()
            recall = recall_score(y_train_tensor.argmax(1), output.argmax(1), average='macro')
            precision = precision_score(y_train_tensor.argmax(1), output.argmax(1), average='macro')
            f1 = f1_score(y_train_tensor.argmax(1), output.argmax(1), average='macro')
            print(
                f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, acc: {acc:.4f}, recall: {recall:.4f}, precision: {precision:.4f}, f1: {f1:.4f}")

    return compute_metrics(model(X_val_tensor),y_val_tensor)
    # # Final prediction
    # print("Predictions:", model(torch.tensor(X, dtype=torch.float32)).detach())

# --------------------------------------
# EXAMPLE TRAINING
# --------------------------------------

def debug_func():
    connectivity = torch.tensor([
        [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)

    input_nodes = [7, 8, 9, 10]  # [-4:]
    output_nodes = [4, 5, 6]  # [-7:-4]

    from sklearn import datasets
    from sklearn.utils import shuffle
    iris = datasets.load_iris()
    X = np.array(iris.data)
    y = np.array(iris.target)
    y_oh = np.eye(3)[y]
    X, y = shuffle(X, y_oh, random_state=42)
    X_train = X[:100]
    X_val = X[100:]
    y_train = np.array(y[:100])
    y_val = np.array(y[100:])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    model = CustomNetwork(adjacency_matrix=connectivity, steps=4)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        model.apply_mask()

        if epoch % 10 == 0:
            compute_metrics(y_train_tensor, output)
            acc = (output.argmax(1) == y_train_tensor.argmax(1)).float().mean()
            recall = recall_score(y_train_tensor.argmax(1), output.argmax(1), average='macro')
            precision = precision_score(y_train_tensor.argmax(1), output.argmax(1), average='macro')
            f1 = f1_score(y_train_tensor.argmax(1), output.argmax(1), average='macro')
            print(
                f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, acc: {acc:.4f}, recall: {recall:.4f}, precision: {precision:.4f}, f1: {f1:.4f}")


# debug_func()
#
#
# test_array = np.array([
# [1, 0, 0.5, 0,   0],
# [0, 1, 0.5, 0.5, 0],
# [0, 0, 0,   0,   0.5],
# [0, 0, 0.5, 0,   0],
# [0, 0, 0,   0,   0]
# ])
#
# neuron_activation = np.array([1,1,0,0,0])
#
# a_1 = neuron_activation @ test_array
#
# a_2 = a_1 @ test_array
#
# a_3 = a_2 @ test_array
#
# a_4 = a_3 @ test_array

