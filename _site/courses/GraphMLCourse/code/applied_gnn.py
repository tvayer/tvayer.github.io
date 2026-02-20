# %% TD GNN applied
import matplotlib.ticker as plticker
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.datasets import KarateClub, Planetoid, WebKB
import torch_geometric as pyg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
cmap = plt.cm.get_cmap('tab10')

# %%
# dataset = KarateClub()
dataset = Planetoid(root='./Cora', name='Cora')
# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#common-benchmark-datasets
# Print information
print(dataset)
print('------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
# %%
classes = dataset.y
_, counts = np.unique(classes.numpy(), return_counts=True)
print(counts)
nx_g = pyg.utils.to_networkx(dataset.data, to_undirected=True)

# %% Simple GCN model


class SimpleGCN(nn.Module):
    def __init__(self, d_in, d_inter,  d_out, n_layers=1, bias=True, dropout=0):
        super(SimpleGCN, self).__init__()
        self.n_layers = n_layers
        self.d_in = d_in
        self.d_out = d_out
        self.d_inter = d_inter
        self.bias = bias
        self.dropout = dropout
        layers = []
        if self.n_layers == 1:
            layers.append(nn.Linear(self.d_in, self.d_out, bias=self.bias))
        if n_layers > 1:
            layers.append(nn.Linear(self.d_in, self.d_inter, bias=self.bias))
            layers.append(nn.ReLU())
            for _ in range(self.n_layers - 1):
                layers.append(
                    nn.Linear(self.d_inter, self.d_inter, bias=self.bias))
                layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.d_inter, self.d_out, bias=self.bias))
        self.neural_net = nn.Sequential(*layers)

    def forward(self, X, L):
        for layer in self.neural_net:
            if (isinstance(layer, nn.Linear)):
                X = layer(L @ X)
                # Same as
                # if self.bias:
                #     X = L @ X @ layer.weight.T + layer.bias
                # else:
                #     X = L @ X @ layer.weight.T
            else:
                X = layer(X)
        return X
# %% Classification


def kipf_laplacian(A):
    # Corresponds to the GCN of Kipf & Welling
    tildeA = A + np.eye(A.shape[0])
    deg = tildeA @ np.ones(tildeA.shape[0])
    Dhalf = np.diag(1.0 / np.sqrt(deg))
    normalized_L = Dhalf @ tildeA @ Dhalf
    normalized_L_tensor = torch.Tensor(normalized_L).type(torch.float)
    return normalized_L_tensor


def normalized_L(A):
    # do not use the identity : might be bad
    deg = A @ np.ones(A.shape[0])
    D = np.diag(deg)
    normalized_L = D - A
    normalized_L_tensor = torch.Tensor(normalized_L).type(torch.float)
    return normalized_L_tensor


def sum_aggr(A):
    # correspond to sum aggr of Scarselli et al. 2008
    tildeA = A + np.eye(A.shape[0])
    return torch.Tensor(tildeA).type(torch.float)


def get_degree(A):
    deg = A @ np.ones(A.shape[0])
    return torch.tensor(deg.reshape(-1, 1)).type(torch.float)


# %%
A = np.array(nx.adjacency_matrix(nx_g).todense())
N = A.shape[0]
print(f'Size of the graph = {N}')

# index_nodes_train, index_nodes_test = train_test_split(
#     np.arange(N), test_size=0.33, random_state=33)

index_nodes_train = np.arange(N)[dataset.train_mask]
index_nodes_test = np.arange(N)[dataset.test_mask]

L_kipf = kipf_laplacian(A)
L_normalized = normalized_L(A)
L_sum = sum_aggr(A)
I = torch.eye(A.shape[0]).type(torch.float)
# %%
loss_fn = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)


def train_model(model, X, y,  L,
                index_nodes_train,
                index_nodes_test,
                lr=1e-1,
                n_epochs=500,
                weight_decay=0,
                # optim_alg='Adam'
                ):
    lr = lr
    # if optim_alg == 'Adam':
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    # elif optim_alg == 'AdamW':
    #     optimizer = torch.optim.AdamW(
    #         model.parameters(), lr=lr, weight_decay=weight_decay)
    y_train, y_test = y[index_nodes_train], y[index_nodes_test]
    losses = []
    train_accuracies = []
    test_accuracies = []
    for i in range(n_epochs):
        # Go in train mode to use dropout
        model.train()
        # Compute prediction and loss
        pred = model(X, L)
        pred_train = pred[index_nodes_train]
        pred_test = pred[index_nodes_test]
        loss = loss_fn(input=pred_train, target=y_train)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Go in eval mode to remove dropout (we do not want dropout for prediction)
        model.eval()
        # the softmax is useless but this is the intuitive way of doing classification
        y_pred_train = softmax(pred_train).argmax(1).numpy()
        train_acc = accuracy_score(y_pred_train, y_train.numpy())
        train_accuracies.append(train_acc)

        y_pred_test = softmax(pred_test).argmax(1).numpy()
        test_acc = accuracy_score(y_pred_test, y_test.numpy())
        test_accuracies.append(test_acc)

        if i % 100 == 0:
            print(
                'epoch = {0}: loss = {1:.5f}, train acc. = {2:.3f}, test acc. = {3:.3f}'
                .format(i, loss.item(), train_acc, test_acc))

        # if i > 2 and np.abs(losses[-1] - losses[-2])/np.abs(losses[-2]) <= 1e-6:
        #     break

    return losses, train_accuracies, test_accuracies


# %%
d_inter = 16
n_epochs = 500
lr = 1e-1
X = dataset.x  # features of the dataset
# X = get_degree(A)
# %% Check with a one layer GNN with L is the best for our task
params = {'d_in': X.shape[1],
          'd_inter': d_inter,
          'd_out': len(np.unique(classes)),  # number of classes for the output
          'n_layers': 1}
one_layer_gcn_L_kipf = SimpleGCN(**params)
one_layer_gcn_L_normalized = SimpleGCN(**params)
one_layer_gcn_L_sum = SimpleGCN(**params)
# the following will correspond exactly to a reg log
one_layer_gcn_identity = SimpleGCN(**params)

# %% Train all models
all_models = [one_layer_gcn_L_kipf, one_layer_gcn_L_normalized,
              one_layer_gcn_L_sum, one_layer_gcn_identity]
L_operators = [L_kipf, L_normalized, L_sum, I]
names = ['GCN Kipf', 'GCN L-norm', 'GCN sum aggr', 'RegLog']
results = {}
for model, L_op, name in zip(all_models, L_operators, names):
    print('---------------- Train model {} ----------------'.format(name))
    losses, train_accuracies, test_accuracies = train_model(
        model, X, classes, L_op, index_nodes_train, index_nodes_test,
        lr=5e-2,
        n_epochs=n_epochs)
    # store the results
    if name not in results.keys():
        results[name] = {}
    results[name]['loss'] = losses
    results[name]['train_accuracies'] = train_accuracies
    results[name]['test_accuracies'] = test_accuracies
# %% Plot the different results
# The L-norm is very bad: it is natural because not using the identity correspond to an aggregation
# that does no combine the current node feature with neighbours but inly the features of the neighbours
# that is z_u = COMB(z_u, AGGR (z_v, v in N(u))) = AGGR (z_v, v in N(u))
# Also (and this is nice) using the graph does improve the performances, the RegLog is bad
# But we also overfit a lot: we have more parameters that number of nodes
# number of params with zero hidden layers = nb_features * nb_classes + nb_classes = 10 038 >> n
# so we perfectly fit the loss to zero
# We will regularize th model using Adam and weight decay (see documentations)
fs = 15
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, res in results.items():
    ax[0].loglog(res['loss'], label=name, lw=3)
ax[0].set_xlabel('Epochs', fontsize=fs)
ax[0].set_ylabel('Cross entropy', fontsize=fs)
ax[0].grid()
ax[0].legend(fontsize=fs)
for i, (name, res) in enumerate(results.items()):
    ax[1].semilogx(res['train_accuracies'],
                   label='Train acc. {}'.format(name),
                   lw=3,
                   linestyle='--',
                   alpha=0.5,
                   color=cmap(i))
    ax[1].semilogx(res['test_accuracies'], label='Test acc. {}'.format(name),
                   lw=3,
                   linestyle='-',
                   color=cmap(i))
ax[1].set_xlabel('Epochs', fontsize=fs)
ax[1].set_ylabel('Accuracy', fontsize=fs)
ax[1].grid()
ax[1].legend(fontsize=fs-3, bbox_to_anchor=(1.02, 1.05),
             ncol=1, fancybox=True, shadow=True)
# this locator puts ticks at regular intervals
loc = plticker.MultipleLocator(base=0.05)
ax[1].yaxis.set_major_locator(loc)
fig.suptitle('Cora dataset', fontsize=fs+2)

# %% Train all models with regularization
# We must first re-initialize th models
one_layer_gcn_L_kipf = SimpleGCN(**params)
one_layer_gcn_L_normalized = SimpleGCN(**params)
one_layer_gcn_L_sum = SimpleGCN(**params)
one_layer_gcn_identity = SimpleGCN(**params)

all_models = [one_layer_gcn_L_kipf, one_layer_gcn_L_normalized,
              one_layer_gcn_L_sum, one_layer_gcn_identity]
L_operators = [L_kipf, L_normalized, L_sum, I]
names = ['GCN Kipf', 'GCN L-norm', 'GCN sum aggr', 'RegLog']
results = {}
for model, L_op, name in zip(all_models, L_operators, names):
    print('---------------- Train model {} ----------------'.format(name))
    losses, train_accuracies, test_accuracies = train_model(
        model, X, classes, L_op, index_nodes_train, index_nodes_test,
        lr=5e-2,
        n_epochs=n_epochs,
        weight_decay=0.05)
    # store the results
    if name not in results.keys():
        results[name] = {}
    results[name]['loss'] = losses
    results[name]['train_accuracies'] = train_accuracies
    results[name]['test_accuracies'] = test_accuracies

# %% Plot the results
# We see now that the losses do not does to zero anymore, and the performances are a bit better
# (above 0.77 in test for the Kipf and 0.79 for the aggr sum, also the reg log is way better from 0.45 to almost 0.6)
# But weight_decay should be tuned ideally for each model, same for the learning rate
fs = 15
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, res in results.items():
    ax[0].loglog(res['loss'], label=name, lw=3)
ax[0].set_xlabel('Epochs', fontsize=fs)
ax[0].set_ylabel('Cross entropy', fontsize=fs)
ax[0].grid()
ax[0].legend(fontsize=fs)
for i, (name, res) in enumerate(results.items()):
    ax[1].semilogx(res['train_accuracies'],
                   label='Train acc. {}'.format(name),
                   lw=3,
                   linestyle='--',
                   alpha=0.5,
                   color=cmap(i))
    ax[1].semilogx(res['test_accuracies'], label='Test acc. {}'.format(name),
                   lw=3,
                   linestyle='-',
                   color=cmap(i))
ax[1].set_xlabel('Epochs', fontsize=fs)
ax[1].set_ylabel('Accuracy', fontsize=fs)
ax[1].grid()
ax[1].legend(fontsize=fs-3, bbox_to_anchor=(1.02, 1.05),
             ncol=1, fancybox=True, shadow=True)
# this locator puts ticks at regular intervals
loc = plticker.MultipleLocator(base=0.05)
ax[1].yaxis.set_major_locator(loc)
fig.suptitle('Cora dataset', fontsize=fs+2)

# %% Can we achieve better accuracy with a stronger model with more layers ?
gcn = SimpleGCN(d_in=X.shape[1],
                d_inter=d_inter,
                d_out=len(np.unique(classes)),
                n_layers=2)

one_layer_gcn = SimpleGCN(d_in=X.shape[1],
                          d_inter=d_inter,
                          d_out=len(np.unique(classes)),
                          n_layers=1)
L_operators = [L_kipf, L_sum]  # we use the sum aggr for all models
names = ['2 layers GCN', '1 layer GCN']
all_models = [gcn, one_layer_gcn]
# different weight_decay for the more complex model
weight_decays = [0.001, 0.05]
# different learning rates
learning_rates = [5e-2, 5e-2]
results = {}
for model, L_op, name, weight_decay, lr in zip(all_models, L_operators, names, weight_decays, learning_rates):
    print('---------------- Train model {} ----------------'.format(name))
    losses, train_accuracies, test_accuracies = train_model(
        model, X, classes, L_op, index_nodes_train, index_nodes_test,
        lr=lr,
        n_epochs=n_epochs,
        weight_decay=weight_decay)
    # store the results
    if name not in results.keys():
        results[name] = {}
    results[name]['loss'] = losses
    results[name]['train_accuracies'] = train_accuracies
    results[name]['test_accuracies'] = test_accuracies
# %% We also add a very simple baseline which corresponds to prediction the most fequent class
y_train, y_test = classes[index_nodes_train], classes[index_nodes_test]
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(None, y_train.numpy())
dummy_pred = dummy_clf.predict(y_test.numpy())
accuracy_dummy = accuracy_score(dummy_pred, y_test.numpy())

# %%
fs = 15
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, res in results.items():
    ax[0].loglog(res['loss'], label=name, lw=3)
ax[0].set_xlabel('Epochs', fontsize=fs)
ax[0].set_ylabel('Cross entropy', fontsize=fs)
ax[0].grid()
ax[0].legend(fontsize=fs)
for i, (name, res) in enumerate(results.items()):
    ax[1].semilogx(res['train_accuracies'],
                   label='Train acc. {}'.format(name),
                   lw=3,
                   linestyle='--',
                   alpha=0.5,
                   color=cmap(i))
    ax[1].semilogx(res['test_accuracies'], label='Test acc. {}'.format(name),
                   lw=3,
                   linestyle='-',
                   color=cmap(i))

ax[1].hlines(accuracy_dummy, xmin=0, xmax=n_epochs+1,
             label='Dummy classifier',
             linestyle='--',
             lw=3,
             color='k')
ax[1].set_xlabel('Epochs', fontsize=fs)
ax[1].set_ylabel('Accuracy', fontsize=fs)
ax[1].grid()
ax[1].legend(fontsize=fs-3, bbox_to_anchor=(1.02, 1.05),
             ncol=1, fancybox=True, shadow=True)
loc = plticker.MultipleLocator(base=0.05)
ax[1].yaxis.set_major_locator(loc)
fig.suptitle('Cora dataset', fontsize=fs+2)

# %% On the Wisconsin dataset: heterophilic problem: the reg log is way better
dataset = WebKB(root='./Wisconsin', name='Wisconsin')
# Print information
print(dataset)
print('------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
# %%
classes = dataset.y
_, counts = np.unique(classes.numpy(), return_counts=True)
print(counts)
nx_g = pyg.utils.to_networkx(dataset.data, to_undirected=True)
# %%
A = np.array(nx.adjacency_matrix(nx_g).todense())
N = A.shape[0]
print(f'Size of the graph = {N}')

index_nodes_train, index_nodes_test = train_test_split(
    np.arange(N), test_size=0.2, random_state=33)

L_kipf = kipf_laplacian(A)
L_sum = sum_aggr(A)
I = torch.eye(A.shape[0]).type(torch.float)
X = dataset.x

gcn = SimpleGCN(d_in=X.shape[1],
                d_inter=d_inter,
                d_out=len(np.unique(classes)),
                n_layers=2)

reg_log = SimpleGCN(d_in=X.shape[1],
                    d_inter=d_inter,
                    d_out=len(np.unique(classes)),
                    n_layers=1)

one_layer_gcn = SimpleGCN(d_in=X.shape[1],
                          d_inter=d_inter,
                          d_out=len(np.unique(classes)),
                          n_layers=1)
L_operators = [L_kipf, L_sum, I]  # we use the sum aggr for all models
names = ['2 layers GCN', '1 layer GCN', 'Reg Log']
all_models = [gcn, one_layer_gcn, reg_log]
# different weight_decay for the more complex model
weight_decays = [0.05, 0.05, 0.05]
# different learning rates
learning_rates = [5e-2, 1e-2, 5e-2]
n_epochs = 800
results = {}
for model, L_op, name, weight_decay, lr in zip(all_models, L_operators, names, weight_decays, learning_rates):
    print('---------------- Train model {} ----------------'.format(name))
    losses, train_accuracies, test_accuracies = train_model(
        model, X, classes, L_op, index_nodes_train, index_nodes_test,
        lr=lr,
        n_epochs=n_epochs,
        weight_decay=weight_decay)
    # store the results
    if name not in results.keys():
        results[name] = {}
    results[name]['loss'] = losses
    results[name]['train_accuracies'] = train_accuracies
    results[name]['test_accuracies'] = test_accuracies

y_train, y_test = classes[index_nodes_train], classes[index_nodes_test]
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(None, y_train.numpy())
dummy_pred = dummy_clf.predict(y_test.numpy())
accuracy_dummy = accuracy_score(dummy_pred, y_test.numpy())
# %%
fs = 15
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, res in results.items():
    ax[0].loglog(res['loss'], label=name, lw=3)
ax[0].set_xlabel('Epochs', fontsize=fs)
ax[0].set_ylabel('Cross entropy', fontsize=fs)
ax[0].grid()
ax[0].legend(fontsize=fs)
for i, (name, res) in enumerate(results.items()):
    ax[1].semilogx(res['train_accuracies'],
                   label='Train acc. {}'.format(name),
                   lw=3,
                   linestyle='--',
                   alpha=0.5,
                   color=cmap(i))
    ax[1].semilogx(res['test_accuracies'], label='Test acc. {}'.format(name),
                   lw=3,
                   linestyle='-',
                   color=cmap(i))

ax[1].hlines(accuracy_dummy, xmin=0, xmax=n_epochs+1,
             label='Dummy classifier',
             linestyle='--',
             lw=3,
             color='k')
ax[1].set_xlabel('Epochs', fontsize=fs)
ax[1].set_ylabel('Accuracy', fontsize=fs)
ax[1].grid()
ax[1].legend(fontsize=fs-3, bbox_to_anchor=(1.02, 1.05),
             ncol=1, fancybox=True, shadow=True)
loc = plticker.MultipleLocator(base=0.05)
ax[1].yaxis.set_major_locator(loc)
fig.suptitle('Wisconsin dataset', fontsize=fs+2)
# %%
