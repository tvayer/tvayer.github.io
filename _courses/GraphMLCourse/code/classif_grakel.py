# %% Classification and vizu using Grakel
from sklearn.svm import LinearSVC
from grakel.kernels import GraphletSampling, ShortestPath, VertexHistogram, WeisfeilerLehman
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from grakel.datasets import fetch_dataset
from grakel.datasets.base import read_data
import zipfile
import time
import grakel
print("grakel version {}".format(grakel.__version__))
print("networkx version {}".format(nx.__version__))
print("numpy version {}".format(np.__version__))
# %%
name = "MUTAG"
verbose = True
path = '/Users/titouanvayer/Documents/GraphMLcourse/Course_GraphKernel/TD/data'  # path do data
with zipfile.ZipFile(path+"/" + str(name) + '.zip', "r") as zip_ref:
    if verbose:
        print("Extracting dataset ", str(name) + "..")
    zip_ref.extractall()

if verbose:
    print("Parsing dataset ", str(name) + "..")

dataset = read_data(name,
                    with_classes=True,
                    prefer_attr_nodes=False,
                    prefer_attr_edges=False,
                    produce_labels_nodes=False,
                    is_symmetric=False,
                    as_graphs=True
                    )
G = dataset.data
y = dataset.target
# %%
unique, counts = np.unique(y, return_counts=True)
print(counts)
# %%
all_discrete_attributes = []
for g in G:
    nx_g = nx.from_numpy_array(g.get_adjacency_matrix())
    labels = set(g.node_labels.values())
    all_discrete_attributes = all_discrete_attributes + list(labels)
all_discrete_attributes = np.unique(all_discrete_attributes)
# the classes are a bit imbalanced we should do stratified cross-validation
# %%
m = 30
nx_g = nx.from_numpy_array(G[m].get_adjacency_matrix())
i = 0
for _, v in G[m].node_labels.items():
    nx.set_node_attributes(nx_g, {i: {'label': v}})
    i += 1
cmap = plt.cm.get_cmap('tab20')
cols = [cmap(i) for i in all_discrete_attributes]
pos = nx.layout.kamada_kawai_layout(nx_g)
nx.draw_networkx(nx_g,
                 with_labels=True,
                 labels=nx.get_node_attributes(nx_g, 'label'),
                 node_color=[cols[v]
                             for k, v in nx.get_node_attributes(nx_g, 'label').items()]
                 )
# %%
# Splits the dataset into a training and a test set
G_train, G_test, y_train, y_test = train_test_split(
    G, y, test_size=0.1, random_state=42)
# %% Init a graph kernel
# normalization to prevent diagonal dominance effect -> the diagonal is set to one
# K_ij <- K_ij / sqrt(K_ii) sqrt(K_jj
#  n_iter = h of the weisfeler lehman
# vertexhistogram is the base kernel
wl_kernel = WeisfeilerLehman(
    n_iter=5,
    normalize=True,
    base_graph_kernel=VertexHistogram)
# %%
K_train = wl_kernel.fit_transform(G_train)
K_test = wl_kernel.transform(G_test)
# behind the scene it does the same as
# K = wl_kernel.fit_transform(G)
# K_test2 = K[np.ix_(indices_test, indices_train)]
# we have K_test2 == K_test
# %%
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)

y_pred = clf.predict(K_test)
# %%
print("%2.2f %%" % (round(accuracy_score(y_test, y_pred)*100)))
# %% compare with the "random classifier"
# predict the most frequent class
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(G_train, y_train)
print("%2.2f %%" % (round(accuracy_score(y_test, dummy_clf.predict(G_test))*100)))

# %% Implement a cross validation procedure with StratifiedKFold and compute the CV score
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
errors_ = []
for i, (train_index, test_index) in enumerate(skf.split(G_train, y_train)):
    G_subtrain, G_valid = [G_train[i]
                           for i in train_index], [G_train[i] for i in test_index]
    y_subtrain, y_valid = y_train[train_index], y_train[test_index]
    K_subtrain = wl_kernel.fit_transform(G_subtrain)
    K_valid = wl_kernel.transform(G_valid)

    clf = SVC(kernel='precomputed')
    clf.fit(K_subtrain, y_subtrain)
    y_pred = clf.predict(K_valid)
    errors_.append(accuracy_score(y_valid, y_pred))
# %%
print(
    "Average CV accuracy = {0:.2f} +/- std = {1:.2f}".format(np.mean(errors_), np.std(errors_)))
# %% Compare with Graphflet and shortest-path kernel

graphlet_kernel = GraphletSampling(k=3, normalize=False)
# careful there is an error: when calling fit_transform it appears that it overwrite the data
shortest_path = ShortestPath(normalize=False, with_labels=False)
vertex_histogram = VertexHistogram(normalize=False)
wl_kernel = WeisfeilerLehman(
    n_iter=3,
    normalize=False,
    base_graph_kernel=VertexHistogram)
all_models = [
    wl_kernel,
    graphlet_kernel,
    shortest_path,
    vertex_histogram,
]
all_names = [
    'WL',
    'Graphlet',
    'SP',
    'Vertex',
]
errors = {}

dataset = read_data(name,
                    with_classes=True,
                    prefer_attr_nodes=False,
                    prefer_attr_edges=False,
                    produce_labels_nodes=False,
                    is_symmetric=False,
                    as_graphs=True
                    )
G = dataset.data
y = dataset.target
G_train, G_test, y_train, y_test = train_test_split(
    G, y, test_size=0.1, random_state=42)

for model, name_model in zip(all_models, all_names):
    errors[name_model] = []
    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    splits = skf.split(G_train, y_train)
    for i, (train_index, test_index) in enumerate(splits):
        st = time.time()
        G_subtrain, G_valid = [G_train[i] for i in train_index], [G_train[i]
                                                                  for i in test_index]
        y_subtrain, y_valid = y_train[train_index], y_train[test_index]

        K_subtrain = model.fit_transform(G_subtrain)
        K_valid = model.transform(G_valid)

        clf = SVC(kernel='precomputed')
        clf.fit(K_subtrain, y_subtrain)
        y_pred = clf.predict(K_valid)
        errors[name_model].append(accuracy_score(y_valid, y_pred))
        ed = time.time()
    print('Model {0} done in {1:.3f}s'.format(name_model, ed-st))

for name_model, all_errors in errors.items():
    print('CV error of model {0} is {1:.2f} +/- std = {2:.2f}'.format(
        name_model, np.mean(all_errors), np.std(all_errors)))
# %%
# Question: comparer avec l'erreur précédent, est-ce une bonne mesure de l'erreur de généralisation du modèle COMPLET (ie Weisfehler model) ?
# do the nested validation procedure


class BaselineClassifier():
    # very simple baseline, we consider some simple hand made features:
    # the number of nodes of the graph, number of edges, mean degree.
    # We do very simple linear support vector machine of these feature
    # that this svm with the linear kernel k(x,y) = <x,y>
    def __init__(self, C=1):
        self.C = C
        self.svc = LinearSVC(C=self.C)

    def graph_to_nx_graph(self, G):
        nx_g = nx.from_numpy_array(G.get_adjacency_matrix())
        i = 0
        for _, v in G.node_labels.items():
            nx.set_node_attributes(nx_g, {i: {'label': v}})
            i += 1
        return nx_g

    def graph_feature(self, G):
        nx_g = self.graph_to_nx_graph(G)
        n_nodes = nx_g.number_of_nodes()
        n_edges = nx_g.number_of_edges()
        mean_degree = np.mean([nx_g.degree[m]
                              for m in range(len(nx_g.nodes()))])
        feature = [n_nodes, n_edges, mean_degree]
        return feature

    def fit(self, G, y):
        all_features = []
        for g in G:
            feature = self.graph_feature(g)
            all_features.append(feature)
        X = np.array(all_features)
        self.svc.fit(X, y)

    def predict(self, G):
        all_features = []
        for g in G:
            feature = self.graph_feature(g)
            all_features.append(feature)
        X = np.array(all_features)
        return self.svc.predict(X)


class WLClassifier():
    # simple class for WeisfeilerLehman to unify the fit/predict between all methods
    def __init__(self, C=1, n_iter=5, normalize=True):
        self.C = C
        self.n_iter = n_iter
        self.normalize = normalize
        wl_kernel = WeisfeilerLehman(n_iter=self.n_iter,
                                     normalize=self.normalize,
                                     base_graph_kernel=VertexHistogram)
        self.graphkernel = wl_kernel
        self.svc = SVC(kernel='precomputed', C=self.C)

    def fit(self, X, y=None):
        K = self.graphkernel.fit_transform(X)
        self.svc.fit(K, y)

    def predict(self, X):
        K = self.graphkernel.transform(X)
        return self.svc.predict(K)


class VertexClassifier():
    # simple class for VertexHistogram to unify the fit/predict between all methods
    def __init__(self, C=1, normalize=True):
        self.C = C
        self.normalize = normalize
        base_kernel = VertexHistogram(normalize=self.normalize)
        self.graphkernel = base_kernel
        self.svc = SVC(kernel='precomputed', C=self.C)

    def fit(self, X, y=None):
        K = self.graphkernel.fit_transform(X)
        self.svc.fit(K, y)

    def predict(self, X):
        K = self.graphkernel.transform(X)
        return self.svc.predict(K)


def do_cv(X, y, model, n_splits=5, random_state=0, shuffle=True):
    # compute the average CV score
    skf = StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=shuffle)  # careful we should fix the seed to train/valid on the same data
    errors_ = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = [X[i] for i in train_index], [X[i]
                                                        for i in test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        errors_.append(accuracy_score(y_test, y_pred))
    return errors_


# %%
dataset = read_data(name,
                    with_classes=True,
                    prefer_attr_nodes=False,
                    prefer_attr_edges=False,
                    produce_labels_nodes=False,
                    is_symmetric=False,
                    as_graphs=True
                    )
G = dataset.data
y = dataset.target
# %%


def frozendict(d: dict):
    keys = sorted(d.keys())
    return tuple((k, d[k]) for k in keys)


def do_nested_cv(X, y, class_model, param_grid, inner_cv=5, outer_cv=5,
                 higher_the_better=True, random_state=0, shuffle=True):
    # estimate the generalization error of the whole model
    outer_skf = StratifiedKFold(
        n_splits=outer_cv, random_state=random_state, shuffle=shuffle)
    errors = []
    chosen_parameters = []
    for i, (train_index, test_index) in enumerate(outer_skf.split(X, y)):
        X_train, X_test = [X[i] for i in train_index], [X[i]
                                                        for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        res_of_param = {}
        # loop for selecting the best hyperparameter
        for param in param_grid:
            instantiated_model = class_model(**param)
            res = do_cv(X_train, y_train, instantiated_model,
                        n_splits=inner_cv)  # CV error of the model with hyperparam
            res_of_param[frozendict(param)] = np.mean(res)

        if higher_the_better:  # find the best parameters for each model
            # the higher accuracy the better
            best_param = max(res_of_param, key=res_of_param.get)
        else:
            best_param = min(res_of_param, key=res_of_param.get)
        chosen_parameters.append(best_param)
        best_model = class_model(**{elt[0]: elt[1] for elt in best_param})

        # train the best model on the train data
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        # evaluate the performance on the test
        score = accuracy_score(y_test, y_pred)
        errors.append(score)
        print('--- Outer index {} done ---'.format(i))
        print('Best hyperparam chosen are {}'.format(best_param))
        print('Score is {:.2f}'.format(score))

    return errors, chosen_parameters


# %%
parameters = {
    'C': [1e-4, 1e-3, 1e-1,  1, 10, 100, 1000],
    'n_iter': [1, 3, 5],
    'normalize': [False, True]
}
param_grid = ParameterGrid(parameters)
res, chosen_params = do_nested_cv(
    G, y, WLClassifier, param_grid, inner_cv=5, outer_cv=3)
# %%
print(
    "Average nested CV accuracy of WL = {0:.2f} +/- std = {1:.2f}".format(np.mean(res), np.std(res)))

# %%
parameters = {
    'C': [1e-4, 1e-3, 1e-1,  1, 10, 100, 1000],
    'normalize': [False]
}
param_grid = ParameterGrid(parameters)
res, chosen_params = do_nested_cv(
    G, y, VertexClassifier, param_grid, inner_cv=5, outer_cv=3)
# %%
print(
    "Average nested CV accuracy of VertexClassifier = {0:.2f} +/- std = {1:.2f}".format(np.mean(res), np.std(res)))
# %%
parameters = {
    'C': [1e-4, 1e-3, 1e-1,  1, 10, 100, 1000]
}
param_grid = ParameterGrid(parameters)
res, chosen_params = do_nested_cv(
    G, y, BaselineClassifier, param_grid, inner_cv=5, outer_cv=3)
# %%
print(
    "Average nested CV accuracy of BaselineClassifier = {0:.2f} +/- std = {1:.2f}".format(np.mean(res), np.std(res)))
# the very simple baseline is acutally very good ! Try the same on PTC_MR
# %%
