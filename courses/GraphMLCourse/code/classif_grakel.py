#%% Classification and vizu using Grakel
import numpy as np
from grakel.datasets import fetch_dataset
from grakel.datasets.base import read_data
from grakel.kernels import WeisfeilerLehman, VertexHistogram
import zipfile

#%%
name = "MUTAG"
verbose = True
path = '/Users/titouanvayer/Documents/GraphMLcourse/Course_GraphKernel/TD/data' 
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
#%%
unique, counts = np.unique(y, return_counts=True)
# the classes are a bit imbalanced
#%%
import networkx as nx
import matplotlib.pyplot as plt
m = 12
nx_g = nx.from_numpy_array(G[m].get_adjacency_matrix())
i=0
for _,v in G[m].node_labels.items():
    nx.set_node_attributes(nx_g, {i : {'label':v}})
    i+=1
cols = {0:'r', 1: 'b', 2:'g'}
pos=nx.layout.kamada_kawai_layout(nx_g)
nx.draw_networkx(nx_g, 
                 with_labels=True, 
                 labels=nx.get_node_attributes(nx_g,'label'), 
                 node_color=[cols[v] for k,v in nx.get_node_attributes(nx_g, 'label').items()]
                 )
#%%
# Splits the dataset into a training and a test set
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)
#%% Init a graph kernel
wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=VertexHistogram)
#%%
K_train = wl_kernel.fit_transform(G_train)
K_test = wl_kernel.transform(G_test)
#%%
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)

y_pred = clf.predict(K_test)
# %%
print("%2.2f %%" %(round(accuracy_score(y_test, y_pred)*100)))
#%% compare with the random classifier 
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(G_train, y_train)
print("%2.2f %%" %(round(accuracy_score(y_test, dummy_clf.predict(G_test))*100)))

# %% Implement a cross validation procedure with StratifiedKFold and compute the CV score
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
errors_ = []
for i, (train_index, test_index) in enumerate(skf.split(G, y)):
    G_train, G_test = [G[i] for i in train_index], [G[i] for i in test_index]
    y_train, y_test = y[train_index], y[test_index]
    K_train = wl_kernel.fit_transform(G_train)
    K_test = wl_kernel.transform(G_test) 

    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test) 
    errors_.append(accuracy_score(y_test, y_pred))
#%%
print("Average CV accuracy = {0:.2f} +/- std = {1:.2f}".format(np.mean(errors_), np.std(errors_)))

#%%

# Question: comparer avec l'erreur précédent, est-ce une bonne mesure de l'erreur de généralisation du modèle COMPLET (ie Weisfehler model) ?
# do the nested validation procedure

class GK_classifier():
    def __init__(self, C=1, n_iter=5, normalize=True):
        self.C=C
        self.n_iter=n_iter
        self.normalize=normalize
        wl_kernel = WeisfeilerLehman(n_iter=self.n_iter, 
                                     normalize=self.normalize, 
                                     base_graph_kernel=VertexHistogram)
        self.graphkernel = wl_kernel
        self.svc=SVC(kernel='precomputed', C=self.C)

    def fit(self, X, y=None):
        K = self.graphkernel.fit_transform(X)
        self.svc.fit(K, y)

    def predict(self,X):
        K=self.graphkernel.transform(X)
        return self.svc.predict(K)
    
def do_cv(X, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    errors_ = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test) 
        errors_.append(accuracy_score(y_test, y_pred))
    return errors_
#%%
wl_classifier = GK_classifier()
do_cv(G, y, wl_classifier)
# %%
from sklearn.model_selection import ParameterGrid

parameters = {'C':[1e-4, 1e-3, 1e-1,  1, 10, 100, 1000], 'n_iter':[1, 2, 3, 5, 10]}
param_grid = ParameterGrid(parameters)

def frozendict(d: dict):
    keys = sorted(d.keys())
    return tuple((k, d[k]) for k in keys)

def do_nested_cv(X, y, class_model, param_grid, inner_cv=5, outer_cv=5, higher_the_better=True):
    # estimate the generalization error of the whole model
    outer_skf = StratifiedKFold(n_splits=outer_cv)
    errors_ = []
    chosen_parameters = []
    for i, (train_index, test_index) in enumerate(outer_skf.split(X, y)):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        res_of_param = {}
        # loop for selecting the best hyperparameter
        for param in param_grid: 
            instantiated_model = class_model(**param)
            res = do_cv(X_train, y_train, instantiated_model, n_splits=inner_cv)
            res_of_param[frozendict(param)] = np.mean(res)

        if higher_the_better:
            best_param = max(res_of_param, key=res_of_param.get) #the higher accuracy the better
        else:
            best_param = min(res_of_param, key=res_of_param.get) 
        chosen_parameters.append(best_param)
        best_model = class_model(**{elt[0] : elt[1] for elt in best_param})

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        errors_.append(score)
        print('--- index {} done ---'.format(i))
        print('Best hyperparam chosen are {}'.format(best_param))
        print('Score is {:.2f}'.format(score))

    return errors_, chosen_parameters
#%%
res, chosen_params = do_nested_cv(G, y, GK_classifier, param_grid, inner_cv=10, outer_cv=10)
#%%
print("Average nested CV accuracy = {0:.2f} +/- std = {1:.2f}".format(np.mean(res), np.std(res)))

# %%
