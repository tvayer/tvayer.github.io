#%% Temperature
# Data obtained from https://github.com/leouieda/global-temperature-data
# These are obtained from http://berkeleyearth.lbl.gov
# %% Download data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cmap = plt.cm.get_cmap('tab10')
from os import listdir
from os.path import isfile, join
from sklearn.kernel_ridge import KernelRidge

#%%
data_path = './data/temperature' # path where data are stored
all_countries = []
for f in listdir(data_path):
    if isfile(join(data_path, f)) and f.endswith('.csv'):
        all_countries.append(f.replace('.csv',''))
print(all_countries)
#%%
country_name = 'peru'
assert country_name in all_countries

temp = pd.read_csv(data_path+'/'+country_name+'.csv', names=['date','temp'])
temp.head()
#%%
print('Number of points = {}'.format(temp.temp.shape))
# %% Q1: c'est bruité et il y a un peu de saisonnalité
fs=18
fig, ax = plt.subplots(1, 1, figsize=(10,5))
ax.plot(temp.date, temp.temp, lw=2, c=cmap(0), marker='+', markersize=5, linestyle='')
ax.set_xlabel('Date', fontsize=fs)
ax.set_ylabel('temp (°C)', fontsize=fs)
ax.set_title('Monthly avg temperature of {0}'.format(country_name), fontsize=fs+3)

# %% With sklearn

X = np.array(range(temp.date.shape[0])).reshape(-1,1)
y = np.array(temp.temp)
tmax = X.shape[0]


#%% Influence de l'entrainement : message on peut pas prédire ce qu'on ne voit pas dans les données
fs=18
fig, ax = plt.subplots(1, 2, figsize=(10,5))
for i, ratio in enumerate([0.5, 0.8]):
    train_idx = int(np.floor(ratio*X.shape[0]))
    X_train, X_test = X[0:train_idx,], X[train_idx+1:,]
    y_train, y_test = y[0:train_idx], y[train_idx+1:]

    kernel_ridge = KernelRidge(kernel="poly", degree=2, alpha=1, coef0=1)
    kernel_ridge.fit(X_train, y_train)
    pred = kernel_ridge.predict(X)

    ax[i].plot(temp.date, temp.temp, lw=2, c=cmap(0), 
            marker='+', markersize=5, linestyle='', alpha=0.6)
    ax[i].plot(temp.date, pred, lw=3, c=cmap(1), linestyle='-', 
            label='Prediction', alpha=0.8)
    ax[i].vlines(x=temp.date[train_idx], ymin=y.min()-0.5, ymax=y.max()+0.5, 
            linestyle='--', color='grey', lw=3, label='Training set')
    ax[i].set_xlabel('Date', fontsize=fs)
    ax[i].set_ylabel('temp (°C)', fontsize=fs)
    ax[i].legend(loc='upper left', fontsize=fs)
    ax[i].set_ylim([y.min()-0.5, y.max()+0.5])
    ax[i].set_title('Train size = {0:.0%}'.format(ratio), fontsize=fs)
fig.suptitle('Monthly avg temperature of {0}'.format(country_name), fontsize=fs+3)
fig.tight_layout()
#fig.savefig('./wecannotpredictwhatwecannotsee.pdf')
#%% Gaussian kernel 
# Make the interpretation of the parameter about the neighbours because f(x) = sum_j theta_j k(t - t_i)
# sigma can be viewed as the characteristic length scale of the
# problem being learned (the scale on which changes of f
# take place), as discernible from the data (and thus dependent on, e.g., the number of training samples). lambda controls
# the leeway the model has to fit the training point
# https://www.chem.uci.edu/~kieron/dft/pubs/VSLR15.pdf

all_gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
fig, ax = plt.subplots(2, 3, figsize=(10,5))
r, c = 0, 0
for i, gamma in enumerate(all_gammas):
    kernel_ridge = KernelRidge(kernel="rbf", gamma=gamma, alpha=1)
    im = ax[r, c].imshow(kernel_ridge._get_kernel(X))
    ax[r, c].set_title('RBF param = {}'.format(gamma))
    plt.colorbar(im, ax=ax[r, c])
    c+=1
    if i == 2:
        r+=1
        c=0
fig.tight_layout()
#fig.savefig('./scaleinfluence.pdf')
#%% Try to find the best params
from sklearn.model_selection import ParameterGrid

ratio = 0.8
train_idx = int(np.floor(ratio*X.shape[0]))
X_train, X_test = X[0:train_idx,], X[train_idx+1:,]
y_train, y_test = y[0:train_idx], y[train_idx+1:]
all_gammas = np.logspace(-5, -2, 20)
all_alphas = np.logspace(-5, 3, 20)
all_params = {'gamma': all_gammas, 
              'alpha': all_alphas}

param_grid = ParameterGrid(all_params)
print('Parameters to test = {}'.format(len(list(param_grid))))
error = {}
for dic_param in param_grid:
    instantiate_modeled = KernelRidge(kernel='rbf', **dic_param)
    instantiate_modeled.fit(X_train, y_train)
    ypred = instantiate_modeled.predict(X_test)
    error[(dic_param['gamma'], dic_param['alpha'])] = np.linalg.norm(ypred-y_test)
#%%
from matplotlib.colors import LogNorm

zz = np.array(list(error.values())).reshape(all_gammas.shape[0], all_alphas.shape[0])
fs=15
fig, ax = plt.subplots(1, 1)
im=ax.pcolormesh(all_gammas, all_alphas, zz, norm=LogNorm(), cmap='RdBu_r')
ax.set_title('Validation error (in log)', fontsize=fs)
ax.set_xlabel('Scale $\gamma$', fontsize=fs)
ax.set_ylabel('Regularization $\lambda$', fontsize=fs)
ax.set_xlim([all_gammas.min(), all_gammas.max()])
ax.set_ylim([all_alphas.min(), all_alphas.max()])
ax.set_yscale('log')
ax.set_xscale('log')
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig('./parameters.pdf')
#%%
best_ = min(error, key=error.get)
print('Lowest error for (gamma, reg) = {0}, value = {1:.3f}'.format(best_, error[best_]) ) 
gamma_best, alpha_best = best_[0], best_[1]
kernel_ridge_best = KernelRidge(kernel="rbf", gamma=gamma_best, alpha=alpha_best)
polynomial_kernel = KernelRidge(kernel="poly", degree=2, alpha=1, coef0=1)
#%% Compare the best with respect to a simple regression avec polynome d'ordre 2
fs=10
fig, ax = plt.subplots(1, 1, figsize=(5,5))

ax.plot(temp.date, temp.temp, lw=2, c=cmap(0), 
        marker='+', markersize=5, linestyle='', alpha=0.6)

for i, (model, name) in enumerate(zip([kernel_ridge_best, polynomial_kernel], 
                                 ['Kernel ridge (RBF)', 'Kernel ridge (Poly)'])):
    model.fit(X_train, y_train)
    pred = model.predict(X)
    SSE = np.linalg.norm(model.predict(X_test)-y_test)**2
    ax.plot(temp.date, pred, lw=3, c=cmap(i+1), linestyle='-', 
        label='{0}, SSE test = {1:.2f}'.format(name, SSE), alpha=0.8)
    
ax.vlines(x=temp.date[train_idx], ymin=y.min()-0.5, ymax=y.max()+0.5, 
        linestyle='--', color='grey', lw=3, label='Training set')
ax.set_xlabel('Date', fontsize=fs)
ax.set_ylabel('temp (°C)', fontsize=fs)
ax.legend(loc='upper left', fontsize=fs)
ax.set_ylim([y.min()-0.5, y.max()+0.5])
fig.suptitle('Monthly avg temperature of {0}'.format(country_name), fontsize=fs+3)
fig.tight_layout()
#fig.savefig('./predictions.pdf')
# %%
