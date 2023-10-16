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
data_path = './data/temperature'
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
# Que faudrait t'il faire proprement ? (un set de validation)
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
# Demander de faire retourner la grille et regarder les performances in fine
# Ca pose la question importante du choix des hyperparam
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
#%% Discuter (changement de la mérique par exemple: relative error, Gaussian process, noyau vraiment adapté ? 
# eg pas de prise en compte de la saisonnalité, qu'un moins de juin d'une année à l'autre est plus proche mais ici nan )
# unable to capture seasonality
#Question bonus : tout refaire sans sklearn
# peut on prédire la trajectoire des 2°C ?

#%% Avec un truc du style saisonnality

from sklearn.linear_model import Ridge

class SeasonalityKernel():
    # Implement function of the form f(t) = sum_k a_k(cos(2*pi*w0*k*t))+ b*t + c*t^2+ d
    def __init__(self, K=3, w0=12, alpha=1.0):
        self.K=K
        self.w0=w0
        self.alpha=alpha
        self.model = Ridge(alpha=self.alpha, fit_intercept=False)

    def Phi(self, X):
        Y = np.zeros((X.shape[0], 1+2+self.K))
        Y[:,0] = np.ones(X.shape[0])
        Y[:,1] = X.ravel()
        Y[:,2] = X.ravel()**2
        for k in range(1, self.K+1):
                Y[:,2+k] = np.cos((2*k*np.pi*X.ravel()*self.w0))
        return Y
    
    def fit(self, X, y):
        Xtransfo = self.Phi(X)
        self.model.fit(Xtransfo, y)

    def predict(self,X):
        Xtransfo = self.Phi(X)
        return self.model.predict(Xtransfo)
#%%
all_params = {'w0':np.logspace(-3, 3, 30),
              'K': [3, 5, 10, 20, 50, 100], 
              'alpha': np.logspace(-5, 3, 10)}

param_grid = ParameterGrid(all_params)
print('Parameters to test = {}'.format(len(list(param_grid))))
#%%
error = {}
for dic_param in param_grid:
    instantiate_modeled = SeasonalityKernel(**dic_param)
    instantiate_modeled.fit(X_train, y_train)
    ypred = instantiate_modeled.predict(X_test)
    error[(dic_param['w0'], dic_param['K'], dic_param['alpha'])] = np.linalg.norm(ypred-y_test)
#%%
best_ = min(error, key=error.get)
print('Lowest error for (w0, K, alpha) = {0}, value = {1:.3f}'.format(best_, error[best_]) ) 
w0_best, K_best, alpha_best = best_[0], best_[1], best_[2]
best_seasonality_model = SeasonalityKernel(w0=w0_best, K=K_best, alpha=alpha_best)
#%%
fs=10
fig, ax = plt.subplots(1, 1, figsize=(5,5))

ax.plot(temp.date, temp.temp, lw=2, c=cmap(0), 
        marker='+', markersize=5, linestyle='', alpha=0.6)

for i, (model, name) in enumerate(zip([kernel_ridge_best, polynomial_kernel, best_seasonality_model], 
                                 ['RBF', 'Polynomial (degree 2)', 'Seasonality model'])):
    model.fit(X_train, y_train)
    pred = model.predict(X)
    SSE = np.linalg.norm(model.predict(X_test)-y_test)**2
    ax.plot(temp.date, pred, lw=3, c=cmap(i+1), linestyle='-', 
        label='{0}, SSE test = {1:.2f}'.format(name, SSE), alpha=0.8)
    
ax.vlines(x=temp.date[train_idx], ymin=y.min()-0.5, ymax=y.max()+0.5, 
        linestyle='--', color='grey', lw=3, label='Training set', alpha=0.5)
ax.set_xlabel('Date', fontsize=fs)
ax.set_ylabel('temp (°C)', fontsize=fs)
ax.legend(loc='upper left', fontsize=fs)
ax.set_ylim([y.min()-0.5, y.max()+0.5])
fig.suptitle('Monthly avg temperature of {0}'.format(country_name), fontsize=fs+3)
fig.tight_layout()
# %%
# change country ? interpret ?
#%% Projection
import matplotlib.ticker as plticker

X_all = range(1350)
date_all = 1940+(1/12)*np.array(X_all)

fs=10
fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.axvspan(temp.date[0], temp.date[train_idx], facecolor='blue', label='Training set', alpha=0.1)
ax.axvspan(temp.date.values[-1], date_all[-1], facecolor='red', label='Projection', alpha=0.1)

ax.plot(temp.date, temp.temp, lw=2, c=cmap(0), 
        marker='o', markersize=4, linestyle='', alpha=0.9)

for i, (model, name) in enumerate(zip([kernel_ridge_best, polynomial_kernel, best_seasonality_model], 
                                 ['RBF', 'Polynomial (degree 2)', 'Seasonality model'])):
    model.fit(X_train, y_train)
    pred = model.predict(np.array(X_all).reshape(-1,1))
    SSE = np.linalg.norm(model.predict(X_test)-y_test)**2
    ax.plot(date_all, pred, lw=3, c=cmap(i+1), linestyle='-', 
        label='{0}, SSE test = {1:.2f}'.format(name, SSE), alpha=0.8)

ax.set_xlabel('Date', fontsize=fs)
ax.set_ylabel('temp (°C)', fontsize=fs)
ax.legend(loc='upper left', fontsize=fs)
ax.set_ylim([y.min()-0.5, y.max()+0.5])

loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
fig.suptitle('Monthly avg temperature of {0}'.format(country_name), fontsize=fs+3)
fig.tight_layout()
#%% With proper CV 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import median_absolute_error, mean_squared_error

def frozendict(d: dict):
    keys = sorted(d.keys())
    return tuple((k, d[k]) for k in keys)

def do_cv(param_grid, X_in, y_in, model, n_splits=3):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = {}
        for j, dic_param in enumerate(param_grid):

                instantiate_modeled = model(**dic_param)
                cv_errors = []
                for i, (train_index, valid_index) in enumerate(tscv.split(X_in)):
                        X_train, y_train = X_in[train_index], y_in[train_index]
                        X_valid, y_valid = X_in[valid_index], y_in[valid_index]

                        instantiate_modeled.fit(X_train, y_train)
                        ypred = instantiate_modeled.predict(X_valid)
                        err_ = mean_squared_error(y_valid, ypred)
                        cv_errors.append(err_)

                results[frozendict(dic_param)] = np.mean(cv_errors)
                print('Parameter {} done ..'.format(dic_param))
                print('--- {0:.0%} finished ---'.format(j/len(list(param_grid))))
        return results

ratio = 0.8
split_train_test = int(np.floor(ratio*X.shape[0]))
X_in, X_test = X[0:split_train_test,], X[split_train_test+1:,]
y_in, y_test = y[0:split_train_test], y[split_train_test+1:]


all_params = {'w0':np.logspace(-3, 3, 30),
              'K': [3, 5, 10, 20, 50, 100], 
              'alpha': np.logspace(-5, 3, 10)}

param_grid = ParameterGrid(all_params)
print('Parameters to test = {}'.format(len(list(param_grid))))

#%%
results = do_cv(param_grid, X_in, y_in, SeasonalityKernel)

#%%
best_ = min(results, key=results.get)
print('Lowest error for parameters = {0}, value = {1:.3f}'.format(best_, results[best_]))
best_params = {elt[0] : elt[1] for elt in best_}
best_seasonality_model = SeasonalityKernel(**best_params)

#%% Do the same for Kernel ridge
all_gammas = np.logspace(-5, 5, 20)
all_alphas = np.logspace(-5, 5, 20)
all_params = {'gamma': all_gammas, 
              'alpha': all_alphas}
param_grid = ParameterGrid(all_params)
results = do_cv(param_grid, X_in, y_in, KernelRidge)
#%%
best_ = min(results, key=results.get)
print('Lowest error for parameters = {0}, value = {1:.3f}'.format(best_, results[best_]))
best_params = {elt[0] : elt[1] for elt in best_}
best_krr_model = KernelRidge(**best_params)
#%% And the baseline
polynomial_kernel = KernelRidge(kernel="poly", degree=2, alpha=1, coef0=1)

#%%

X_all = range(1350)
date_all = 1940+(1/12)*np.array(X_all)

fs=10
fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.axvspan(temp.date[0], temp.date[split_train_test], facecolor='blue', label='Train/Valid set', alpha=0.1)
ax.axvspan(temp.date.values[-1], date_all[-1], facecolor='red', label='Projection', alpha=0.1)

ax.plot(temp.date, temp.temp, lw=2, c=cmap(0), 
        marker='o', markersize=4, linestyle='', alpha=0.9)

for i, (model, name) in enumerate(zip([best_krr_model, polynomial_kernel, best_seasonality_model], 
                                 ['KRR (RBF)','KRR (poly degree 2)','Seasonality model'])):
    model.fit(X_in, y_in)
    pred = model.predict(np.array(X_all).reshape(-1,1))
    SSE = np.linalg.norm(model.predict(X_test)-y_test)**2
    ax.plot(date_all, pred, lw=3, c=cmap(i+1), linestyle='-', 
        label='{0}, SSE test = {1:.2f}'.format(name, SSE), alpha=0.8)

ax.set_xlabel('Date', fontsize=fs)
ax.set_ylabel('temp (°C)', fontsize=fs)
ax.legend(loc='upper left', fontsize=fs)
ax.set_ylim([y.min()-0.5, y.max()+0.5])
ax.grid(alpha=0.6)
loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
fig.suptitle('Monthly avg temperature of {0}'.format(country_name), fontsize=fs+3)
fig.tight_layout()
fig.savefig('possible_rest.pdf')







#%% 

# %% With a little robustess to outliers
# from sklearn.linear_model import HuberRegressor

# class RobustSeasonalityKernel():
#     # Implement function of the form f(t) = sum_k a_k(cos(2*pi*w0*k*t))+ b*t + c*t^2+ d
#     def __init__(self, K=3, w0=12, alpha=1.0, epsilon=1.35, max_iter=100):
#         self.K=K
#         self.w0=w0
#         self.alpha=alpha
#         self.epsilon=epsilon
#         self.max_iter=max_iter
#         self.model = HuberRegressor(alpha=self.alpha, fit_intercept=False, epsilon=self.epsilon, max_iter=self.max_iter)

#     def Phi(self, X):
#         Y = np.zeros((X.shape[0], 1+2+self.K))
#         Y[:,0] = np.ones(X.shape[0])
#         Y[:,1] = X.ravel()
#         Y[:,2] = X.ravel()**2
#         for k in range(1, self.K+1):
#                 Y[:,2+k] = np.cos((2*k*np.pi*X.ravel()*self.w0))
#         return Y
    
#     def fit(self, X, y):
#         Xtransfo = self.Phi(X)
#         self.model.fit(Xtransfo, y)

#     def predict(self,X):
#         Xtransfo = self.Phi(X)
#         return self.model.predict(Xtransfo)
    
# #%%
# all_params = {'w0':[0.5],
#               'K': [10], 
#               'alpha': np.logspace(-5, 3, 20),
#               'epsilon':np.linspace(1, 3, 5)}

# param_grid = ParameterGrid(all_params)
# print('Parameters to test = {}'.format(len(list(param_grid))))
# #%%
# error = {}
# for dic_param in param_grid:
#     instantiate_modeled = RobustSeasonalityKernel(**dic_param)
#     instantiate_modeled.fit(X_train, y_train)
#     ypred = instantiate_modeled.predict(X_test)
#     error[(dic_param['w0'], dic_param['K'], dic_param['alpha'], dic_param['epsilon'])] = np.linalg.norm(ypred-y_test)
# #%%
# best_ = min(error, key=error.get)
# print('Lowest error for (w0, K, alpha, epsilon) = {0}, value = {1:.3f}'.format(best_, error[best_]) ) 
# w0_best, K_best, alpha_best, epsilon_best = best_[0], best_[1], best_[2], best_[3]
# best_robust_seasonality_model = RobustSeasonalityKernel(w0=w0_best, K=K_best, alpha=alpha_best, epsilon=epsilon_best)
# # %%
# X_all = range(1350)
# date_all = 1940+(1/12)*np.array(X_all)

# fs=10
# fig, ax = plt.subplots(1, 1, figsize=(8,5))
# ax.axvspan(temp.date[0], temp.date[train_idx], facecolor='blue', label='Training set', alpha=0.1)
# ax.axvspan(temp.date.values[-1], date_all[-1], facecolor='red', label='Projection', alpha=0.1)

# ax.plot(temp.date, temp.temp, lw=2, c=cmap(0), 
#         marker='o', markersize=4, linestyle='', alpha=0.9)

# for i, (model, name) in enumerate(zip([kernel_ridge_best, polynomial_kernel, best_seasonality_model, best_robust_seasonality_model], 
#                                  ['RBF', 'Polynomial (degree 2)', 'Seasonality model', 'Hubber'])):
#     model.fit(X_train, y_train)
#     pred = model.predict(np.array(X_all).reshape(-1,1))
#     SSE = np.linalg.norm(model.predict(X_test)-y_test)**2
#     ax.plot(date_all, pred, lw=3, c=cmap(i+1), linestyle='-', 
#         label='{0}, SSE test = {1:.2f}'.format(name, SSE), alpha=0.8)

# ax.set_xlabel('Date', fontsize=fs)
# ax.set_ylabel('temp (°C)', fontsize=fs)
# ax.legend(loc='upper left', fontsize=fs)
# ax.set_ylim([y.min()-0.5, y.max()+0.5])

# loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
# ax.xaxis.set_major_locator(loc)
# fig.suptitle('Monthly avg temperature of {0}'.format(country_name), fontsize=fs+3)
# fig.tight_layout()
# %%
