#%% Proper temperature test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cmap = plt.cm.get_cmap('tab10')
from os import listdir
from os.path import isfile, join
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import median_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.kernel_ridge import KernelRidge
import matplotlib.ticker as plticker

data_path = './data/temperature'
all_countries = []
for f in listdir(data_path):
    if isfile(join(data_path, f)) and f.endswith('.csv'):
        all_countries.append(f.replace('.csv',''))
print(all_countries)
#%%
fs=10
fig, ax = plt.subplots(6, 5, figsize=(15,15))
ax[0,0].set_ylabel('temp (°C)', fontsize=fs)
ax[0,0].set_xlabel('Date', fontsize=fs)
r, c = 0, 0
for country_name in all_countries:
    temp = pd.read_csv(data_path+'/'+country_name+'.csv', names=['date','temp'])
    ax[r,c].plot(temp.date, temp.temp, lw=1, c=cmap(0), marker='o', markersize=1, linestyle='-')
    #ax[r,c].set_ylabel('temp (°C)', fontsize=fs)
    if country_name == 'federated-states-of-micronesia':
         ax[r,c].set_title('{0}'.format('micronesia'), fontsize=fs+3)
    else:
         ax[r,c].set_title('{0}'.format(country_name), fontsize=fs+3)
    ax[r,c].grid(alpha=0.5)        
    loc = plticker.MultipleLocator(base=30) # this locator puts ticks at regular intervals
    ax[r,c].xaxis.set_major_locator(loc)
    c+=1
    if c == 5:
        r += 1
        c = 0 

fig.tight_layout()
#%%
country_name = 'peru'
assert country_name in all_countries
temp = pd.read_csv(data_path+'/'+country_name+'.csv', names=['date','temp'])

# %%

class SeasonalityModel():
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

#%% Data
X = np.array(range(temp.date.shape[0])).reshape(-1,1)
y = np.array(temp.temp)
ratio = 0.85
split_train_test = int(np.floor(ratio*X.shape[0]))
X_in, X_test = X[0:split_train_test,], X[split_train_test+1:,]
y_in, y_test = y[0:split_train_test], y[split_train_test+1:]

#%% Parameters grid
all_params = {'w0': np.logspace(-2, 3, 50),
              'K': [3, 5, 10, 20, 50, 70, 100], 
              'alpha': np.logspace(-5, 3, 50)}

param_grid = ParameterGrid(all_params)
print('Parameters to test = {}'.format(len(list(param_grid))))
# %%
results = do_cv(param_grid, X_in, y_in, SeasonalityModel, n_splits=5)
#%%
best_ = min(results, key=results.get)
print('Lowest error for parameters = {0}, value = {1:.3f}'.format(best_, results[best_]))
best_params = {elt[0] : elt[1] for elt in best_}
best_seasonality_model = SeasonalityModel(**best_params)
# Baseline
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

for i, (model, name) in enumerate(zip([best_seasonality_model, polynomial_kernel], 
                                 ['Seasonality model','KRR (poly degree 2)'])):
    model.fit(X_in, y_in)
    pred = model.predict(np.array(X_all).reshape(-1,1))
    SSE = np.linalg.norm(model.predict(X_test)-y_test)**2
    ax.plot(date_all, pred, lw=3, c=cmap(i+1), linestyle='-', 
        label='{0}, SSE test = {1:.2f}'.format(name, SSE), alpha=0.9)

ax.set_xlabel('Date', fontsize=fs)
ax.set_ylabel('temp (°C)', fontsize=fs)
ax.legend(loc='upper left', fontsize=fs)
ax.set_ylim([y.min()-1.5, y.max()+1.5])

loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.grid(alpha=0.5)
fig.suptitle('Monthly avg temperature of {0}'.format(country_name), fontsize=fs+3)
fig.tight_layout()
# %%
