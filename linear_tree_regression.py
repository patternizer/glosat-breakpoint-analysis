#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: linear_tree_regression.py
#------------------------------------------------------------------------------
#
# Version 0.2
# 13 August, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
from itertools import product
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import *
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from lineartree import LinearTreeClassifier, LinearTreeRegressor
import basis_expansions
from basis_expansions import LinearSpline
#from basis_expansions import (Binner,
#                              GaussianKernel,
#                              Polynomial, 
#                              LinearSpline, 
#                              CubicSpline,
#                              NaturalCubicSpline)
#from dftransformers import ColumnSelector, FeatureUnion, Intercept, MapFeature
#from simulation import (run_simulation_expreiment, 
#                        plot_simulation_expreiment, 
#                        make_random_train_test,
#                        run_residual_simulation)

#-----------------------------------------------------------------------------
# ML Linear Tree Regression
#-----------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=4,
                       n_informative=2, n_targets=1,
                       random_state=0, shuffle=False)
regr = LinearTreeRegressor(base_estimator=LinearRegression())
regr.fit(X, y)

#-----------------------------------------------------------------------------
# ML Linear Tree Classification
#-----------------------------------------------------------------------------

from sklearn.linear_model import RidgeClassifier
from lineartree import LinearTreeClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = LinearTreeClassifier(base_estimator=RidgeClassifier())
clf.fit(X, y)

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

min_samples_leaf = 100
max_bins = 24
max_depth = 12 # in range [1,20]
max_r_over_rmax = 0.995 

fontsize = 16
plot_cusum = False
plot_correlation = True

stationcode = '037401' # HadCET
#stationcode = '103810'     # Berlin-Dahlem (breakpoint: 1908)
#stationcode = '685880'     # Durban/Louis Botha (breakpoint: 1939)

documented_change = np.nan
#documented_change = 1908
#documented_change = 1939 

filename = 'DATA/cusum_' + stationcode + '_obs.csv'

#-----------------------------------------------------------------------------
# METHODS
#-----------------------------------------------------------------------------

def factors(n):    
    '''
    https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
    '''
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def rc_subplots(n):    

    # DEDUCE: most rectangular array of subpolots (nrows=nr, ncolumns=nc)

    factor_vec = np.sort(np.array(list( factors(n) )))
    factor_diff = [ np.abs(factor_vec[i]-np.median(factor_vec)) for i in range(len(factor_vec)) ]
    nr = factor_vec[ factor_diff == np.min(factor_diff) ][0] 
    nc = int(np.max(factor_vec) / nr)

    return nc, nr

def linear_regression_ols(x,y):
    
    regr = linear_model.LinearRegression()
    # regr = TheilSenRegressor(random_state=42)
    # regr = RANSACRegressor(random_state=42)
    
    X = x.values.reshape(len(x),1)
    t = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
    regr.fit(X, y)
    ypred = regr.predict(t.reshape(-1, 1))
    slope = regr.coef_[0]
    intercept = regr.intercept_
        
    return t, ypred, slope, intercept

def adjusted_r_squared(x,y):
    
    X = x.reshape(len(x),1)
    model = sm.OLS(y, X).fit()
    R2adj = model.rsquared_adj

    return R2adj
    
#------------------------------------------------------------------------------
# LOAD: CUSUM timeseries from local expectation Kriging (LEK) analysis
#------------------------------------------------------------------------------
         
df = pd.read_csv(filename, index_col=0)
x = df.index.values
y = df.cu.values
mask = np.isfinite(y)
nx = len(df)
x = np.arange(nx) / nx
x = x[mask].reshape(-1, 1)
y = y[mask].reshape(-1, 1)

#------------------------------------------------------------------------------
# FIT: linear tree regression (LTR) model
#------------------------------------------------------------------------------
                    
# DEDUCE: subplot number or rows and columns

nc, nr = rc_subplots(max_depth)
    
# DEDUCE: optimal tree depth

r = []
r2adj = []
for depth in range(1,max_depth+1):       
       
    lt = LinearTreeRegressor(
        base_estimator = LinearRegression(),
        min_samples_leaf = min_samples_leaf,
        max_bins = max_bins,
        max_depth = depth
    ).fit(x, y)            
    y_fit = lt.predict(x)    
       
    X = y_fit.reshape(-1,1)
    Y = y
    mask_ols = np.isfinite(X) & np.isfinite(Y)
    corrcoef = scipy.stats.pearsonr(X[mask_ols], Y[mask_ols])[0]
    R2adj = adjusted_r_squared(X,Y)
    r.append(corrcoef)
    r2adj.append(R2adj)    

r_diff = [0.0] + list(np.diff(r))
max_depth_optimum = np.arange(1,max_depth+1)[np.array(r/np.max(r)) > max_r_over_rmax][1] - 1

# GRAPHVIZ: plot regression tree model (full) and to depth=3
            
# lt.plot_model(max_depth=3max_depth_optimum)

if plot_correlation == True:
        
    #------------------------------------------------------------------------------
    # PLOT: correlation r and R2adj as a function of knot number
    #------------------------------------------------------------------------------
        
    figstr = stationcode + '-' + 'cusum-curve-linear-tree-correlation.png'
                
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(np.arange(1,max_depth+1), r, marker='o', ls='-', lw=3, color='black', alpha=0.5, label=r'$\rho$')
    plt.plot(np.arange(1,max_depth+1), r2adj, marker='o', ls='-', lw=3, color='purple', alpha=0.5, label=r'$R^{2}_{adj}$')
    plt.plot(max_depth_optimum, r[max_depth_optimum-1], marker='o', markersize=20, ls='-', lw=3, color='black', alpha=0.5, label=r'BEST')
    plt.plot(max_depth_optimum, r2adj[max_depth_optimum-1], marker='o', markersize=20, ls='-', lw=3, color='purple', alpha=0.5, label=r'BEST')
    plt.axvline(x=max_depth_optimum, ls='--', lw=1, color='black')
    plt.xticks(np.arange(1,max_depth+1))
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Depth', fontsize=fontsize)
    plt.ylabel(r'Correlation', fontsize=fontsize)
    plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    plt.suptitle(stationcode, fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    

#------------------------------------------------------------------------------
# PLOT: breakpoints as a function of linear tree depth
#------------------------------------------------------------------------------
    
figstr = stationcode + '-' + 'cusum-curve-linear-tree-loop.png'

fig, ax = plt.subplots(figsize=(15,10))
    
for depth in range(1,max_depth+1):       

    # FIT: linear tree regressor
        
    lt = LinearTreeRegressor(
        base_estimator = LinearRegression(),
        min_samples_leaf = min_samples_leaf,
        max_bins = max_bins,
        max_depth = depth        
    ).fit(x, y)    

    # FIT: decision tree regressor

    # dr = DecisionTreeRegressor(   
    #    max_depth = depth
    # ).fit(x, y)
    # x_fit = dr.predict(x)

    y_fit = lt.predict(x)            
    y_fit_diff = [0.0] + list(np.diff(y_fit))
        
    breakpoints = df.index[mask][ np.abs(y_fit_diff) >= np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)) ][0:]
        
    plt.subplot(nr, nc, depth)
    plt.scatter(df.index[mask], y, s=3, c='blue', zorder=3)
    plt.scatter(df.index[mask], y_fit, s=3, c='darkorange', zorder=4)
    for i in range(len(breakpoints)):
        plt.axvline(x=breakpoints[i], ls='--', lw=2, color='teal', alpha=0.25, zorder=1)        
    plt.fill_between(df.index[mask], 
             np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)), 
             np.abs(np.nanmean(y_fit_diff)) - 6.0*np.abs(np.nanstd(y_fit_diff)), 
             ls='-', lw=1, color='teal', alpha=0.1, zorder=1)
    plt.plot(df.index[mask], [np.nan] + list(np.diff(y_fit)), ls='-', lw=2, color='teal', zorder=2)
    plt.axvline(x=documented_change, ls='--', lw=2, color='black')

    ylimits = plt.ylim()    

    if depth == max_depth_optimum:

        df_breakpoints = pd.DataFrame({'breakpoint':breakpoints})
        df_breakpoints.to_csv(stationcode + '-' + 'breakpoints.csv')

        plt.title('depth ' + str(depth) + r' (BEST): $\rho$=' + str(np.round(r[depth-1],3)), color='darkorange', fontsize=12)   
        
        for j in range(len(breakpoints)):        
            if (j%2 == 0) & (j<len(breakpoints)-1):
                plt.fill_betweenx(ylimits, breakpoints[j], breakpoints[j+1], facecolor='cyan', alpha=0.5, zorder=0)
            elif (j%2 != 0) & (j<len(breakpoints)-1):
                plt.fill_betweenx(ylimits, breakpoints[j], breakpoints[j+1], facecolor='yellow', alpha=0.5, zorder=0)        

    else:
        plt.title('depth ' + str(depth) + r': $\rho$=' + str(np.round(r[depth-1],3)))    
               
plt.suptitle(stationcode, fontsize=fontsize)
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')    
                  
#------------------------------------------------------------------------------
print('** END')