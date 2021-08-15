#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: piecewise_linear_spline_regression.py
#------------------------------------------------------------------------------
#
# Version 0.1
# 10 August, 2021
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
# SETTINGS
#-----------------------------------------------------------------------------

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

def noisy_sinusoid(timesteps, amp, freq, noise, random_state=None):
    '''
    EXAMPLE: SoS (2 noisy sinusoids)

    y1 = noisy_sinusoid(timesteps=4000, amp=10, freq=240, noise=3, random_state=33)
    y2 = noisy_sinusoid(timesteps=4000, amp=10, freq=240*7, noise=3, random_state=33)
    y = y1+y2
    X = np.arange(y.shape[0]).reshape(-1,1)

    '''
    
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.arange(timesteps)
    e = np.random.normal(0,noise, (timesteps,))
    y = amp * np.sin( X*(2*np.pi/freq) ) + e
    
    return y

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
    
def make_pl_regression(knot_strategy, n_knots):
    '''
    min: float
        Minimum of interval containing the knots.
    max: float
        Maximum of the interval containing the knots.
    n_knots: positive integer
        The number of knots to create.
    knot_strategy: str
        Strategy for determining the knots at fit time. Current options are:
          - 'even': Evenly position the knots within the range (min, max).
          - 'quantiles': Set the knots to even quantiles of the data distribution.
    ''' 
    
    return Pipeline([
        ('pl', LinearSpline(0, 1, knot_strategy=knot_strategy, n_knots=n_knots)),
        ('regression', LinearRegression(fit_intercept=True))
    ])

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
# FIT: quartile knot model
#------------------------------------------------------------------------------

regressions = {}
knot_strategy = 'quantiles'
min_knots = 2
max_knots = 14
nknots = (max_knots-min_knots)

for n_knots in range(min_knots, max_knots):
    regressions[n_knots] = make_pl_regression(knot_strategy, n_knots)
    regressions[n_knots].fit(x.reshape(-1, 1), y)

# DEDUCE: subplot number or rows and columns

nc, nr = rc_subplots(nknots)

# DEDUCE: optimal number of knots

r = []
r2adj = []
for i in range(nknots):
    n_knots = i + 2
    breakpoints = regressions[n_knots]['pl'].knots
    xbreakpoints = [ df.index[int(nx*breakpoints[k])] for k in range(len(breakpoints))]
    X = regressions[n_knots].predict(x.reshape(-1, 1))
    Y = y
    mask_ols = np.isfinite(X) & np.isfinite(Y)
    corrcoef = scipy.stats.pearsonr(X[mask_ols], Y[mask_ols])[0]
    R2adj = adjusted_r_squared(X,Y)
    r.append(corrcoef)
    r2adj.append(R2adj)    

r_diff = np.diff(r)
mask_r = (r_diff<0) & (np.abs(r_diff) < 0.01)
mask_r_full = [False] + list(mask_r)
nknots_optimum = np.arange(nknots)[mask_r_full][0] + 1

#------------------------------------------------------------------------------
# PLOT: piecewise linear spline regressions for each number of knots case
#------------------------------------------------------------------------------

figstr = stationcode + '-' + 'cusum-curve-pwlr-loop.png'

fig, ax = plt.subplots(nr, nc, figsize=(15,10))
for i, ax in enumerate(ax.flatten()):
    
    n_knots = i + 2
    breakpoints = regressions[n_knots]['pl'].knots
    xbreakpoints = [ df.index[int(nx*breakpoints[k])] for k in range(len(breakpoints))]
    X = regressions[n_knots].predict(x.reshape(-1, 1))
    Y = y
    mask_ols = np.isfinite(X) & np.isfinite(Y)
    corrcoef = scipy.stats.pearsonr(X[mask_ols], Y[mask_ols])[0]
#   OLS_X, OLS_Y, OLS_slope, OLS_intercept = linear_regression_ols(X[mask_ols], Y[mask_ols])
    R2adj = adjusted_r_squared(X,Y)
    
    ax.plot(df.index[mask], regressions[n_knots].predict(x.reshape(-1, 1)), ls='-', lw=3, color='darkorange', zorder=2, label='PWLR')
    ax.plot(df.index[mask], y, marker='o', ls='-', lw=3, alpha=0.5, color='blue', zorder=1, label='CUSUM')
    ax.fill_between(df.index[mask], -2*df.ce[mask], 2*df.ce[mask], color='black', alpha=0.2, zorder=0, label='Erf')      
    xlimits = ax.axes.get_xlim()
    ylimits = ax.axes.get_ylim()    
    if n_knots == 2:
        ax.axvline(x=xbreakpoints[0], ls='--', lw=1, color='black', label='Changepoint')
        for j in range(len(breakpoints)):        
            ax.axvline(x=xbreakpoints[j], ls='--', lw=1, color='black')
        ax.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=1, fontsize=10)  
        ax.set_title(str(n_knots) + r' knots: $\rho$=' + str(np.round(corrcoef,3)), fontsize=12)    
    elif n_knots == nknots_optimum:
        for j in range(len(breakpoints)):        
            ax.axvline(x=xbreakpoints[j], ls='--', lw=1, color='black')
#            ax.axvline(x=xbreakpoints[j], ls='--', lw=2, color='darkorange')
            if (j%2 == 0) & (j<len(breakpoints)-1):
                ax.fill_betweenx(ylimits, xbreakpoints[j], xbreakpoints[j+1], facecolor='cyan', alpha=0.5)
            elif (j%2 != 0) & (j<len(breakpoints)-1):
                ax.fill_betweenx(ylimits, xbreakpoints[j], xbreakpoints[j+1], facecolor='yellow', alpha=0.5)
        ax.set_title(str(n_knots) + r' knots (BEST): $\rho$=' + str(np.round(corrcoef,3)), color='darkorange', fontsize=12)    
    else:        
        for j in range(len(breakpoints)):        
            ax.axvline(x=xbreakpoints[j], ls='--', lw=1, color='black')
            ax.set_title(str(n_knots) + r' knots: $\rho$=' + str(np.round(corrcoef,3)), fontsize=12)    
                
plt.suptitle(stationcode)
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')    


if plot_cusum == True:
    
    #------------------------------------------------------------------------------
    # PLOT: station LEK CUSUM 
    #------------------------------------------------------------------------------
    
    figstr = stationcode + '-' + 'cusum-curve.png'
            
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(df.index, df.cu, marker='o', ls='-', lw=3, color='blue', alpha=0.5, label='CUSUM')
    plt.fill_between(df.index, -2*df.ce, 2*df.ce, color='black', alpha=0.2, label=r'Erf')        
    plt.tick_params(labelsize=fontsize)    
    plt.xlabel('Time', fontsize=fontsize)
    plt.ylabel(r'CUSUM', fontsize=fontsize)
    plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    plt.suptitle(stationcode)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    

if plot_correlation == True:
    
    #------------------------------------------------------------------------------
    # PLOT: correlation r and R2adj as a function of knot number
    #------------------------------------------------------------------------------
    
    figstr = stationcode + '-' + 'cusum-curve-pwlr-correlation.png'
            
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(np.arange(2,2+nknots), r, marker='o', ls='-', lw=3, color='black', alpha=0.5, label=r'$\rho$')
    plt.plot(np.arange(2,2+nknots), r2adj, marker='o', ls='-', lw=3, color='purple', alpha=0.5, label=r'$R^{2}_{adj}$')
    plt.plot(nknots_optimum, r[nknots_optimum-2], marker='o', markersize=20, ls='-', lw=3, color='black', alpha=0.5, label=r'BEST')
    plt.plot(nknots_optimum, r2adj[nknots_optimum+1], marker='o', markersize=20, ls='-', lw=3, color='purple', alpha=0.5, label=r'BEST')
    plt.axvline(x=nknots_optimum, ls='--', lw=1, color='black')
    plt.xticks(np.arange(2,2+nknots))
    plt.tick_params(labelsize=fontsize)  
    plt.ylim(0.88,1.0) 
    plt.xlabel('n (knots)', fontsize=fontsize)
    plt.ylabel(r'Correlation', fontsize=fontsize)
    plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    plt.suptitle(stationcode, fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    

#------------------------------------------------------------------------------
print('** END')
