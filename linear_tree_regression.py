#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: linear_tree_regression.py
#------------------------------------------------------------------------------
#
# Version 0.3
# 3 November, 2021
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
#import basis_expansions
#from basis_expansions import LinearSpline
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

max_depth = 9 # in range [1,20]
#max_r_over_rmax = 0.999
max_r_over_rmax = 0.99
#max_r_over_rmax = 0.95
#max_r_over_rmax = 0.9
min_separation = 120
min_slope_change = 6 # CUSUM / decade

fontsize = 16
plot_correlation = False
plot_best = True

stationcode = '037401'     # HadCET
#stationcode = '103810'     # Berlin-Dahlem (breakpoint: 1908)
#stationcode = '685880'     # Durban/Louis Botha (breakpoint: 1939)
#stationcode = '024581'     # Uppsala
#stationcode = '725092'     # Boston City WSO
#stationcode = '062600'     # St Petersberg
#stationcode = '260630'     # De Bilt
#stationcode = '688177'     # Cape Town
#stationcode = '619930'     # Pamplemousses

#documented_change = np.nan
#documented_change = 1908
#documented_change = 1939 

use_lek_cusum = False

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
    
    X = x.reshape(len(x),1)
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

def calculate_piecewise_regression( x, y, depth ):
    
    # FIT: linear tree regressor
        
#    min_samples_leaf = 24
#    max_bins = 60

    min_samples_leaf = 120
#    max_bins = int(len(x)/60) # range=[10,120]
    max_bins = 40 # range=[10,120]
        
    lt = LinearTreeRegressor(
        base_estimator = LinearRegression(),
        min_samples_leaf = min_samples_leaf,
        max_bins = max_bins,
        max_depth = depth        
    ).fit(x, y)    
    y_fit = lt.predict(x)           
        
    # FIT: decision tree regressor

    dr = DecisionTreeRegressor(   
       max_depth = depth
    ).fit(x, y)
    x_fit = dr.predict(x)

#    x_fit = x_fit.reshape(-1,1)
#    y_fit = y_fit.reshape(-1,1)

    # COMPUTE: goodness of fit

    mask_ols = np.isfinite(y) & np.isfinite(y_fit.reshape(-1,1))
    corrcoef = scipy.stats.pearsonr(y[mask_ols], y_fit.reshape(-1,1)[mask_ols])[0]
    R2adj = adjusted_r_squared(y, y_fit.reshape(-1,1))
    
    return x_fit, y_fit, corrcoef, R2adj
 
def calculate_breakpoints( y, y_fit, min_separation, min_slope_change ):
       
    # BREAKPOINT: detection ( using slopes )

    y_fit_diff1 = np.array([np.nan] + list(np.diff(y_fit)))
    y_fit_diff2 = np.array([np.nan, np.nan] + list(np.diff(y_fit, 2)))
    y_fit_diff2[ y_fit_diff2 < 1e-6 ] = np.nan
    idx = np.arange( len(y_fit_diff2) )[ np.abs(y_fit_diff2) > 0] - 1     
    slopes_all = np.zeros(len(y))
    slopes_all[:] = y_fit_diff1[:]
    slopes_all[ idx ] = np.nan
    slopes_all[ idx + 1] = np.nan
    slopes_all = slopes_all * min_separation # slope = Q-sum / decade if min_separation=120                  
    slopes = np.zeros(len(y))
    for i in range(len(y)):    
        if i==0:        
            slopes[0] = 0.0        
        else:        
            if np.isnan(slopes_all[i]):        
                slopes[i] = slopes[i-3]
            else:            
                slopes[i] = slopes_all[i]                
    slopes_diff = np.array( [0.0] + list(np.diff(slopes)) )
    breakpoints_all = np.arange(len(y))[ np.abs(slopes_diff) > min_slope_change ] - 1     
    breakpoints_diff = np.array( [breakpoints_all[0]] + list( np.diff(breakpoints_all) ) )
    breakpoints = breakpoints_all[ breakpoints_diff > min_separation ] # decade minimum breakpoint separation
#   breakpoints_all = np.arange(len(t))[ np.abs(y_fit_diff1) >= np.abs(np.nanmean(y_fit_diff1)) + 6.0*np.abs(np.nanstd(y_fit_diff1)) ][0:] 
#   breakpoints = t[ breakpoints_all > min_separation ]
                
    return y_fit_diff2, slopes, breakpoints    

#------------------------------------------------------------------------------
# LOAD: CUSUM timeseries from local expectation Kriging (LEK) analysis
#------------------------------------------------------------------------------

if use_lek_cusum == True:
    
    file_cusum = 'DATA/cusum_' + stationcode + '_obs.csv'
    df = pd.read_csv( file_cusum, index_col=0 )
    t_monthly = pd.date_range(start=str(int(np.floor(df.index[0]))), periods=len(df), freq='M')     

    t = t_monthly
    c = df.cu.values # 12-MA smoothed by LEK algorithm    
    
    mask = np.isfinite(c)
    c = c[mask]
    t = t[mask]
    
#    documented_change_datetime = documented_change
    
else:
    
    df_temp = pd.read_pickle('DATA/df_temp_expect_reduced.pkl', compression='bz2')
    df_compressed = df_temp[ df_temp['stationcode'] == stationcode ].sort_values(by='year').reset_index(drop=True).dropna()
    t_yearly = np.arange( df_compressed.iloc[0].year, df_compressed.iloc[-1].year + 1)
    df_yearly = pd.DataFrame({'year':t_yearly})
    df = df_yearly.merge(df_compressed, how='left', on='year')
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,43:55]).reset_index()        
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)                   
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')     
#    documented_change_datetime = pd.to_datetime('01-01-'+str(documented_change),format='%d-%m-%Y')
    
    # COMPUTE: 12-m MA
    
    a = pd.Series(ts_monthly).rolling(12, center=True).mean().values
    e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
    s = pd.Series(sd_monthly).rolling(12, center=True).mean().values
        
    diff_monthly = ts_monthly - ex_monthly
    diff_yearly = a - e
    
    # CALCULATE: CUSUM
        	
    t = t_monthly
    c = np.nancumsum( diff_yearly )
    
x = ( np.arange(len(c)) / len(c) ).reshape(-1, 1)
y = c.reshape(-1, 1)

#------------------------------------------------------------------------------
# FIT: linear tree regression (LTR) model
#------------------------------------------------------------------------------
                    
# DEDUCE: subplot number or rows and columns

nc, nr = rc_subplots( max_depth )
    
# DEDUCE: optimal tree depth

r = []
r2adj = []
for depth in range(1,max_depth+1):       
       
    x_fit, y_fit, corrcoef, R2adj = calculate_piecewise_regression( x, y, depth )    
    
    r.append(corrcoef)
    r2adj.append(R2adj)    
               
r_diff = np.array( [np.nan] + list(np.diff(r)) )
max_depth_optimum = np.arange(1,max_depth+1)[ r_diff < 0.001 ][0] - 1
#max_depth_optimum = np.arange(1,max_depth+1)[np.array(r/np.max(r)) >= max_r_over_rmax][1] - 1 

# GRAPHVIZ: plot regression tree model (full) and to depth=3
            
# lt.plot_model(max_depth=max_depth_optimum)

if plot_correlation == True: 
        
    #------------------------------------------------------------------------------
    # PLOT: correlation r and R2adj as a function of knot number
    #------------------------------------------------------------------------------
        
    figstr = stationcode + '-' + 'cusum-curve-linear-tree-correlation.png'
                
    fig, ax = plt.subplots(figsize=(15,10))
    plt.axvline( x=max_depth_optimum, ls='--', lw=1, color='white')
    plt.plot(np.arange( 1, max_depth+1), r, marker='o', ms=6, ls='-', lw=1, color='teal', alpha=1, label=r'$\rho$')
    plt.plot(np.arange( 1, max_depth+1), r2adj, marker='o', ms=6, ls='-', lw=1, color='purple', alpha=1, label=r'$R^{2}_{adj}$')
    plt.plot( max_depth_optimum, r[max_depth_optimum-1], marker='o', ms=12, ls='-', lw=3, color='teal', alpha=0.5, label=r'$\rho$( BEST )')
    plt.plot( max_depth_optimum, r2adj[max_depth_optimum-1], marker='o', markersize=12, ls='-', lw=3, color='purple', alpha=0.5, label=r'$R^{2}_{adj}$( BEST )')
    ax.xaxis.grid(b=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(b=None, which='major', color='none', linestyle='-')
    plt.grid(b=None)
    plt.xticks(np.arange(1,max_depth+1))
    ax.xaxis.grid(b=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(b=None, which='major', color='none', linestyle='-')
    plt.grid(b=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Depth', fontsize=fontsize)
    plt.ylabel(r'Correlation', fontsize=fontsize)
    plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    plt.title( stationcode, color='white', fontsize=fontsize)           
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    

#------------------------------------------------------------------------------
# PLOT: breakpoints as a function of linear tree depth
#------------------------------------------------------------------------------

fontsize = 12
    
figstr = stationcode + '-' + 'cusum-curve-linear-tree-loop.png'
fig, ax = plt.subplots(figsize=(15,10))

for depth in range(1, max_depth+1):       

    x_fit, y_fit, corrcoef, R2adj = calculate_piecewise_regression( x, y, depth )    
    y_fit_diff2, slopes, breakpoints = calculate_breakpoints( y, y_fit, min_separation, min_slope_change )

    plt.subplot(nr, nc, depth)
    if depth < max_depth:        
#        plt.axvline(x=documented_change_datetime, ls='--', lw=1, color='black')     
        plt.scatter(t, y.ravel(), marker='.', s=3, fc='navy', ls='-', lw=1, color='blue', alpha=0.5)
        plt.plot(t, y_fit.ravel(), color='red', ls='-', lw=1)        
        plt.fill_between( t, slopes.ravel(), 0, color='blue', alpha=0.1)
    else:
#        plt.axvline(x=documented_change_datetime, ls='--', lw=1, color='black', label='Documented change: ' + str(documented_change) )     
        plt.scatter(t, y, marker='.', s=3, fc='navy', ls='-', lw=1, color='blue', alpha=0.5, label='CUSUM (O-E)')
        plt.plot(t, y_fit, color='red', ls='-', lw=1, label='LTR fit')        
        plt.fill_between( t, slopes, 0, color='blue', alpha=0.1, label='slope: CUSUM/decade' )
    ylimits = plt.ylim()       
                
    if depth == max_depth_optimum:
        plt.title( stationcode + ': depth=' + str(depth) + r' ( BEST ): $\rho$=' + str( f'{r[depth-1]:03f}' ), color='black', fontsize=fontsize)                   
        df_breakpoints = pd.DataFrame({'breakpoint':t[breakpoints]})
        df_breakpoints.to_csv(stationcode + '-' + 'breakpoints.csv')
        for i in range(len(t[(y_fit_diff2>0).ravel()])):
            if i==0:
                plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color='black', alpha=0.2, label='LTR segment') 
            else:
                plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color='black', alpha=0.2) 
        for i in range(len(breakpoints)):
            if i==0:
                plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color='black', label='Breakpoint')
            else:
                plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color='black')     
    else:
        plt.title( stationcode + ': depth=' + str(depth) + r' : $\rho$=' + str( f'{r[depth-1]:03f}' ), color='black', fontsize=fontsize)           
        for i in range(len(t[(y_fit_diff2>0).ravel()])):
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color='black', alpha=0.2) 
        for i in range(len(breakpoints)):
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color='black')     
               
    ax.xaxis.grid(b=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(b=None, which='major', color='none', linestyle='-')
    ax.grid(b=None)    
    plt.tick_params(labelsize=fontsize)    
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'CUSUM, $^{\circ}$C', fontsize=fontsize)    
    fig.tight_layout()

fig.legend(loc='lower right', ncol=2, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)       
fig.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=None, hspace=None)  
plt.savefig(figstr, dpi=300)
plt.close('all')    
                  
#------------------------------------------------------------------------------
print('** END')

if plot_best == True:
        
    x_fit, y_fit, corrcoef, R2adj = calculate_piecewise_regression( x, y, max_depth_optimum )    
    y_fit_diff2, slopes, breakpoints = calculate_breakpoints( y, y_fit, min_separation, min_slope_change )
    
    figstr = stationcode + '-' + 'cusum-curve-breakpoint-selection.png'                
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot( t, y, color='blue', ls='-', lw=3, label='CUSUM (O-E)')
    plt.plot( t, y_fit, color='red', ls='-', lw=2, label='LTR fit')
    plt.fill_between( t, slopes, 0, color='blue', alpha=0.1, label='slope: CUSUM/decade' )    
    for i in range(len(t[(y_fit_diff2>0).ravel()])):
        if i==0:
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color='black', alpha=0.2, label='LTR segment') 
        else:
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color='black', alpha=0.2) 
    for i in range(len(breakpoints)):
        if i==0:
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color='black', label='Breakpoint')
        else:
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color='black')    
    ax.xaxis.grid(b=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(b=None, which='major', color='none', linestyle='-')
    plt.grid(b=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'CUSUM, $^{\circ}$C', fontsize=fontsize)
    plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    plt.title( stationcode, color='black', fontsize=fontsize)           
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    

    if use_lek_cusum == False:
    
        figstr = stationcode + '-' + 'cusum-curve-breakpoint-timeseries.png'                
        fig, ax = plt.subplots(figsize=(15,10))
        plt.scatter(t, diff_yearly, marker='o', fc='yellow', ls='-', lw=1, color='gold', alpha=1, label='O-E')
        for i in range(len(t[(y_fit_diff2>0).ravel()])):
            if i==0:
                plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color='black', alpha=0.2, label='LTR segment') 
            else:
                plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color='black', alpha=0.2) 
        for i in range(len(breakpoints)):
            if i==0:
                plt.plot( t[0:breakpoints[i]], np.tile( np.nanmean(diff_yearly[0:breakpoints[i]]), breakpoints[i] ), color='red')
                plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color='black', label='Breakpoint')
            else:
                plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color='black')    
                plt.plot( t[breakpoints[i-1]:breakpoints[i]], np.tile( np.nanmean(diff_yearly[breakpoints[i-1]:breakpoints[i]]), breakpoints[i]-breakpoints[i-1] ), color='red')
            if i==len(breakpoints)-1:
                plt.plot( t[breakpoints[i]:], np.tile( np.nanmean(diff_yearly[breakpoints[i]:]), len(t)-breakpoints[i] ), color='red', label='segment mean')
                
        ax.xaxis.grid(b=None, which='major', color='none', linestyle='-')
        ax.yaxis.grid(b=None, which='major', color='none', linestyle='-')
        plt.grid(b=None)
        plt.tick_params(labelsize=fontsize)  
        plt.xlabel('Year', fontsize=fontsize)
        plt.ylabel(r'Anomaly difference, $^{\circ}$C', fontsize=fontsize)
        plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
        plt.title( stationcode, color='black', fontsize=fontsize)           
        fig.tight_layout()
        plt.savefig(figstr, dpi=300)
        plt.close('all')    
        
        
    
