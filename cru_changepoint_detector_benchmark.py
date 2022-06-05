#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: cru_changepoint_detector_benchmark.py
#------------------------------------------------------------------------------
#
# Version 0.1
# 15 December, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns; sns.set()
from datetime import datetime
import netCDF4
from netCDF4 import Dataset, num2date, date2num

# System libraries:
import os

#------------------------------------------------------------------------------
import filter_cru_dft as cru_filter # CRU DFT filter
import cru_changepoint_detector as cru # CRU changepoint detector
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def find_nearest(array, value):
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

fontsize = 16 
default_color = 'black'

use_reproducible = False
use_colocation = True
use_smoothing = False
plot_signal = True
plot_cusum = True
plot_benchmark = True

amplitude = 1.0     # degC
noise_mean = 0.0    # degC
noise_sd = 0.1      # degC
min_segment = 240   # 2 decades
nbreakpoints = 8   

level_size = 0.2
slope_size = 0.6 # degC / decade

benchmark_type = 1 # steps ( no seasonality )
#benchmark_type = 2 # steps + seasonality 
#benchmark_type = 3 # linear trends ( no seasonality )
#benchmark_type = 4 # linear trends + seasonality

#==============================================================================
# GENERATE: benchmark timeseries
#==============================================================================
        
# SET: monthly time vector

t_monthly = pd.date_range(start='1781', end='2021', freq='MS')[:-1]

# GENERATE: annual cycle in monthly data 1781-2021

t = np.arange( len(t_monthly) )  # index of monthly datetimes
k = ( t_monthly.year[-1]-t_monthly.year[0]+1 ) # number of years
tone = amplitude * np.sin( 2*np.pi*k*t/len(t) ) # annual cycle    
seasonality = noise_sd * np.sin( 2*np.pi*k*t/len(t) ) # annual cycle    

# GENERATE: white noise N( mean, std ) and a priori breaks

allowed_levels = list( range(-20, 20+1) )
allowed_slopes = list( range(-1, 1+1) )

if use_reproducible == True:
    
    rng = np.random.default_rng(20211215) # seed for reproducibility
    noise = rng.normal( loc=noise_mean, scale=noise_sd, size=len(t))
    levels = [ rng.choice( allowed_levels ) * level_size for i in range( nbreakpoints + 1 ) ]
    slope_sign = [ rng.choice( allowed_slopes ) * slope_size for i in range( nbreakpoints + 1 ) ]
            
else:
    
    noise = np.random.normal( loc=0.0, scale=0.1, size=len(t))
    levels = [ random.choice(allowed_levels) * level_size for i in range( nbreakpoints + 1 ) ]
    slope_sign = [ random.choice( allowed_slopes) * slope_size for i in range( nbreakpoints + 1 ) ]
    
idx = np.zeros( nbreakpoints )
while ( np.diff(idx) < min_segment ).sum() > 0: idx = np.sort( np.random.randint( min_segment, len(t) - min_segment, size = nbreakpoints ) )

if benchmark_type == 1:
    
    #------------------------------------------------------------------------------
    # GENERATE: benchmark Type I: steps ( no seasonality )
    #------------------------------------------------------------------------------
        
    benchmark = []
    x = np.arange( len(t) )
    y = np.ones( len(t) )
    for i in range( nbreakpoints + 1 ): 
        if i == 0: 
            segment = list( y[ 0:idx[i] ] * levels[i] )   
        elif ( i > 0 ) & ( i < nbreakpoints ):        
            segment = list( y[ idx[i-1]:idx[i] ] * levels[i] ) 
        elif i == (nbreakpoints ): 
            segment = list( y[ idx[i-1]: ] * levels[i] ) 
        benchmark += segment

elif benchmark_type == 2:    

    #------------------------------------------------------------------------------
    # GENERATE: benchmark Type II: steps + seasonality
    #------------------------------------------------------------------------------
    
    benchmark = []
    x = np.arange( len(t) )
    y = np.ones( len(t) )
    for i in range( nbreakpoints + 1 ): 
        if i == 0: 
            segment = list( y[ 0:idx[i] ] * levels[i] + seasonality[ 0:idx[i] ] )   
        elif ( i > 0 ) & ( i < nbreakpoints ):        
            segment = list( y[ idx[i-1]:idx[i] ] * levels[i] + seasonality[ idx[i-1]:idx[i] ] ) 
        elif i == (nbreakpoints ): 
            segment = list( y[ idx[i-1]: ] * levels[i] + seasonality[ idx[i-1]: ] ) 
        benchmark += segment

elif benchmark_type == 3:    

    #------------------------------------------------------------------------------
    # GENERATE: benchmark Type III: linear trends ( no seasonality )
    #------------------------------------------------------------------------------
        
    benchmark = []
    x = np.arange( len(t) )
    y = np.ones( len(t) )
    for i in range( nbreakpoints + 1 ): 
        if i == 0:             
            segment = list( levels[0] + ( slope_sign[i] / ( idx[i] - 0 ) )  * ( x[ 0:idx[i] ] - 0 ) )   
            level_start = segment[-1]
        elif ( i > 0 ) & ( i < nbreakpoints ):        
            segment = list( level_start + ( slope_sign[i] / ( idx[i] - idx[i-1] ) )  * ( x[ idx[i-1]:idx[i] ] - idx[i-1] ) ) 
            level_start = segment[-1]
        elif i == (nbreakpoints ): 
            segment = list( level_start + ( slope_sign[i] / ( len(x) - idx[i-1] ) )  * ( x[ idx[i-1]: ] - idx[i-1] ) ) 
        benchmark += segment

    #------------------------------------------------------------------------------
    # GENERATE: benchmark Type IV: linear trends + seasonality
    #------------------------------------------------------------------------------

elif benchmark_type == 4:    
    
    benchmark = []
    x = np.arange( len(t) )
    y = np.ones( len(t) )
    for i in range( nbreakpoints + 1 ): 
        if i == 0: 
            segment = list( levels[0] + ( slope_sign[i] / ( idx[i] - 0 ) )  * ( x[ 0:idx[i] ] - 0 )  + seasonality[ 0:idx[i] ] )   
            level_start = segment[-1]
        elif ( i > 0 ) & ( i < nbreakpoints ):        
            segment = list( level_start + ( slope_sign[i] / ( idx[i] - idx[i-1] ) )  * ( x[ idx[i-1]:idx[i] ] - idx[i-1] ) + seasonality[ idx[i-1]:idx[i] ] ) 
            level_start = segment[-1]
        elif i == (nbreakpoints ): 
            segment = list( level_start + ( slope_sign[i] / ( len(x) - idx[i-1] ) )  * ( x[ idx[i-1]: ] - idx[i-1] ) + seasonality[ idx[i-1]: ] ) 
        benchmark += segment

#------------------------------------------------------------------------------
# GENERATE: surrogate timeseries
#------------------------------------------------------------------------------

ts_monthly = tone + noise + benchmark

# COMPUTE: 12-m MA
    
if use_smoothing == True:    
    ts = pd.Series( ts_monthly ).rolling(12, center=True).mean().values
else:
    ts = ts_monthly
t = t_monthly
       
# COMPUTE: CUSUM
        	
y = np.nancumsum( ts )
x = ( np.arange(len(y)) / len(y) )

#------------------------------------------------------------------------------
# CALL: cru_changepoint_detector
#------------------------------------------------------------------------------

y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector( x, y )

if benchmark_type > 2:

    # breakpoint detection for linear trends
    
    mask_linear = np.isfinite( slopes )
    
    import scipy
    from sklearn import linear_model
    from sklearn.linear_model import *
    import statsmodels.api as sm
    from lineartree import LinearTreeClassifier, LinearTreeRegressor
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    
    max_depth = 6                        
    max_bins = 30          # range[10,120]
    min_samples_leaf = 3   # > 2 
    
    lt = LinearTreeRegressor(
        base_estimator = LinearRegression(),
        min_samples_leaf = min_samples_leaf,
        max_bins = max_bins,
        max_depth = max_depth        
    ).fit(x[mask_linear].reshape(-1,1), slopes[mask_linear].reshape(-1,1))    
    y_fitB = lt.predict(x[mask_linear].reshape(-1,1))          
    
    y_fitC, y_fit_diffC, y_fit_diff2C, slopesC, breakpointsC, depthC, rC, R2adjC = cru.changepoint_detector( x[mask_linear], y_fitB )
            
    #slopes_diff = np.array( [0.0] + list( np.diff( slopes, 1 ) ) )
    #slopes_diff[ np.abs( slopes_diff ) < 1e-6 ] = np.nan
    #x_linear = np.arange( len(t) )
    #mask_linear = np.isfinite( slopes_diff )
    #idx_linear = x_linear[ mask_linear ]
    #slopes_diff_change = [0.0] + list( np.diff( slopes_diff[ mask_linear ], 1 ) )
    #breakpoint_idx = slopes_diff_change > np.percentile( np.abs( slopes_diff_change ), 50 )
    #breakpoints = idx_linear[ breakpoint_idx]
    
    breakpoints = breakpointsC
     
#------------------------------------------------------------------------------
# COMPUTE: breakpoint timing errors
#------------------------------------------------------------------------------

if use_colocation == True:
    
    keep = np.array( [np.nan] + list(np.diff( levels )) ) !=0
    idx_a_priori = np.array( [np.nan] + list(idx) )[ keep ]
    levels = np.array(levels)[keep]
    idx = idx_a_priori[ np.isfinite(idx_a_priori) ].astype(int)
    nbreakpoints = len(idx)    

    bp = []
    for i in range(len(idx)):
        
        value = idx[i]
        loc = breakpoints[ find_nearest(breakpoints, value) ]
        if np.abs( loc - value ) < 120:

            bp.append( loc )

        else:

            bp.append( np.nan )            

    # STORE: time errors
    
    errors = np.abs( idx - bp )
    uncertainty = np.ceil( np.nanmean( errors ) ).astype(int)

    # USE: ensemble estimates of uncertainty
    
    if benchmark_type < 3:
        
        uncertainty = np.ceil( 20.5 ).astype(int)
        
    else:
        
        uncertainty = np.ceil( 49.5 ).astype(int)
    
#------------------------------------------------------------------------------
# COMPUTE: adjustment magnitude / timeseries s.d. 
#------------------------------------------------------------------------------

s = np.nanstd( ts )
d = np.diff( levels )
ratios = d / s

print('levels:', levels)
print('ratios:', ratios)
print('errors:', errors)
print('uncertainty:', uncertainty)
print('bp idx ( a priori ):', idx)
print('bp idx ( detected ):', breakpoints)
print('bp idx ( nearest ):', bp)

#==============================================================================
# PLOTS
#==============================================================================

#------------------------------------------------------------------------------
# PLOT: signal + benchmark
#------------------------------------------------------------------------------

if plot_signal == True:
	    
    figstr = 'benchmark-baseline.png'       
                 
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(t, tone, marker='o', ls='-', lw=1, color='black', alpha=0.5, label='Annual cycle: amplitude=' + str(amplitude) )
    plt.plot(t, noise, marker='*', ls='-', lw=1, color='teal', alpha=0.5, label='White noise: N(' + str(noise_mean) + ',' + str(noise_sd) + ')' )
    plt.plot(t, tone+noise, marker='.', ls='-', lw=0.5, color='purple', alpha=1, label='Baseline signal' )
    plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
    plt.xlim( pd.to_datetime( t_monthly[0], format='%Y-%m-%d' ), pd.to_datetime( t_monthly[120], format='%Y-%m-%d' ) )
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Temperature anomaly, $^{\circ}$C', fontsize=fontsize)
    fig.legend(loc='lower center', ncol=5, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)  
    plt.title( 'Baseline components', fontsize=fontsize)      
    plt.savefig(figstr, dpi=300)
    plt.close('all')    

#------------------------------------------------------------------------------
# PLOT: benchmark cusum analysis
#------------------------------------------------------------------------------

if plot_cusum == True:

    figstr = 'benchmark' + '-' + str(benchmark_type) + '-' + 'cusum.png'                

    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot( t, y, color='blue', ls='-', lw=3, label='CUSUM')
    plt.plot( t, y_fit, color='red', ls='-', lw=2, label='LTR fit')
    plt.fill_between( t, slopes, 0, color='lightblue', alpha=0.5, label='CUSUM/decade' )    
    #plt.scatter( t, slopes_diff, color='red', alpha=0.5, label='CUSUM diff' )    
    for i in range(len(t[(y_fit_diff2>0).ravel()])):
        if i==0: plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2, label='LTR boundary') 
        else: plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2) 
    for i in range(len(breakpoints)):
        if i==0: plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color=default_color, label='Breakpoint')
        else: plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color=default_color)    
    for i in range(nbreakpoints):
        if i==0:
            plt.axvline( t[ idx[i] ], ls='dashed', lw=2, color='blue', label='A Priori')
        else:
            plt.axvline( t[ idx[i]] , ls='dashed', lw=2, color='blue')                    
    plt.tick_params(labelsize=fontsize)    
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'CUSUM, $^{\circ}$C', fontsize=fontsize)
    plt.title( 'Benchmark ' + str(benchmark_type) + ' cusum analysis: depth=' + str(depth) + r' ( BEST ): $\rho$=' + str( f'{r[depth-1]:03f}' ), color=default_color, fontsize=fontsize)      
    fig.legend(loc='lower center', ncol=5, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)       
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)       
    plt.savefig(figstr, dpi=300)
    plt.close('all')              
    
#------------------------------------------------------------------------------
# PLOT: benchmark timeseries
#------------------------------------------------------------------------------

if plot_benchmark == True:
	    
    figstr = 'benchmark' + '-' + str(benchmark_type) + '-' + 'breakpoints.png'       
                 
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(t, ts, marker='.', ls='-', lw=0.5, color='purple', alpha=0.2, label='Benchmark series')
    plt.plot(t, benchmark, ls='-', lw=3, color='purple', alpha=1, label='Benchmark adjustments')
    ylimits = plt.ylim()
    for i in range(len(breakpoints)):
        if i==0:
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color='black', label='Breakpoint')
            plt.fill_betweenx( ylimits, t[ breakpoints[i] - uncertainty ], t[ breakpoints[i] + uncertainty ], facecolor='grey', alpha=0.5, label=r'$\sigma$=' + str(uncertainty) + ' months')    
        else:
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color='black')    
            plt.fill_betweenx( ylimits, t[ breakpoints[i] - uncertainty ], t[ breakpoints[i] + uncertainty ], facecolor='grey', alpha=0.5)                                        
    for i in range(nbreakpoints):
        if i==0:
            plt.axvline( t[ idx[i] ], ls='-', lw=1, color='black', label='A priori')
        else:
            plt.axvline( t[ idx[i]], ls='-', lw=1, color='black')    
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Temperature anomaly, $^{\circ}$C', fontsize=fontsize)
    fig.legend(loc='lower center', ncol=5, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)  
    plt.title( 'Benchmark ' + str(benchmark_type) + ' timeseries and breakpoints', fontsize=fontsize)      
    plt.savefig(figstr, dpi=300)
    plt.close('all')        

#------------------------------------------------------------------------------
print('** END')
