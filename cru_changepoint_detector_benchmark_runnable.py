#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: cru_changepoint_detector_benchmark.py
#------------------------------------------------------------------------------
#
# Version 0.2
# 19 January, 2022
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
import random
import netCDF4
from netCDF4 import Dataset, num2date, date2num

# Statistics libraries:
import scipy
from sklearn import linear_model
from sklearn.linear_model import *
import statsmodels.api as sm
from lineartree import LinearTreeClassifier, LinearTreeRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# System libraries:
import os

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

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

amplitude = 1.0     # degC
noise_mean = 0.0    # degC
noise_sd = 0.1      # degC
min_segment = 240   # 2 decades
nbreakpoints = 8   
level_size = 0.2
slope_size = 0.6 # degC / decade

#benchmark_type = 1 # steps ( no seasonality )
#benchmark_type = 2 # steps + seasonality 
benchmark_type = 3 # linear trends ( no seasonality )
#benchmark_type = 4 # linear trends + seasonality

simulations = 1000

#==============================================================================
# GENERATE: baseline
#==============================================================================
        
# SET: monthly time vector

t_monthly = pd.date_range(start='1781', end='2021', freq='MS')[:-1]

# GENERATE: annual cycle in monthly data 1780-2020

t = np.arange( len(t_monthly) )  # index of monthly datetimes
k = ( t_monthly.year[-1]-t_monthly.year[0]+1 ) # number of years
tone = amplitude * np.sin( 2*np.pi*k*t/len(t) ) # annual cycle    
seasonality = noise_sd * np.sin( 2*np.pi*k*t/len(t) ) # annual cycle    

# GENERATE: white noise N( mean, std ) and a priori breaks

allowed_levels = list( range(-20, 20+1) )
allowed_slopes = list( range(-1, 1+1) )

ensemble_errors = []
ensemble_jumps = []
ensemble_ratios = []

for loop in range( simulations ):
            
    #==============================================================================
    # GENERATE: benchmark timeseries
    #==============================================================================
    
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
    # CONSTRUCT: surrogate timeseries
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
        
        
        max_depth = 6                        
        min_separation = 30                         # [30,360] NB: 120 = 1 decade
        max_bins = int( min_separation/3 ) 			# 1/3 of min_samples_leaf in range[10,120]
        if max_bins < 10: max_bins = 10
        if max_bins > 120: max_bins = 120
        min_samples_leaf = 3                        # > 2 
        
        lt = LinearTreeRegressor(
            base_estimator = LinearRegression(),
            min_samples_leaf = min_samples_leaf,
            max_bins = max_bins,
            max_depth = max_depth        
        ).fit(x[mask_linear].reshape(-1,1), slopes[mask_linear].reshape(-1,1))    
        y_fitB = lt.predict(x[mask_linear].reshape(-1,1))          
        
        y_fitC, y_fit_diffC, y_fit_diff2C, slopesC, breakpointsC, depthC, rC, R2adjC = cru.changepoint_detector( x[mask_linear], y_fitB )
                
        breakpoints = breakpointsC
         
    #------------------------------------------------------------------------------
    # COMPUTE: breakpoint timing errors
    #------------------------------------------------------------------------------
    
    if use_colocation == True:
        
        keep = np.array( [np.nan] + list(np.diff( levels )) ) !=0
        idx_a_priori = np.array( [np.nan] + list(idx) )[ keep ]
        levels = np.array(levels)[keep]
        idx = idx_a_priori[ np.isfinite(idx_a_priori) ].astype(int)
    
        bp = []
        for i in range(len(idx)):
            
            value = idx[i]
            loc = breakpoints[ find_nearest(breakpoints, value) ]
            if np.abs( loc - value ) < 120:
    
                bp.append( loc )
    
            else:
    
                bp.append( np.nan )            
        
        errors = np.abs( idx - bp )

    #------------------------------------------------------------------------------
    # COMPUTE: jumps and jump/sd ratios 
    #------------------------------------------------------------------------------
    
    s = np.nanstd( ts )
    jumps = np.diff( levels )
    ratios = jumps / s

    ensemble_errors += list( errors )    
    ensemble_jumps += list( jumps )    
    ensemble_ratios += list( ratios )    

    #------------------------------------------------------------------------------
    # SAVE: to CSV
    #------------------------------------------------------------------------------

    df = pd.DataFrame( { 'ensemble_errors':ensemble_errors, 'ensemble_jumps':ensemble_jumps, 'ensemble_ratios':ensemble_ratios } )
    df.to_csv( 'ensemble_stats.csv' )

#    df = pd.read_csv( 'ensemble_stats.csv', index_col=0 )

#------------------------------------------------------------------------------
# COMPUTE: mean values
#------------------------------------------------------------------------------

mean_uncertainty = np.nanmean( df.ensemble_errors ) 
mean_jump = np.nanmean( df.ensemble_jumps )
mean_ratio = np.nanmean( df.ensemble_ratios )

print( 'mean_uncertainty:', mean_uncertainty )
print( 'mean_jump:', mean_jump )
print( 'mean_ratio:', mean_ratio )

#------------------------------------------------------------------------------
# PLOTS:
#------------------------------------------------------------------------------

fig, ax = plt.subplots()
sns.ecdfplot(df.ensemble_errors)
plt.axvline( x = mean_uncertainty, color='k', ls='--', label='mean=' + str(np.round( mean_uncertainty, 1)) + ' months' )
plt.legend()
plt.xlabel( 'Breakpoint timing error' )
plt.ylabel( 'ECDF proportion' )
plt.title( 'CASE 3: 1000 simulated benchmark timeseries ( ~ 8 breaks each )')
plt.savefig('benchmark-ensemble-timing-errors', dpi=300)

fig, ax = plt.subplots()
sns.distplot( df.ensemble_ratios, bins=37, kde=False)
plt.xlabel( 'Jump / s.d.' )
plt.ylabel( 'Frequency' )
plt.title( 'CASE 3: 1000 simulated benchmark timeseries ( ~ 8 breaks each )')
plt.savefig('benchmark-ensemble-histogram-ratios', dpi=300)

fig, ax = plt.subplots()
sns.distplot( df.ensemble_errors, bins=57, kde=False )
plt.xlabel( 'Breakpoint timing error' )
plt.ylabel( 'Frequency' )
plt.title( 'CASE 3: 1000 simulated benchmark timeseries ( ~ 8 breaks each )')
plt.savefig('benchmark-ensemble-histogram-errors', dpi=300)

#------------------------------------------------------------------------------
print('** END')
