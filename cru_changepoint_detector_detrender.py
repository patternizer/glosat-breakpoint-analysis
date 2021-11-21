#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: cru_changepoint_detector_detrender.py
#------------------------------------------------------------------------------
#
# Version 0.1
# 21 November, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime

import scipy
from sklearn import linear_model
from sklearn.linear_model import *
import statsmodels.api as sm
from lineartree import LinearTreeClassifier, LinearTreeRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


#----------------------------------------------------------------------------------
import filter_cru_dft as cru_filter # CRU DFT filter
import cru_changepoint_detector as cru # CRU changepoint detector
#----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

stationcode = 'HadCRUT5'     # 
documented_change = np.nan

fontsize = 16 
use_dark_theme = False

plot_timeseries = True
plot_difference = True
plot_cusum = True
plot_adjustments = True
plot_seasonal = True

nfft = 10                     # decadal smoothing

#----------------------------------------------------------------------------
# DARK THEME
#----------------------------------------------------------------------------

if use_dark_theme == True:
    
    matplotlib.rcParams['text.usetex'] = False
#    rcParams['font.family'] = ['DejaVu Sans']
#    rcParams['font.sans-serif'] = ['Avant Garde']
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Avant Garde', 'Lucida Grande', 'Verdana', 'DejaVu Sans' ]
    plt.rc('text',color='white')
    plt.rc('lines',color='white')
    plt.rc('patch',edgecolor='white')
    plt.rc('grid',color='lightgray')
    plt.rc('xtick',color='white')
    plt.rc('ytick',color='white')
    plt.rc('axes',labelcolor='white')
    plt.rc('axes',facecolor='black')
    plt.rc('axes',edgecolor='lightgray')
    plt.rc('figure',facecolor='black')
    plt.rc('figure',edgecolor='black')
    plt.rc('savefig',edgecolor='black')
    plt.rc('savefig',facecolor='black')
    
else:

    matplotlib.rcParams['text.usetex'] = False
#    rcParams['font.family'] = ['DejaVu Sans']
#    rcParams['font.sans-serif'] = ['Avant Garde']
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Avant Garde', 'Lucida Grande', 'Verdana', 'DejaVu Sans' ]
    plt.rc('text',color='black')
    plt.rc('lines',color='black')
    plt.rc('patch',edgecolor='black')
    plt.rc('grid',color='lightgray')
    plt.rc('xtick',color='black')
    plt.rc('ytick',color='black')
    plt.rc('axes',labelcolor='black')
    plt.rc('axes',facecolor='white')    
    plt.rc('axes',edgecolor='black')
    plt.rc('figure',facecolor='white')
    plt.rc('figure',edgecolor='white')
    plt.rc('savefig',edgecolor='white')
    plt.rc('savefig',facecolor='white')

# Calculate current time

now = datetime.now()
currentdy = str(now.day).zfill(2)
currentmn = str(now.month).zfill(2)
currentyr = str(now.year)
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr    

#----------------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------------

def smooth_fft(x, span):  
    
    y_lo, y_hi, zvarlo, zvarhi, fc, pctl = cru_filter.cru_filter_dft(x, span)    
    x_filtered = y_lo

    return x_filtered
    
def linear_regression_ols(x,y):
    
    regr = linear_model.LinearRegression()
    # regr = TheilSenRegressor(random_state=42)
    # regr = RANSACRegressor(random_state=42)
    
    X = x.reshape(len(x),1)
    xpred = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
    regr.fit(X, y)
    ypred = regr.predict(xpred.reshape(-1, 1))
    slope = regr.coef_[0]
    intercept = regr.intercept_     
        
    return xpred, ypred, slope, intercept

#------------------------------------------------------------------------------
# LOAD: CUSUM timeseries from local expectation Kriging (LEK) analysis
#------------------------------------------------------------------------------

ds = xr.open_dataset('DATA/HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.nc', decode_cf=True)
t_monthly = pd.date_range(start=str(ds.time.dt.year[0].values), periods=len(ds.time), freq='M')
ts_monthly = ds.tas_mean.values
df = pd.DataFrame({'datetime':t_monthly, 'ts_monthly':ts_monthly})

# TRIM: to start of Pandas datetime range
        
df = df[ (df.datetime.dt.year >= 1678) & (df.datetime.dt.year <= 2020) ].reset_index(drop=True)
t_monthly = pd.date_range(start=str(df['datetime'].iloc[0]), periods=len(df.ts_monthly), freq='M')     
       
# COMPUTE: 12-m MA
    
ts_monthly = df.ts_monthly.rolling(12, center=True).mean().values
    
# COMPUTE: CUSUM
     
t = t_monthly 
a = ts_monthly  	
c = np.nancumsum( a )
x = ( np.arange(len(c)) / len(c) )
y = c

if np.isnan(documented_change):
    documented_change_datetime = np.nan
else:        
    documented_change_datetime = pd.to_datetime('01-01-'+str(documented_change),format='%d-%m-%Y')

#------------------------------------------------------------------------------
# CALL: cru_changepoint_detector
#------------------------------------------------------------------------------

y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)

# CALCULATE: intra-breakpoint fragment means
        
y_means = []
adjustments = []
for j in range(len(breakpoints)+1):                
    if j == 0:              
        y_means = y_means + list( np.tile( -np.nanmean(ts_monthly[0:breakpoints[j]]), breakpoints[j] ) )         
        adjustment = [ -np.nanmean(ts_monthly[0:breakpoints[j]]) ]
    if (j > 0) & (j<len(breakpoints)):
        y_means = y_means + list( np.tile( -np.nanmean(ts_monthly[breakpoints[j-1]:breakpoints[j]]), breakpoints[j]-breakpoints[j-1] )) 
        adjustment = [ -np.nanmean(ts_monthly[breakpoints[j-1]:breakpoints[j]]) ]
    if (j == len(breakpoints)):              
        y_means = y_means + list( np.tile( -np.nanmean(ts_monthly[breakpoints[-1]:]), len(ts_monthly)-breakpoints[-1] ) ) 
        adjustment = [ -np.nanmean(ts_monthly[breakpoints[-1]:]) ]
    adjustments.append(adjustment)
        
y_means = np.array( y_means ) 
adjustments = np.array(adjustments).ravel()

#------------------------------------------------------------------------------
# WRITE: breakpoints and segment adjustments to CSV
#------------------------------------------------------------------------------

file_breakpoints = stationcode + '-' + 'breakpoints.csv'    
file_adjustments = stationcode + '-' + 'adjustments.csv'    

df_breakpoints = pd.DataFrame( {'breakpoint':t[breakpoints]}, index=np.arange(1,len(breakpoints)+1) )
df_breakpoints.to_csv( file_breakpoints )
df_adjustments = pd.DataFrame( {'adjustment':adjustments}, index=np.arange(1,len(adjustments)+1) )
df_adjustments.to_csv( file_adjustments )    

#==============================================================================
# PLOTS
#==============================================================================

if use_dark_theme == True:
    default_color = 'white'
else:    
    default_color = 'black'    	
                    
#------------------------------------------------------------------------------
# PLOT: CUSUM(O-E) with LTR fit, LTR boundaries, slopes and breakpoints
#------------------------------------------------------------------------------

if plot_cusum == True:
	
	figstr = stationcode + '-' + 'cusum.png'                

	fig, ax = plt.subplots(figsize=(15,10))
	plt.plot( t, y, color='blue', ls='-', lw=3, label='CUSUM (O-E)')
	plt.plot( t, y_fit, color='red', ls='-', lw=2, label='LTR fit')
	plt.fill_between( t, slopes, 0, color='lightblue', alpha=0.5, label='CUSUM/decade' )    
	ylimits = plt.ylim()    
	if ~np.isnan(documented_change): plt.axvline(x=documented_change_datetime, ls='-', lw=2, color='gold', label='Documented change: ' + str(documented_change) )                   
	for i in range(len(t[(y_fit_diff2>0).ravel()])):
		if i==0: plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2, label='LTR boundary') 
		else: plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2) 
	for i in range(len(breakpoints)):
		if i==0: plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color, label='Breakpoint')
		else: plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color)    
	plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
	ax.xaxis.grid(b=None, which='major', color='none', linestyle='-')
	ax.yaxis.grid(b=None, which='major', color='none', linestyle='-')
	plt.grid(b=None)
	plt.tick_params(labelsize=fontsize)    
	plt.xlabel('Year', fontsize=fontsize)
	plt.ylabel(r'CUSUM (O-E), $^{\circ}$C', fontsize=fontsize)
	plt.title( stationcode + ': depth=' + str(depth) + r' ( BEST ): $\rho$=' + str( f'{r[depth-1]:03f}' ), color=default_color, fontsize=fontsize)      
	fig.legend(loc='lower center', ncol=5, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)       
	fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)       
	plt.savefig(figstr, dpi=300)
	plt.close('all')    

#------------------------------------------------------------------------------
# PLOT: O, O(adjusted) and E with breakpoints and adjustments
#------------------------------------------------------------------------------
        
if plot_adjustments == True:
	    
    figstr = stationcode + '-' + 'ltr-ols.png'                

    fig, ax = plt.subplots(figsize=(15,10))
    plt.scatter(t, a, marker='o', fc='blue', ls='-', lw=1, color='blue', alpha=0.5, label='O')
    plt.scatter(t, a + y_means, marker='o', fc='lightblue', ls='-', lw=1, color='lightblue', alpha=0.5, label='O (detrended)')    
    for i in range(len(t[(y_fit_diff2>0).ravel()])):
        if i==0:
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2, label='LTR boundary') 
        else:
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2)        
    for i in range(len(breakpoints)):
        if i==0:
            plt.plot( t[0:breakpoints[i]], np.tile( -np.nanmean(a[0:breakpoints[i]]), breakpoints[i] ), ls='-', lw=3, color='gold')
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color, label='Breakpoint')

            # FIT: OLS to segment

            T = t[0:breakpoints[i]]
            X = x[0:breakpoints[i]]
            Y = a[0:breakpoints[i]]
            mask = np.isfinite(Y)
            T = T[mask]
            X = X[mask]
            Y = Y[mask]
            xpred, ypred, slope, intercept = linear_regression_ols(X,Y)
            plt.scatter(T, ypred, marker='.', fc='red', ls='-', lw=1, color='red', alpha=0.5, label='OLS trend')    

        else:
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color)    
            plt.plot( t[breakpoints[i-1]:breakpoints[i]], np.tile( -np.nanmean(a[breakpoints[i-1]:breakpoints[i]]), breakpoints[i]-breakpoints[i-1] ), ls='-', lw=3, color='gold')

            # FIT: OLS to segment
            
            T = t[breakpoints[i-1]:breakpoints[i]]
            X = x[breakpoints[i-1]:breakpoints[i]]
            Y = a[breakpoints[i-1]:breakpoints[i]]
            mask = np.isfinite(Y)
            T = T[mask]
            X = X[mask]
            Y = Y[mask]
            xpred, ypred, slope, intercept = linear_regression_ols(X,Y)
            plt.scatter(T, ypred, marker='.', fc='red', ls='-', lw=1, color='red', alpha=0.5)    

        if i==len(breakpoints)-1:
            plt.plot( t[breakpoints[i]:], np.tile( -np.nanmean(a[breakpoints[i]:]), len(t)-breakpoints[i] ), ls='-', lw=3, color='gold', label='Adjustment')                
            
            # FIT: OLS to segment

            T = t[breakpoints[i]:]
            X = x[breakpoints[i]:]
            Y = a[breakpoints[i]:]
            mask = np.isfinite(Y)
            T = T[mask]
            X = X[mask]
            Y = Y[mask]
            xpred, ypred, slope, intercept = linear_regression_ols(X,Y)
            plt.scatter(T, ypred, marker='.', fc='red', ls='-', lw=1, color='red', alpha=0.5)    
            
    plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
    ax.xaxis.grid(b=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(b=None, which='major', color='none', linestyle='-')
    plt.grid(b=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Temperature anomaly (from 1961-1990), $^{\circ}$C', fontsize=fontsize)
    plt.title( stationcode + ': depth=' + str(depth) + r' ( BEST ): $\rho$=' + str( f'{r[depth-1]:03f}' ), color=default_color, fontsize=fontsize)      
    fig.legend(loc='lower center', ncol=6, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)       
    plt.savefig(figstr, dpi=300)
    plt.close('all')                      
    
#------------------------------------------------------------------------------
print('** END')
