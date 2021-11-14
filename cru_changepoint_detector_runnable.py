#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: cru_changepoint_detector_static.py
#------------------------------------------------------------------------------
#
# Version 0.2
# 2 November, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

# Numerics and dataframe libraries:
import numpy as np
import pandas as pd
import pickle

# Datetime libraries:
#from datetime import datetime
#import nc_time_axis
#import cftime
#from cftime import num2date, DatetimeNoLeap

# OS libraries:
import os, sys
from  optparse import OptionParser
#import argparse

# Plotting libraries:
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.close('all')
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
matplotlib.rcParams['text.usetex'] = False

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#------------------------------------------------------------------------------

#----------------------------------------------------------------------------
# DARK BACKGROUND THEME
#----------------------------------------------------------------------------
matplotlib.rcParams['text.usetex'] = False
rcParams['font.family'] = ['DejaVu Sans']
rcParams['font.sans-serif'] = ['Avant Garde']
plt.rc('text',color='white')
plt.rc('lines',color='white')
plt.rc('patch',edgecolor='white')
plt.rc('grid',color='lightgray')
plt.rc('xtick',color='white')
plt.rc('ytick',color='white')
plt.rc('axes',edgecolor='lightgray')
plt.rc('axes',facecolor='black')
plt.rc('axes',labelcolor='white')
plt.rc('figure',facecolor='black')
plt.rc('figure',edgecolor='black')
plt.rc('savefig',edgecolor='black')
plt.rc('savefig',facecolor='black')

# Calculate current time

#now = datetime.now()
#currentdy = str(now.day).zfill(2)
#currentmn = str(now.month).zfill(2)
#currentyr = str(now.year)
#titletime = str(currentdy) + '/' + currentmn + '/' + currentyr

def calculate_adjustments(stationcode):
    
    #------------------------------------------------------------------------------
    import cru_changepoint_detector as cru # CRU changepoint detector
    #------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------
    # METHODS
    #----------------------------------------------------------------------------------
    
    def moving_average(x, w):
      """
      Calculate moving average of a vector
      
      Parameters:
        x (vector of float): the vector to be smoothed
        w (int): the number of samples over which to average
        
      Returns:
        (vector of float): smoothed vector, which is shorter than the input vector
      """
      return np.convolve(x, np.ones(w), 'valid') / w
   
    #-----------------------------------------------------------------------------
    # SETTINGS
    #-----------------------------------------------------------------------------
    
    fontsize = 16
               
    plot_timeseries = True
    plot_differences = True
    plot_changepoints = True
    plot_adjustments = True
        
    #------------------------------------------------------------------------------
    # LOAD: LEK global dataframe
    #------------------------------------------------------------------------------
    
    df_temp = pd.read_pickle('DATA/df_temp_expect_reduced.pkl', compression='bz2')
    
    df = df_temp[ df_temp['stationcode'] == stationcode ].sort_values(by='year').reset_index(drop=True).dropna()
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    sd = (df.groupby('year').mean().iloc[:,43:55]).reset_index()        
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel()    
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel()    
    sd_monthly = np.array( sd.groupby('year').mean().iloc[:,0:12]).ravel()           
    
    if len(ts_monthly) == 0:    
        print('STATION DATA: not available. Returning ...')
        breakpoints = []
        adjustments = []
        return breakpoints, adjustments
    else:
        print('STATION DATA:', ts_monthly)
        
    ts_monthly = np.array( moving_average( ts_monthly, 12 ) )    
    ex_monthly = np.array( moving_average( ex_monthly, 12 ) )    
    sd_monthly = np.array( moving_average( sd_monthly, 12 ) )        
    diff_monthly = ts_monthly - ex_monthly
    
    # Solve Y1677-Y2262 Pandas bug with Xarray:        
    # t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')     
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')     
    mask = np.isfinite(ex_monthly)
    
    # CALCULATE: CUSUM
        	
    x = t_monthly[mask]
    y = np.cumsum( diff_monthly[mask] )
    
    # CALL: cru_changepoint_detector
    
    #y_fit, y_fit_diff, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
    #y_fit_diff = np.array( y_fit_diff ) 
    y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
       
    # COMPUTE: breakpoints
    
    breakpoints_all = x[mask][ np.abs(y_fit_diff) >= np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)) ][0:]    
    breakpoints_idx = np.arange(len(x[mask]))[ np.abs(y_fit_diff) >= np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)) ][0:]        
    breakpoints = pd.to_datetime( [x[mask][0]] + list( breakpoints_all ) )
    
    # CALCULATE: intra-breakpoint fragment means
        
    y_means = []
    adjustments = []
    for j in range(len(breakpoints_all)+1):                
        if j == 0:              
            y_means = y_means + list( len( ts_monthly[mask][0:breakpoints_idx[0]] ) * [ -np.nanmean(ts_monthly[mask][0:breakpoints_idx[0]]) + np.nanmean(ex_monthly[mask][0:breakpoints_idx[0]]) ] ) 
            adjustment = [ -np.nanmean(ts_monthly[mask][0:breakpoints_idx[0]]) + np.nanmean(ex_monthly[mask][0:breakpoints_idx[0]]) ]
        if (j > 0) & (j<len(breakpoints_all)):
            y_means = y_means + list( len( ts_monthly[mask][breakpoints_idx[j-1]:breakpoints_idx[j]] ) * [ -np.nanmean(ts_monthly[mask][breakpoints_idx[j-1]:breakpoints_idx[j]]) + np.nanmean(ex_monthly[mask][breakpoints_idx[j-1]:breakpoints_idx[j]]) ] ) 
            adjustment = [ -np.nanmean(ts_monthly[mask][breakpoints_idx[j-1]:breakpoints_idx[j]]) + np.nanmean(ex_monthly[mask][breakpoints_idx[j-1]:breakpoints_idx[j]]) ]
        if (j == len(breakpoints_all)):              
            y_means = y_means + list( len( ts_monthly[mask][breakpoints_idx[-1]:] ) * [ -np.nanmean(ts_monthly[mask][breakpoints_idx[-1]:]) + np.nanmean(ex_monthly[mask][breakpoints_idx[-1]:]) ] ) 
            adjustment = [ -np.nanmean(ts_monthly[mask][breakpoints_idx[-1]:]) + np.nanmean(ex_monthly[mask][breakpoints_idx[-1]:]) ]
        adjustments.append(adjustment)
    
    y_means = np.array( y_means ) 
    adjustments = np.array(adjustments).ravel()
    
    # SAVE: breakpoints and adjustments to CSV
    
    db = pd.DataFrame({'breakpoint':breakpoints, 'adjustment-to-next-breakpoint':adjustments})
    db.to_csv(stationcode + '-' + 'breakpoints-and-adjustments.csv')
    
    #------------------------------------------------------------------------------
    # PLOT: station timeseries + LEK + LEK uncertainty
    #------------------------------------------------------------------------------
    
    if plot_timeseries == True:
       
        figstr = stationcode + '-' + 'timeseries' + '.png'
        
        fig, ax = plt.subplots(figsize=(15,10))
        plt.scatter(t_monthly[mask], ts_monthly[mask], marker='o', fc='navy', ls='-', lw=1, color='blue', alpha=0.5, label='O')
        plt.scatter(t_monthly[mask], ex_monthly[mask], marker='o', fc='maroon', ls='-', lw=1, color='red', alpha=0.5, label='E')
        plt.fill_between(t_monthly[mask], ex_monthly[mask]-sd_monthly[mask], ex_monthly[mask]+sd_monthly[mask], color='grey', alpha=0.2, label='uncertainty')
        plt.legend(loc='lower right', ncol=2, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)       
        plt.tick_params(labelsize=fontsize)    
        plt.xlabel('Year', fontsize=fontsize)
        plt.ylabel(r'Anomaly (from 1961-1990), $^{\circ}$C', fontsize=fontsize)
        plt.title( stationcode, color='white', fontsize=fontsize)           
        fig.tight_layout()
        plt.savefig(figstr, dpi=300)
        plt.close('all')   
    
    if plot_differences == True:
       
        figstr = stationcode + '-' + 'differences' + '.png'
    
        fig, ax = plt.subplots(figsize=(15,10))
        plt.scatter(t_monthly[mask], diff_monthly[mask], marker='o', fc='gold', ls='-', lw=1, color='yellow', alpha=0.5, label='O-E')
        plt.fill_between(t_monthly[mask], -sd_monthly[mask], sd_monthly[mask], color='grey', alpha=0.2, label='uncertainty')
        plt.legend(loc='lower right', ncol=2, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)       
        plt.tick_params(labelsize=fontsize)    
        plt.xlabel('Year', fontsize=fontsize)
        plt.ylabel(r'O-E difference, $^{\circ}$C', fontsize=fontsize)
        plt.title( stationcode, color='white', fontsize=fontsize)           
        fig.tight_layout()
        plt.savefig(figstr, dpi=300)
        plt.close('all')   
    
    if plot_changepoints == True:
        
        figstr = stationcode + '-' + 'changepoints' + '.png'
    
        fig, ax = plt.subplots(figsize=(15,10))
        plt.scatter(x, y, marker='o', fc='navy', ls='-', lw=1, color='blue', alpha=0.5, label='CUSUM (O-E)')
        plt.scatter(x, y_fit, marker='.', fc='maroon', ls='-', lw=1, color='red', alpha=0.5, label='LTR fit')
        plt.scatter(x, y_fit_diff, marker='.', s=6, fc='gold', ls='-', lw=0.5, color='yellow', alpha=1, label=r'$\delta$(LTR)')
        plt.fill_between(x, 
                         np.tile( np.abs(np.nanmean(y_fit_diff)) - 6.0*np.abs(np.nanstd(y_fit_diff)), len(x)), 
                         np.tile( np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)), len(x)), 
                                 color='gold', alpha=0.2, label=r'$\mu \pm 6\sigma$')                                                                                   
        plt.legend(loc='upper right', ncol=2, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)       
        plt.tick_params(labelsize=fontsize)    
        plt.xlabel('Year', fontsize=fontsize)
        plt.ylabel(r'CUSUM, $^{\circ}$C', fontsize=fontsize)
        plt.title( stationcode + ': depth=' + str(depth) + r' : $\rho$=' + str(f'{r[depth-1]:03f}'), color='white', fontsize=fontsize)               
        fig.tight_layout()
        plt.savefig(figstr, dpi=300)
        plt.close('all')   
            
    if plot_adjustments == True:
        
        figstr = stationcode + '-' + 'adjustments' + '.png'
        
        fig, ax = plt.subplots(figsize=(15,10))
        plt.scatter(t_monthly[mask], ts_monthly[mask], marker='o', fc='navy', ls='-', lw=1, color='blue', alpha=0.5, label='O')
        plt.scatter(t_monthly[mask], ex_monthly[mask], marker='o', fc='maroon', ls='-', lw=1, color='red', alpha=0.5, label='E')
        plt.scatter(x, ts_monthly[mask] + y_means, marker='+', ls='-', lw=1, color='skyblue', alpha=1, label='O (adjusted)')
        plt.scatter(x, y_means, marker='.', ls='-', lw=1, color='gold', alpha=1, label='adjustment')
        plt.fill_between(t_monthly[mask], ex_monthly[mask]-sd_monthly[mask], ex_monthly[mask]+sd_monthly[mask], color='grey', alpha=0.2, label='uncertainty')
        plt.legend(loc='lower right', ncol=2, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)       
        plt.tick_params(labelsize=fontsize)    
        plt.xlabel('Year', fontsize=fontsize)
        plt.ylabel(r'Anomaly (from 1961-1990), $^{\circ}$C', fontsize=fontsize)
        plt.title( stationcode + ': depth=' + str(depth) + r' : $\rho$=' + str(f'{r[depth-1]:03f}'), color='white', fontsize=fontsize)               
        fig.tight_layout()
        plt.savefig(figstr, dpi=300)
        plt.close('all')   

    return breakpoints, adjustments
                      
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    parser = OptionParser("usage: %prog [options] stationcode")
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.error("incorrect number of arguments: please enter a stationcode")

    stationcode = args[0]
    
    print('calculating breakpoints and adjustments ...')
    breakpoints, adjustments = calculate_adjustments(stationcode)
    print('BREAKPOINTS: ', breakpoints)
    print('ADJUSTMENTS: ', adjustments)

    # ------------------------
    print('** END')
    
    
