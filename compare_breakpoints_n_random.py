#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: compare_breakpoinnts_n_random.py
#------------------------------------------------------------------------------
# Version 0.1
# 4 June, 2022
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

# Dataframe libraries:
import numpy as np
import pandas as pd
from datetime import datetime
import netCDF4
from netCDF4 import Dataset, num2date, date2num

# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Stats libraries:
import ruptures
import random

# System libraries:
import os

#----------------------------------------------------------------------------------
import filter_cru_dft as cru_filter # CRU DFT filter
import cru_changepoint_detector as cru # CRU changepoint detector
#----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16 
use_dark_theme = True
use_smoothing = True
 
filename_lek = 'DATA/df_temp_expect_reduced.pkl'
filename_emily = 'DATA/df_breaks.pkl'

min_separation = 120 # months
ndraws = 10

#----------------------------------------------------------------------------
# DARK THEME
#----------------------------------------------------------------------------

if use_dark_theme == True:
    
    matplotlib.rcParams['text.usetex'] = False
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
        
    print('Using Seaborn graphics ... ')
    import seaborn as sns; sns.set()
    
# Calculate current time

now = datetime.now()
currentdy = str(now.day).zfill(2)
currentmn = str(now.month).zfill(2)
currentyr = str(now.year)
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr    

#----------------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------------

def changepoints( dorig, **opts ):
  """
  from: Kevin Cowtan - calc_homogenisation_full.py
   
  Change point detection using PELT (Killick 2012),
  implemented in the ruptures package (Truong 2020)
  
  Parameters:
    dorig (vector of float): original data with no seasonal cycle, e.g.
      difference between obs and local expectation. No missing values.
    opts (dictionary): additional options, including:
      "nbuf": minimum number of months between changepoints
    
  Returns:
    (list of float): list of indices of changepoints
  """  
  min_size = opts["nbuf"]
  penalty_value = 10
  algo = ruptures.KernelCPD( kernel = "linear", min_size = min_size ).fit( dorig )
  #algo = ruptures.Pelt(model="l2", min_size=min_size).fit(dorig)
  result = algo.predict( pen = penalty_value )

  return result[:-1]

def changemissing( dnorm, **opts ):
  """
  from: Kevin Cowtan - calc_homogenisation_full.py

  Change point detection wrapper to allow missing data and return a
  vector of flags
  
  Parameters:
    dorig (vector of float): original data with no seasonal cycle, e.g.
      difference between obs and local expectation. No missing values.
    opts (dictionary): additional options for changepoints function
    
  Returns:
    (vector of unit8): vector of station fragment flags
  """  

  mask = ~np.isnan( dnorm )
  diff = dnorm[ mask ]
  chg = []
  if diff.shape[0] > 2*opts["nbuf"]: chg = changepoints( diff, **opts )
  index = np.arange( dnorm.shape[0] )[ mask ]
  flags = np.full( dnorm.shape, 0, np.uint8 )
  for i in chg:
    flags[ index[i]: ] += 1
    
  return flags

#------------------------------------------------------------------------------
# LOAD: Emily's metadata breaks
#------------------------------------------------------------------------------

df_emily = pd.read_pickle( filename_emily, compression='bz2')
df_emily['stationcode'] = [ str(df_emily['stationcode'].iloc[i]).zfill(6) for i in range(len(df_emily)) ]

#------------------------------------------------------------------------------
# LOAD: CUSUM timeseries from local expectation Kriging (LEK) analysis
#------------------------------------------------------------------------------

df_temp = pd.read_pickle( filename_lek, compression='bz2')
    
#==============================================================================
# SAMPLE: n random stationcodes
#==============================================================================

nstations = df_emily.stationcode.unique().shape[0]
rng = np.random.default_rng(20220604) # seed for reproducibility

allowed = list( np.arange( nstations ) )    
draws = [ random.choice( allowed ) for i in range( ndraws) ]
#draws = [ rng.choice( allowed ) for i in range( ndraws) ]
stationcodes_emily = [ df_emily.stationcode.unique()[draws[i]] for i in range(len(draws)) ]    
    
#==============================================================================
# LOOP: over all stations
#==============================================================================

for k in range(len(draws)):
        
    stationcode = stationcodes_emily[k]
    
    df_compressed = df_temp[ df_temp['stationcode'] == stationcode ].sort_values(by='year').reset_index(drop=True).dropna()        
    df_yearly = pd.DataFrame({'year': np.arange( 1781, 2022 )}) # 1781-2021 inclusive
    df = df_yearly.merge(df_compressed, how='left', on='year')
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,43:55]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,55:67]).reset_index()        
    
    # TRIM: to start of Pandas datetime range
            
    da = da[da.year >= 1678].reset_index(drop=True)
    de = de[de.year >= 1678].reset_index(drop=True)
    ds = ds[ds.year >= 1678].reset_index(drop=True)
    
    # EXTRACT: monthly timeseries
    
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)                   
    
    # SET: monthly and seasonal time vectors
    
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='MS')     
    
    # COMPUTE: 12-m MA
        
    if use_smoothing == True:
        
        a = pd.Series(ts_monthly).rolling(12, center=True).mean().values
        e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
        s = pd.Series(sd_monthly).rolling(12, center=True).mean().values
    
    else:
    
        a = ts_monthly
        e = ex_monthly
        s = sd_monthly
        
    d = a - e # difference
    t = t_monthly
           
    # COMPUTE: CUSUM
            	
    y = np.nancumsum( d )
    x = np.arange(len(y)) / len(y)
    
    #------------------------------------------------------------------------------
    # EXTRACT: Emily's breaks
    #------------------------------------------------------------------------------
    
    emily = df_emily[ df_emily['stationcode'] == stationcode ].year.values
    t_index = np.arange(len(t))
    breakpoints_emily = []
    if np.isfinite(len(emily)).sum() > 0:
        for i in range(len(emily)): 
            breakpoints_emily.append( t_index[t.year==emily[i]][6] ) # 6 --> mid-year=Jun
        
    #------------------------------------------------------------------------------
    # CALL: cru_changepoint_detector
    #------------------------------------------------------------------------------
    
    y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
    
    #------------------------------------------------------------------------------
    # CALL: ruptures
    #------------------------------------------------------------------------------
    
    flags = changemissing( d, nbuf = min_separation )
    
    # FIND: break indices
    
    breakpoints_ruptures = np.where( np.diff( flags ) )[0] + 1  
        
    #------------------------------------------------------------------------------
    # CREATE: breakpoint boolean vectors
    #------------------------------------------------------------------------------
    
    breakpoint_flags = np.array(len(ts_monthly) * [False])
    for j in range(len(breakpoints)): breakpoint_flags[breakpoints[j]] = True        
    
    ruptures_flags = np.array(len(ts_monthly) * [False])
    for j in range(len(breakpoints_ruptures)): ruptures_flags[breakpoints_ruptures[j]] = True        
    
    #==============================================================================
    # PLOTS
    #==============================================================================
    
    if use_dark_theme == True:
        default_color = 'white'
    else:    
        default_color = 'black'    	
    
    #------------------------------------------------------------------------------
    # PLOT: O vs E with LEK uncertainty + breakpoints
    #------------------------------------------------------------------------------

    a_smoothed = pd.Series(ts_monthly).rolling(12, center=True).mean().values
    e_smoothed = pd.Series(ex_monthly).rolling(12, center=True).mean().values
    d_smoothed = a_smoothed - e_smoothed	
		
    figstr = stationcode + '-' + 'compare_breakpoints.png'       
                 
    fig, ax = plt.subplots(figsize=(15,10))
    # plt.fill_between(t, e-s, e+s, color='lightgrey', alpha=0.5, label='uncertainty')
    # plt.scatter(t, a, marker='o', fc='blue', ls='-', lw=1, color='blue', alpha=0.5, label='O')
    # plt.scatter(t, e, marker='o', fc='red', ls='-', lw=1, color='red', alpha=0.5, label='E')         
    plt.scatter(t, d, marker='o', fc='white', ls='-', lw=1, color='grey', alpha=0.2, label='O-E')         
            
    for i in range(len(breakpoints)): 
        if i == 0: 
            plt.axvline( t[ breakpoints[i] ], ls='-', lw=2, color='blue', alpha=1, label='cru')                    
        else:
            plt.axvline( t[ breakpoints[i] ], ls='-', lw=2, color='blue', alpha=1)     
    for i in range(len(breakpoints_ruptures)): 
        if i == 0: 
            plt.axvline( t[ breakpoints_ruptures[i] ], ls='-', lw=3, color='red', alpha=1, label='ruptures')                    
        else:
            plt.axvline( t[ breakpoints_ruptures[i] ], ls='-', lw=3, color='red', alpha=1)                    
    for i in range(len(breakpoints_emily)): 
        if i == 0: 
            plt.axvline( t[ breakpoints_emily[i] ], ls='-', lw=1, color='yellow', alpha=1, label='emily')                    
        else:
            plt.axvline( t[ breakpoints_emily[i] ], ls='-', lw=1, color='yellow', alpha=1)     
                   
    plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
    plt.xlim(t[0],t[-1])
    ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
    plt.grid(visible=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Temperature anomaly (from 1961-1990) difference, $^{\circ}$C', fontsize=fontsize)
    plt.title( stationcode, color=default_color, fontsize=fontsize)           
    fig.legend(loc='lower center', ncol=6, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)       
    plt.savefig(figstr, dpi=300)
    plt.close('all')       
           
#------------------------------------------------------------------------------
print('** END')
