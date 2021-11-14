#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: cru_changepoint_detector_wrapper.py
#------------------------------------------------------------------------------
#
# Version 0.2
# 12 November, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime

#----------------------------------------------------------------------------------
import cru_changepoint_detector as cru # CRU changepoint detector
#----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16 

#stationcode = '037401'     # HadCET
#stationcode = '103810'     # Berlin-Dahlem (breakpoint: 1908)
#stationcode = '685880'     # Durban/Louis Botha (breakpoint: 1939)
#stationcode = '024581'     # Uppsala
#stationcode = '725092'     # Boston City WSO
#stationcode = '062600'     # St Petersberg
#stationcode = '260630'     # De Bilt
#stationcode = '688177'     # Cape Town
stationcode = '619930'     # Pamplemousses

if stationcode == '103810':
    documented_change = 1908
elif stationcode == '685880':
    documented_change = 1939 
else:
    documented_change = np.nan

use_dark_theme = True
use_lek_cusum = False

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

#    mask = np.isfinite(ts_monthly)
    mask = len(ts_monthly) * [True]

    # COMPUTE: 12-m MA
    
    a = pd.Series(ts_monthly).rolling(12, center=True).mean().values
    e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
    s = pd.Series(sd_monthly).rolling(12, center=True).mean().values

    t = t_monthly[mask]
    a = a[mask]
    e = e[mask]
    s = s[mask]

    diff_yearly = a - e
    
    # CALCULATE: CUSUM
        	
    c = np.nancumsum( diff_yearly )
    diff_yearly = diff_yearly

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

#------------------------------------------------------------------------------
# WRITE: breakpoints to CSV
#------------------------------------------------------------------------------

file_breakpoints = stationcode + '-' + 'breakpoints.csv'    

df_breakpoints = pd.DataFrame({'breakpoint':t[breakpoints]})
df_breakpoints.to_csv( file_breakpoints )

#------------------------------------------------------------------------------
# PLOT: CUSUM(O-E) with breakpoints
#------------------------------------------------------------------------------

figstr = stationcode + '-' + 'cusum-curve-breakpoint-selection.png'                

fig, ax = plt.subplots(figsize=(15,10))
plt.plot( t, y, color='blue', ls='-', lw=3, label='CUSUM (O-E)')
plt.plot( t, y_fit, color='red', ls='-', lw=2, label='LTR fit')
if use_dark_theme == True:
    plt.fill_between( t, slopes, 0, color='lightblue', alpha=0.5, label='slope: CUSUM/decade' )    
    default_color = 'white'
else:    
    plt.fill_between( t, slopes, 0, color='lightblue', alpha=0.5, label='slope: CUSUM/decade' )    
    default_color = 'black'
ylimits = plt.ylim()    
if ~np.isnan(documented_change): plt.axvline(x=documented_change_datetime, ls='-', lw=2, color='gold', label='Documented change: ' + str(documented_change) )                   
for i in range(len(t[(y_fit_diff2>0).ravel()])):
    if i==0: plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2, label='LTR boundary') 
    else: plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2) 
for i in range(len(breakpoints)):
    if i==0: plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color, label='Breakpoint')
    else: plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color)    
ax.xaxis.grid(b=None, which='major', color='none', linestyle='-')
ax.yaxis.grid(b=None, which='major', color='none', linestyle='-')
plt.grid(b=None)
plt.tick_params(labelsize=fontsize)    
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'CUSUM, $^{\circ}$C', fontsize=fontsize)
plt.title( stationcode + ': depth=' + str(depth) + r' : $\rho$=' + str( f'{r[depth-1]:03f}' ), color='white', fontsize=fontsize)      
fig.legend(loc='lower right', ncol=2, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)       
fig.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)       
plt.savefig(figstr, dpi=300)
plt.close('all')    

#------------------------------------------------------------------------------
# PLOT: O-E with breakpoints
#------------------------------------------------------------------------------

if use_lek_cusum == False:
    
    figstr = stationcode + '-' + 'cusum-curve-breakpoint-timeseries.png'                

    fig, ax = plt.subplots(figsize=(15,10))
    plt.scatter(t, a, marker='o', fc='maroon', ls='-', lw=1, color='red', alpha=0.5, label='O')
    plt.scatter(t, e, marker='o', fc='navy', ls='-', lw=1, color='blue', alpha=0.5, label='E')
    plt.scatter(t, diff_yearly, marker='+', fc='cyan', ls='-', lw=1, color='lightblue', alpha=1, label='O-E')
    if use_dark_theme == True:
        default_color = 'white'
    else:    
        default_color = 'black'    
    for i in range(len(t[(y_fit_diff2>0).ravel()])):
        if i==0:
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2, label='LTR segment') 
        else:
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2) 
    for i in range(len(breakpoints)):
        if i==0:
            plt.plot( t[0:breakpoints[i]], np.tile( np.nanmean(diff_yearly[0:breakpoints[i]]), breakpoints[i] ), lw=2, color='gold')
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color, label='Breakpoint')
        else:
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color)    
            plt.plot( t[breakpoints[i-1]:breakpoints[i]], np.tile( np.nanmean(diff_yearly[breakpoints[i-1]:breakpoints[i]]), breakpoints[i]-breakpoints[i-1] ), lw=2, color='gold')
        if i==len(breakpoints)-1:
            plt.plot( t[breakpoints[i]:], np.tile( np.nanmean(diff_yearly[breakpoints[i]:]), len(t)-breakpoints[i] ), lw=2, color='gold', label='Segment mean')                
    ax.xaxis.grid(b=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(b=None, which='major', color='none', linestyle='-')
    plt.grid(b=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Anomaly difference, $^{\circ}$C', fontsize=fontsize)
    plt.title( stationcode, color=default_color, fontsize=fontsize)           
    fig.legend(loc='lower right', ncol=2, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)       
    plt.savefig(figstr, dpi=300)
    plt.close('all')                      

#------------------------------------------------------------------------------
print('** END')
