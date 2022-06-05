#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: cru_changepoint_detector_channel_islands.py
#------------------------------------------------------------------------------
# Version 0.2
# 5 June, 2022
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
import netCDF4
from netCDF4 import Dataset, num2date, date2num

# System libraries:
import os

#----------------------------------------------------------------------------------
import filter_cru_dft as cru_filter # CRU DFT filter
import cru_changepoint_detector as cru # CRU changepoint detector
#----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

stationcode = '038940'     # Guernsey

fontsize = 16 
default_color = 'black'
use_smoothing = True
plot_timeseries = True
plot_difference = True
    
#------------------------------------------------------------------------------
# LOAD: CUSUM timeseries from local expectation Kriging (LEK) analysis
#------------------------------------------------------------------------------
         
df_temp = pd.read_pickle('DATA/df_temp_expect_reduced.pkl', compression='bz2')
df_compressed = df_temp[ df_temp['stationcode'] == stationcode ].sort_values(by='year').reset_index(drop=True).dropna()

#['year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
#       'stationcode', 'stationlat', 'stationlon', 'stationelevation',
#       'stationname', 'stationcountry', 'stationfirstyear', 'stationlastyear',
#       'stationsource', 'stationfirstreliable', 'n1', 'n2', 'n3', 'n4', 'n5',
#       'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'e1', 'e2', 'e3', 'e4',
#       'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 's1', 's2', 's3',
#       's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12'],
#      dtype='object')
    
df_yearly = pd.DataFrame({'year': np.arange( 1781, 2022 )}) # 1780-2020 inclusive
df = df_yearly.merge(df_compressed, how='left', on='year')
dt = df.groupby('year').mean().iloc[:,0:12]
dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
dn = dt.copy()
dn.iloc[:,0:] = dn_array
da = (dt - dn).reset_index()
de = (df.groupby('year').mean().iloc[:,43:55]).reset_index()
ds = (df.groupby('year').mean().iloc[:,55:67]).reset_index()        
dt = dt.reset_index()

# TRIM: to start of Pandas datetime range
        
dt = dt[dt.year >= 1678].reset_index(drop=True)
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
    
#------------------------------------------------------------------------------
# LOAD: WeatherRescue Guernsey data ( via Ed Hawkins with thanks )
#------------------------------------------------------------------------------
         
dg = pd.read_table('DATA/Guernsey-Monthly-Mean-Temps-1843-to-present.dat', index_col=0) # WeatherRescue monthly data
baseline = np.nanmean(dg[ (dg.index>=1961) & (dg.index<=1990) ], axis=0) 
ts_guernsey = []    
for i in range(len(dg)):                
    monthly = dg.iloc[i,0:] - baseline
    ts_guernsey = ts_guernsey + monthly.to_list()
ts_guernsey = np.array(ts_guernsey)   

if use_smoothing == True: ts_guernsey = pd.Series(ts_guernsey).rolling(12, center=True).mean()
        
t_guernsey = pd.date_range(start=str(dg.index[0]), periods=len(ts_guernsey), freq='MS')

df_monthly = pd.DataFrame( {'t':t, 'O':a, 'E':e, 'S':s} )
dg_monthly = pd.DataFrame( {'t':t_guernsey, 'G':ts_guernsey} )                          
dg = df_monthly.merge(dg_monthly, how='left', on='t')
dg['O-E'] = dg['O'] - dg['E']
dg['G-E'] = dg['G'] - dg['E']

# COMPUTE: CUSUM
        	
yA = np.nancumsum( d )
yB = np.nancumsum( dg['G-E'].values )
xA = ( np.arange(len(yA)) / len(yA) )
xB = ( np.arange(len(yB)) / len(yB) )
        
#------------------------------------------------------------------------------
# CALL: cru_changepoint_detector
#------------------------------------------------------------------------------

print('BREAKPOINTS: O-E')

y_fitA, y_fit_diffA, y_fit_diff2A, slopesA, breakpointsA, depthA, rA, R2adjA = cru.changepoint_detector(xA, yA)

print('BREAKPOINTS: G-E')

y_fitB, y_fit_diffB, y_fit_diff2B, slopesB, breakpointsB, depthB, rB, R2adjB = cru.changepoint_detector(xB, yB)

# DOCUMENTED CHANGES:
    
breakpointsC = [ 1215, 1344, 1473, 1697, 1999 ]
#1881-04
#1892-01
#1902-10
#1921-06
#1946-08

#==============================================================================
# PLOTS
#==============================================================================

#------------------------------------------------------------------------------
# PLOT: O vs E with LEK uncertainty timeseries
#------------------------------------------------------------------------------

if plot_timeseries == True:

    figstr = 'channel-islands-guernsey' + '-' + 'timeseries.png'       
             
    fig, ax = plt.subplots(figsize=(15,10))

    plt.plot( t, dg.G, marker='o', ls='-', lw=1, color='orange', alpha=1, label='WeatherRescue')
    plt.plot( t, dg.O, marker='o', ls='-', lw=1, color='blue', alpha=0.3, label='GloSAT.p04')
    plt.plot( t, dg.E, marker='o', ls='-', lw=1, color='red', alpha=0.3, label='LEK')
    plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
    ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
    plt.grid(visible=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Temperature anomaly (from 1961-1990), $^{\circ}$C', fontsize=fontsize)
    plt.title( 'Monthly temperatures on Guernsey (' + stationcode + ')', color=default_color, fontsize=fontsize)           
    fig.legend(loc='lower center', ncol=6, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)       
    plt.savefig(figstr, dpi=300)
    plt.close('all')       
    
#------------------------------------------------------------------------------
# PLOT: O-E with LEK uncertainty
#------------------------------------------------------------------------------
    
if plot_difference == True:
	    
    figstr = 'channel-islands-guernsey' + '-' + 'difference.png'   
                 
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(t, dg['O-E'], marker='o', ls='-', lw=1, color='lightblue', alpha=0.7, label='O-E')
    plt.plot(t, dg['G-E'], marker='o', ls='-', lw=1, color='pink', alpha=0.7, label='G-E')
    for i in range(len(breakpointsB)):
        if i==0:
            plt.axvline( t[breakpointsB[i]], ls='-', lw=3, color='red', label='Breakpoint (G-E)')
        else:
            plt.axvline( t[breakpointsB[i]], ls='-', lw=3, color='red')    

    for i in range(len(breakpointsA)):
        if i==0:
            plt.axvline( t[breakpointsA[i]], ls='-', lw=2, color='blue', label='Breakpoint (O-E)')
        else:
            plt.axvline( t[breakpointsA[i]], ls='-', lw=2, color='blue')    

    for i in range(len(breakpointsC)):
        if i==0:
            plt.axvline( t[breakpointsC[i]], ls='dashed', lw=2, color=default_color, label='Documented')
        else:
            plt.axvline( t[breakpointsC[i]], ls='dashed', lw=2, color=default_color)    
    plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
    ylimits = np.array(plt.ylim())
    plt.ylim( -ylimits.max(), ylimits.max() )
    ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
    plt.grid(visible=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Anomaly difference, $^{\circ}$C', fontsize=fontsize)
    fig.legend(loc='lower center', ncol=5, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)  
    plt.title( 'Observations - LEK and breakpoints', fontsize=fontsize)      
    plt.savefig(figstr, dpi=300)
    plt.close('all')    
      
#------------------------------------------------------------------------------
print('** END')

