#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: cru_changepoint_detector_global_stats.py
#------------------------------------------------------------------------------
#
# Version 0.1
# 8 December, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

# Data array libraries:
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
from netCDF4 import Dataset, num2date, date2num

# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns; sns.set()

# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
#from cartopy.io import shapereader
#import cartopy.feature as cfeature
#from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.cm as cm
#from matplotlib import colors as mcol
#from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#import matplotlib.dates as mdates
#import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
#from matplotlib.collections import PolyCollection
#from mpl_toolkits import mplot3d
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d

# Datetime libraries:
from datetime import datetime

# System libraries:
import os

#----------------------------------------------------------------------------------
import cru_changepoint_detector as cru # CRU changepoint detector
#----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16 
use_dark_theme = False
use_csv = True
 
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

    print('Using Seaborn graphics ...')
    
#    matplotlib.rcParams['text.usetex'] = False
#    rcParams['font.family'] = 'sans-serif'
#    rcParams['font.sans-serif'] = ['Avant Garde', 'Lucida Grande', 'Verdana', 'DejaVu Sans' ]
#    plt.rc('text',color='black')
#    plt.rc('lines',color='black')
#    plt.rc('patch',edgecolor='black')
#    plt.rc('grid',color='lightgray')
#    plt.rc('xtick',color='black')
#    plt.rc('ytick',color='black')
#    plt.rc('axes',labelcolor='black')
#    plt.rc('axes',facecolor='white')    
#    plt.rc('axes',edgecolor='black')
#    plt.rc('figure',facecolor='white')
#    plt.rc('figure',edgecolor='white')
#    plt.rc('savefig',edgecolor='white')
#    plt.rc('savefig',facecolor='white')

# Calculate current time

now = datetime.now()
currentdy = str(now.day).zfill(2)
currentmn = str(now.month).zfill(2)
currentyr = str(now.year)
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr    

#----------------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------------

def merge_fix_cols(df1,df2,var):
    '''
    df1: full time range dataframe (container)
    df2: observation time range
    df_merged: merge of observations into container
    var: 'time' or name of datetime column
    '''
    
    df_merged = pd.merge( df1, df2, how='left', left_on=var, right_on=var)    
#   df_merged = pd.merge( df1, df2, how='left', on=var)    
#   df_merged = df1.merge(df2, how='left', on=var)
    
    for col in df_merged:
        if col.endswith('_y'):
            df_merged.rename(columns = lambda col:col.rstrip('_y'),inplace=True)
        elif col.endswith('_x'):
            to_drop = [col for col in df_merged if col.endswith('_x')]
            df_merged.drop( to_drop, axis=1, inplace=True)
        else:
            pass
    return df_merged
    
def calc_breakpoints( stationcode ):
        
    #------------------------------------------------------------------------------
    # EXTRACT: station data
    #------------------------------------------------------------------------------
    
    df_compressed = df_temp[ df_temp['stationcode'] == stationcode ].sort_values(by='year').reset_index(drop=True).dropna()    
    df = df_yearly.merge(df_compressed, how='left', on='year')
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,43:55]).reset_index()        
    
    # TRIM: to start of Pandas datetime range
            
    da = da[da.year >= 1678].reset_index(drop=True)
    de = de[de.year >= 1678].reset_index(drop=True)
    ds = ds[ds.year >= 1678].reset_index(drop=True)
    
    # EXTRACT: monthly timeseries
    
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)                   
    
    # COMPUTE: 12-m MA
        
    a = pd.Series(ts_monthly).rolling(12, center=True).mean().values
    e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
    s = pd.Series(sd_monthly).rolling(12, center=True).mean().values
    d = a - e # difference
        
    # COMPUTE: CUSUM
            	
    y = np.nancumsum( d )
    x = ( np.arange(len(y)) / len(y) )
    
    #------------------------------------------------------------------------------
    # CALL: cru_changepoint_detector
    #------------------------------------------------------------------------------
    
    y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
    
    if len(breakpoints) > 0:
    
        # CALCULATE: inter-breakpoint segment means
                
        y_means = []
        for j in range(len(breakpoints)+1):                
            if j == 0:              
                if np.isfinite( d[0:breakpoints[j]] ).sum() == 0:
                    y_means = y_means + list( np.tile( np.nan, breakpoints[j] ) ) 
                else:            
                    y_means = y_means + list( np.tile( -np.nanmean( d[0:breakpoints[j]] ), breakpoints[j] ) ) 
            if (j > 0) & (j<len(breakpoints)):
                if np.isfinite( d[breakpoints[j-1]:breakpoints[j]] ).sum() == 0:
                    y_means = y_means + list( np.tile( np.nan, breakpoints[j]-breakpoints[j-1] )) 
                else:
                    y_means = y_means + list( np.tile( -np.nanmean( d[breakpoints[j-1]:breakpoints[j]] ), breakpoints[j]-breakpoints[j-1] )) 
            if (j == len(breakpoints)):              
                if np.isfinite( d[breakpoints[-1]:] ).sum() == 0:        
                    y_means = y_means + list( np.tile( np.nan, len(d)-breakpoints[-1] ) )         
                else:
                    y_means = y_means + list( np.tile( -np.nanmean( d[breakpoints[-1]:] ), len(d)-breakpoints[-1] ) )         
        y_means = np.array( y_means ) 

    else:
        
        y_means = np.zeros( len(t_monthly) )
    
    # STATISTICS
    
    rmse = np.sqrt( np.nanmean( d**2.0 ) )
    mae = np.nanmean( np.abs( y_means ) )
    
    # STORE: breakpoints, adjustments and segment adjustment means
    
    breakpoint_flags = np.array(len(t_monthly) * [False])
    for j in range(len(breakpoints)): breakpoint_flags[breakpoints[j]] = True    
    adjustments = -1.0*d
    adjustment_means = y_means
    
    return breakpoint_flags, adjustments, adjustment_means, rmse, mae

#------------------------------------------------------------------------------
# LOAD: timeseries from local expectation Kriging (LEK) analysis
#------------------------------------------------------------------------------
             
df_temp = pd.read_pickle('DATA/df_temp_expect_reduced.pkl', compression='bz2')
    
# SET: monthly time axis
    
df_yearly = pd.DataFrame({'year': np.arange( 1780, 2021 )}) # 1780-2020 inclusive
t_monthly = pd.date_range(start=str(df_yearly['year'].iloc[0]), end=str(df_yearly['year'].iloc[-1]+1), freq='MS')[0:-1]  
    
# LOAD: vectors of station codes, lat, lon
    
stationcodes = df_temp.stationcode.unique()
stationlats = df_temp.groupby('stationcode').mean()['stationlat'].values
stationlons = df_temp.groupby('stationcode').mean()['stationlon'].values

if use_csv == True:
    
    #------------------------------------------------------------------------------
    # LOAD: saved global stats
    #------------------------------------------------------------------------------
    
    df_stats = pd.read_csv('global_stats.csv', index_col=0)
    df_breakpoints = pd.read_csv('global_breakpoints.csv', index_col=0)
    df_breakpoints_2digit = pd.read_csv('global_breakpoints_2digit.csv', index_col=0)
    
    # RECONSTRUCT: dataframes

    stationrmse = df_stats.rmse.values
    stationmae = df_stats.mae.values
        
else:
                                   
    #------------------------------------------------------------------------------
    # COMPUTE: global stats ( slow ) 
    #------------------------------------------------------------------------------
    
    # COMPUTE: RMSE (O-E) and MAE ( segment means ) for each station
    
    stationrmse = []
    stationmae = []
    for i in range(len(stationcodes)):    
        
        breakpoint_flags, adjustments, adjustment_means, rmse, mae = calc_breakpoints( stationcodes[i] )
        stationrmse.append(rmse)
        stationmae.append(mae)

        if i%100 == 0: print(i,rmse,mae)

    df_stats = pd.DataFrame( {'stationcode':stationcodes, 'rmse':stationrmse, 'mae':stationmae} )
    df_stats.to_csv('global_stats.csv')
        
    # COMPUTE: vector of breakpoint flags for each station

    df_breakpoints = pd.DataFrame( {'time':t_monthly} )
    for i in range(len(stationcodes)):    

        breakpoint_flags, adjustments, adjustment_means, rmse, mae = calc_breakpoints( stationcodes[i] )
        df_breakpoints[ stationcodes[i] ] = breakpoint_flags

    df_breakpoints.to_csv('global_breakpoints.csv')

    # COMPUTE: merge breakpoint flags from all 2-digit station codes and sum ( i.e. by country )
        
    df_breakpoints_2digit = pd.DataFrame( {'time':t_monthly} )
    for i in range(100):
            
        col_list = []
        for j in range(len(df_breakpoints.columns)-1):
            if ( df_breakpoints.columns[j+1][0:2] == str(i).zfill(2) ):
                col_list.append( df_breakpoints.columns[j+1] )    
    #   df_breakpoints_2digit[str(i).zfill(2)] = df_breakpoints[ col_list ].any(axis='columns')
        df_breakpoints_2digit[str(i).zfill(2)] = df_breakpoints[ col_list ].sum(axis='columns')
    
        if i%100 == 0: print(i)
    
    df_breakpoints_2digit.to_csv('global_breakpoints_2digit.csv')

#==============================================================================
# PLOTS
#==============================================================================

if use_dark_theme == True:
    default_color = 'white'
else:    
    default_color = 'black'    	

cmap_r = 'nipy_spectral_r'
cmap = 'nipy_spectral'
#cmap = 'tab20c_r'

# PLOT: breakpoint histogram by station code - binary flag

df_breakpoints_2digit = pd.read_csv('global_breakpoints_2digit.csv', index_col=0)

#df_breakpoints_2digit.index = pd.to_datetime(df_breakpoints_2digit.time).dt.date
df_breakpoints_2digit.index = pd.to_datetime(df_breakpoints_2digit.time)
df_breakpoints_2digit = df_breakpoints_2digit[ df_breakpoints_2digit.columns.drop('time') ]
df_breakpoints_2digit = df_breakpoints_2digit.resample('1AS').sum()
#df_breakpoints_2digit['time'] = pd.to_datetime(df_breakpoints_2digit.time).dt.year
df_breakpoints_2digit.index = df_breakpoints_2digit.index.year

figstr = 'global-breakpoints-per-2-digit-stationcode-binary-flags.png'
titlestr = 'Accumulated breakpoints per 2-digit station code (00-99)'
            
fig, ax = plt.subplots(figsize=(15,10))
X = np.arange(100)
Y = df_breakpoints_2digit.index
Z = df_breakpoints_2digit
plt.pcolormesh(X, Y, Z, edgecolors='w', linewidth=0.1, vmin=0, vmax=1, cmap=cmap_r)
cb = plt.colorbar(shrink=0.7, extend='both')    
cb.set_label('Flag', labelpad=10, fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)
#g = sns.heatmap(df_breakpoints_2digit, vmin=0, vmax=1, cmap='binary',
#                cbar_kws={'drawedges': False, 'shrink':0.7, 'extend':'both', 'label':'binary flag'})
xtick_spacing = 5
ytick_spacing = 10
ax.xaxis.set_major_locator(mticker.MultipleLocator(xtick_spacing))
ax.yaxis.set_major_locator(mticker.MultipleLocator(ytick_spacing))
ax.set_xlabel('2-digit station code', fontsize=fontsize)
ax.set_ylabel('Year', fontsize=fontsize)
ax.set_title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: breakpoint histogram by station code - accumulated station count
    
figstr = 'global-breakpoints-per-2-digit-stationcode-n-stations.png'
titlestr = 'Accumulated breakpoints per 2-digit station code (00-99)'
            
fig, ax = plt.subplots(figsize=(15,10))
X = np.arange(100)
Y = df_breakpoints_2digit.index
Z = df_breakpoints_2digit
#plt.pcolormesh(X, Y, Z, edgecolors='w', linewidth=0.1, vmin=0, vmax=50, cmap='tab20c_r')
plt.pcolormesh(X, Y, Z, edgecolors='w', linewidth=0.1, vmin=0, vmax=50, cmap=cmap_r)
cb = plt.colorbar(shrink=0.7, extend='both')    
cb.set_label('N (stations)', labelpad=10, fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)
#g = sns.heatmap(df_breakpoints_2digit, vmin=0, vmax=50, cmap='tab20c_r', 
#                cbar_kws={'drawedges': False, 'shrink':0.7, 'extend':'both', 'label':'number of stations'})
xtick_spacing = 5
ytick_spacing = 10
ax.xaxis.set_major_locator(mticker.MultipleLocator(xtick_spacing))
ax.yaxis.set_major_locator(mticker.MultipleLocator(ytick_spacing))
ax.set_xlabel('2-digit station code', fontsize=fontsize)
ax.set_ylabel('Year', fontsize=fontsize)
ax.set_title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: global map of RMSE

dg = pd.DataFrame({'lon':stationlons, 'lat':stationlats, 'rmse':stationrmse})
v = dg['rmse'].values
#v[v<1e-9] = np.nan
x, y = np.meshgrid(dg['lon'], dg['lat'])    
        
figstr = 'global-stats-rmse.png'
titlestr = 'RMSE (O-E)'
colorbarstr = r'RMSE (O-E), $^{\circ}C$'

fig  = plt.figure(figsize=(15,10))
p = ccrs.PlateCarree(central_longitude=0)
ax = plt.axes(projection=p)
ax.set_global()
ax.set_extent([-180, 180, -90, 90], crs=p)    
gl = ax.gridlines(crs=p, draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='-')
gl.top_labels = False
gl.right_labels = False
gl.xlines = True
gl.ylines = True
gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}              
plt.scatter(x=dg['lon'], y=dg['lat'], c=dg['rmse'], marker='s', s=10, alpha=0.5, transform=ccrs.PlateCarree(), cmap=cmap)  
ax.stock_img()
cb = plt.colorbar(shrink=0.7, extend='both')    
cb.set_label(colorbarstr, labelpad=10, fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close('all')
    
# PLOT: global map of MAE

dg = pd.DataFrame({'lon':stationlons, 'lat':stationlats, 'mae':stationmae})
v = dg['mae'].values
#v[v<=1e-9] = np.nan
x, y = np.meshgrid(dg['lon'], dg['lat'])    
        
figstr = 'global-stats-mae.png'
titlestr = 'MAE (O-E)'
colorbarstr = r'MAE (O-E), $^{\circ}C$'

fig  = plt.figure(figsize=(15,10))
p = ccrs.PlateCarree(central_longitude=0)
ax = plt.axes(projection=p)
ax.set_global()
ax.set_extent([-180, 180, -90, 90], crs=p)    
gl = ax.gridlines(crs=p, draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='-')
gl.top_labels = False
gl.right_labels = False
gl.xlines = True
gl.ylines = True
gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}              
plt.scatter(x=dg['lon'], y=dg['lat'], c=dg['mae'], marker='s', s=10, alpha=0.5, transform=ccrs.PlateCarree(), cmap=cmap)  
ax.stock_img()
cb = plt.colorbar(shrink=0.7, extend='both')    
cb.set_label(colorbarstr, labelpad=10, fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close('all')
    
#------------------------------------------------------------------------------
print('** END')
