#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: lek-climatological-sd.py
#------------------------------------------------------------------------------
# Version 0.1
# 31 March, 2022
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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.cm as cm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.ticker as mticker

# Datetime libraries:
from datetime import datetime

# System libraries:
import os

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16 
use_dark_theme = False
use_csv = True

filename_lek = 'DATA/df_temp_expect_reduced.pkl'             
  
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

# Calculate current time

now = datetime.now()
currentdy = str(now.day).zfill(2)
currentmn = str(now.month).zfill(2)
currentyr = str(now.year)
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr    

#----------------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------------

def calc_sd( stationcode ):
        
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
    de = (df.groupby('year').mean().iloc[:,43:55]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,55:67]).reset_index()        
    
    # TRIM: to start of Pandas datetime range
            
    da = da[da.year >= 1941].reset_index(drop=True)
    de = de[de.year >= 1941].reset_index(drop=True)
    ds = ds[ds.year >= 1941].reset_index(drop=True)

    da = da[da.year < 1991].reset_index(drop=True)
    de = de[de.year < 1991].reset_index(drop=True)
    ds = ds[ds.year < 1991].reset_index(drop=True)
    
    # EXTRACT: monthly timeseries
    
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)                   
    
    # COMPUTE: 12-m MA
        
    e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
            
    # STATISTICS
    
    sd = np.nanstd( e )
    
    return sd

#------------------------------------------------------------------------------
# LOAD: timeseries from local expectation Kriging (LEK) analysis
#------------------------------------------------------------------------------
             
df_temp = pd.read_pickle( filename_lek, compression='bz2' )
    
# SET: monthly time axis
    
df_yearly = pd.DataFrame({'year': np.arange( 1941, 1991 )}) # 1941-1990 inclusive
t_monthly = pd.date_range(start=str(df_yearly['year'].iloc[0]), end=str(df_yearly['year'].iloc[-1]+1), freq='MS')[0:-1]  
    
# LOAD: vectors of station codes, lat, lon
    
stationcodes = df_temp.stationcode.unique()
stationlats = df_temp.groupby('stationcode').mean()['stationlat'].values
stationlons = df_temp.groupby('stationcode').mean()['stationlon'].values
                                   
# COMPUTE: climatological SD
    
stationsd = []
for i in range(len(stationcodes)):    
        
    sd = calc_sd( stationcodes[i] )
    stationsd.append(sd)

    if i%100 == 0: print(i,sd)
    print(i,sd)

df_stats = pd.DataFrame( {'stationcode':stationcodes, 'sd':stationsd} )        
stationsd = df_stats.sd.values

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

# PLOT: global map of SD

dg = pd.DataFrame({'lon':stationlons, 'lat':stationlats, 'sd':stationsd})
v = dg['sd'].values
x, y = np.meshgrid(dg['lon'], dg['lat'])    
        
figstr = 'global-stats-sd.png'
titlestr = 'Climatological SD (O-E)'
colorbarstr = r'SD, $^{\circ}C$'

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
plt.scatter(x=dg['lon'], y=dg['lat'], c=dg['sd'], marker='s', s=10, alpha=0.5, transform=ccrs.PlateCarree(), cmap=cmap)  
ax.stock_img()
cb = plt.colorbar(shrink=0.7, extend='both')    
cb.set_label(colorbarstr, labelpad=10, fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close('all')
    
#------------------------------------------------------------------------------
print('** END')
