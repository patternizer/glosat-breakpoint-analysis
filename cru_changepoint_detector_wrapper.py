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

#stationcode = '037401'     # HadCET
#stationcode = '103810'     # Berlin-Dahlem (breakpoint: 1908)
#stationcode = '685880'     # Durban/Louis Botha (breakpoint: 1939)
#stationcode = '024581'     # Uppsala
#stationcode = '725092'     # Boston City WSO
#stationcode = '062600'     # St Petersberg
#stationcode = '260630'     # De Bilt
#stationcode = '688177'     # Cape Town
#stationcode = '619930'     # Pamplemousses
stationcode = '038940'     # Guernsey

#if stationcode == '103810':
#    documented_change = 1908
#elif stationcode == '685880':
#    documented_change = 1939 
#else:
#    documented_change = np.nan
documented_change = np.nan

fontsize = 16 
use_dark_theme = False
use_smoothing = True

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

def smooth_fft(x, span):  
    
    y_lo, y_hi, zvarlo, zvarhi, fc, pctl = cru_filter.cru_filter_dft(x, span)    
    x_filtered = y_lo

    return x_filtered

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
    
df_yearly = pd.DataFrame({'year': np.arange( 1780, 2021 )}) # 1780-2020 inclusive
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

# SET: monthly and seasonal time vectors

t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='MS')     
t_seasonal = [ pd.to_datetime( str(da['year'].iloc[i+1])+'-01-01') for i in range(2020-1780) ] # Timestamp('YYYY-01-01 00:00:00')]

# EXTRACT: seasonal components ( D from first year only --> N-1 seasonal estimates )

trim_months = len(ex_monthly)%12
df = pd.DataFrame({'Tg':ex_monthly[:-1-trim_months]}, index=t_monthly[:-1-trim_months])         
DJF = ( df[df.index.month==12]['Tg'].values + df[df.index.month==1]['Tg'].values[1:] + df[df.index.month==2]['Tg'].values[1:] ) / 3
MAM = ( df[df.index.month==3]['Tg'].values[1:] + df[df.index.month==4]['Tg'].values[1:] + df[df.index.month==5]['Tg'].values[1:] ) / 3
JJA = ( df[df.index.month==6]['Tg'].values[1:] + df[df.index.month==7]['Tg'].values[1:] + df[df.index.month==8]['Tg'].values[1:] ) / 3
SON = ( df[df.index.month==9]['Tg'].values[1:] + df[df.index.month==10]['Tg'].values[1:] + df[df.index.month==11]['Tg'].values[1:] ) / 3
ONDJFM = ( df[df.index.month==10]['Tg'].values[1:] + df[df.index.month==11]['Tg'].values[1:] + df[df.index.month==12]['Tg'].values + df[df.index.month==1]['Tg'].values[1:] + df[df.index.month==2]['Tg'].values[1:] + df[df.index.month==3]['Tg'].values[1:] ) / 6
AMJJAS = ( df[df.index.month==4]['Tg'].values[1:] + df[df.index.month==5]['Tg'].values[1:] + df[df.index.month==6]['Tg'].values[1:] + df[df.index.month==7]['Tg'].values[1:] + df[df.index.month==8]['Tg'].values[1:] + df[df.index.month==9]['Tg'].values[1:] ) / 6
df_seasonal = pd.DataFrame({'DJF':DJF, 'MAM':MAM, 'JJA':JJA, 'SON':SON, 'ONDJFM':ONDJFM, 'AMJJAS':AMJJAS}, index = t_seasonal)     
mask = np.isfinite(df_seasonal)
df_seasonal_fft = pd.DataFrame(index=df_seasonal.index)
df_seasonal_fft['DJF'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['DJF'].values[mask['DJF']], nfft)}, index=df_seasonal['DJF'].index[mask['DJF']])
df_seasonal_fft['MAM'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['MAM'].values[mask['MAM']], nfft)}, index=df_seasonal['MAM'].index[mask['MAM']])
df_seasonal_fft['JJA'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['JJA'].values[mask['JJA']], nfft)}, index=df_seasonal['JJA'].index[mask['JJA']])
df_seasonal_fft['SON'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['SON'].values[mask['SON']], nfft)}, index=df_seasonal['SON'].index[mask['SON']])
df_seasonal_fft['ONDJFM'] = pd.DataFrame({'ONDJFM':smooth_fft(df_seasonal['ONDJFM'].values[mask['ONDJFM']], nfft)}, index=df_seasonal['ONDJFM'].index[mask['ONDJFM']])
df_seasonal_fft['AMJJAS'] = pd.DataFrame({'AMJJAS':smooth_fft(df_seasonal['AMJJAS'].values[mask['AMJJAS']], nfft)}, index=df_seasonal['AMJJAS'].index[mask['AMJJAS']])
       
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
x = ( np.arange(len(y)) / len(y) )

if np.isnan(documented_change):
    documented_change_datetime = np.nan
else:        
    documented_change_datetime = pd.to_datetime('01-01-'+str(documented_change),format='%d-%m-%Y')

#------------------------------------------------------------------------------
# CALL: cru_changepoint_detector
#------------------------------------------------------------------------------

y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)

# CALCULATE: inter-breakpoint segment means

if len(breakpoints) > 0:
    
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

#------------------------------------------------------------------------------
# EXPORT: breakpoints and segment adjustments to CSV
#------------------------------------------------------------------------------

breakpoint_flags = np.array(len(ts_monthly) * [False])
for j in range(len(breakpoints)): breakpoint_flags[breakpoints[j]] = True        

file_breakpoints = stationcode + '-' + 'breakpoints.csv'    
df_breakpoints = pd.DataFrame( {'time':t, 'breakpoint':breakpoint_flags, 'adjustment':-1.0*d, 'segment_mean':y_means}, index=np.arange(len(t)) )          
df_breakpoints.to_csv( file_breakpoints )
print(t[breakpoints])

#------------------------------------------------------------------------------
# EXPORT: data to netCDF-4
#------------------------------------------------------------------------------
            
# OPEN: netCDF file for writing

nc_filename = stationcode + '-' + 'breakpoints.nc'  
if os.path.exists(nc_filename): os.remove(nc_filename)
ncout = Dataset( nc_filename, 'w', format='NETCDF4')
    
# ADD: // global attributes
    
ncout.title = 'GloSAT station breakpoints and adjustments'
ncout.source = 'GloSAT.p03'
ncout.version = 'GloSAT.p03-lek'
ncout.Conventions = 'CF-1.7'
ncout.reference = 'Osborn, T. J., P. D. Jones, D. H. Lister, C. P. Morice, I. R. Simpson, J. P. Winn, E. Hogan and I. C. Harris (2020), Land surface air temperature variations across the globe updated to 2019: the CRUTEM5 data set, Journal of Geophysical Research: Atmospheres, 126, e2019JD032352. https://doi.org/10.1029/2019JD032352'
ncout.institution = 'Climatic Research Unit, University of East Anglia / Met Office Hadley Centre / University of York'
ncout.licence = 'GloSAT is licensed under the Open Government Licence v3.0 except where otherwise stated. To view this licence, visit https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3'
ncout.history = 'File generated on {} (UTC) by {}'.format(datetime.utcnow().strftime('%c'), os.path.basename(__file__))            
ncout.stationid = df_compressed.stationcode.unique()[0]
ncout.stationlat = np.round( float(df_compressed.stationlat.unique()[0]) ,2)
ncout.stationlon = np.round( float(df_compressed.stationlon.unique()[0]), 2)
ncout.stationelevation = np.round( float(df_compressed.stationelevation.unique()[0]), 2)
ncout.stationname = df_compressed.stationname.unique()[0]
ncout.stationcountry = df_compressed.stationcountry.unique()[0]
ncout.stationfirstyear = int(df_compressed.stationfirstyear.unique()[0])
ncout.stationlastyear = int(df_compressed.stationlastyear.unique()[0])
ncout.stationsource = int(df_compressed.stationsource.unique()[0])
ncout.stationfirstreliable = int(df_compressed.stationfirstreliable.unique()[0])
ncout.stationrmse = np.round( float(rmse) ,6)
ncout.stationmae = np.round( float(mae), 6)

# CREATE: dimensions

ncout.createDimension( 'time', len(df_breakpoints.time) )

# SAVE: data to variables
    
# datatype specifiers include: 
# 'f4' (32-bit floating point), 
# 'f8' (64-bit floating point), 
# 'i4' (32-bit signed integer), 
# 'i2' (16-bit signed integer), 
# 'i8' (64-bit signed integer), 
# 'i1' (8-bit signed integer), 
# 'u1' (8-bit unsigned integer), 
# 'u2' (16-bit unsigned integer), 
# 'u4' (32-bit unsigned integer), 
# 'u8' (64-bit unsigned integer), 
# 'S1' (single-character string)
        
ncout_time = ncout.createVariable('time', 'i4', ('time',))
units = 'months since 1850-01-01 00:00:00'
ncout_time.setncattr('unit',units)
ncout_time[:] = [ date2num(df_breakpoints.time[i], units, calendar='360_day') for i in range(len(df_breakpoints.time)) ]
# calendar: 'standard’, ‘gregorian’, ‘proleptic_gregorian’ ‘noleap’, ‘365_day’, ‘360_day’, ‘julian’, ‘all_leap’, ‘366_day’
    
ncout_breakpoint = ncout.createVariable('breakpoint_flag', 'f4', ('time',))
ncout_breakpoint.units = '1'
ncout_breakpoint.standard_name = 'breakpoint_flag'
ncout_breakpoint.long_name = 'breakpoint_flag_boolean'
ncout_breakpoint.cell_methods = 'time: mean (interval: 1 month)'
ncout_breakpoint.fill_value = -1.e+30
ncout_breakpoint[:] = df_breakpoints['breakpoint'].values

ncout_adjustment = ncout.createVariable('adjustment', 'f4', ('time',))
ncout_adjustment.units = '1'
ncout_adjustment.standard_name = 'adjustment'
ncout_adjustment.long_name = 'qdjustment_degC'
ncout_adjustment.cell_methods = 'time: mean (interval: 1 month)'
ncout_adjustment.fill_value = -1.e+30
ncout_adjustment[:] = df_breakpoints['adjustment'].values

ncout_adjustment_mean = ncout.createVariable('adjustment_mean', 'f4', ('time',))
ncout_adjustment_mean.units = '1'
ncout_adjustment_mean.standard_name = 'adjustment_mean'
ncout_adjustment_mean.long_name = 'qdjustment_segment_mean_degC'
ncout_adjustment_mean.cell_methods = 'time: mean (interval: 1 month)'
ncout_adjustment_mean.fill_value = -1.e+30
ncout_adjustment_mean[:] = df_breakpoints['segment_mean'].values

# CLOSE: netCDF file

ncout.close()

#==============================================================================
# PLOTS
#==============================================================================

if use_dark_theme == True:
    default_color = 'white'
else:    
    default_color = 'black'    	

#------------------------------------------------------------------------------
# PLOT: O vs E with LEK uncertainty timeseries
#------------------------------------------------------------------------------

if plot_timeseries == True:

    figstr = stationcode + '-' + 'timeseries.png'       
             
    fig, ax = plt.subplots(figsize=(15,10))
    plt.fill_between(t, e-s, e+s, color='lightgrey', alpha=0.5, label='uncertainty')
    plt.scatter(t, a, marker='o', fc='blue', ls='-', lw=1, color='blue', alpha=0.5, label='O')
    plt.scatter(t, e, marker='o', fc='red', ls='-', lw=1, color='red', alpha=0.5, label='E')         
    plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
    plt.xlim(t[0],t[-1])
    ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
    plt.grid(visible=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Temperature anomaly (from 1961-1990), $^{\circ}$C', fontsize=fontsize)
    plt.title( stationcode, color=default_color, fontsize=fontsize)           
    fig.legend(loc='lower center', ncol=6, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)       
    plt.savefig(figstr, dpi=300)
    plt.close('all')       
    
#------------------------------------------------------------------------------
# PLOT: O-E with LEK uncertainty
#------------------------------------------------------------------------------
    
if plot_difference == True:
	    
    figstr = stationcode + '-' + 'difference.png'   
                 
    fig, ax = plt.subplots(figsize=(15,10))
    plt.fill_between(t, -s, s, color='lightgrey', alpha=0.5, label='uncertainty')
    plt.scatter(t, d, marker='o', fc='lightblue', ls='-', lw=1, color='lightblue', alpha=0.5, label='O-E')
    for i in range(len(t[(y_fit_diff2>0).ravel()])):
        if i==0:
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2, label='LTR boundary') 
        else:
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2) 
    for i in range(len(breakpoints)):
        if i==0:
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color=default_color, label='Breakpoint')
        else:
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color=default_color)    
    plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
    plt.xlim(t[0],t[-1])    
    ylimits = np.array(plt.ylim())
    plt.ylim( -ylimits.max(), ylimits.max() )
    ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
    plt.grid(visible=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Anomaly difference, $^{\circ}$C', fontsize=fontsize)
    fig.legend(loc='lower center', ncol=4, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)  
    plt.title( stationcode + ': depth=' + str(depth) + r' ( BEST ): $\rho$=' + str( f'{r[depth-1]:03f}' ), color=default_color, fontsize=fontsize)      
    plt.savefig(figstr, dpi=300)
    plt.close('all')    
                
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
        if i==0: plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color=default_color, label='Breakpoint')
        else: plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color=default_color)    
    plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
    plt.xlim(t[0],t[-1])
    ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
    plt.grid(visible=None)
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
	    
    figstr = stationcode + '-' + 'adjustments.png'                

    fig, ax = plt.subplots(figsize=(15,10))
    plt.scatter(t, a, marker='o', fc='blue', ls='-', lw=1, color='blue', alpha=0.5, label='O')
    plt.scatter(t, a + y_means, marker='o', fc='lightblue', ls='-', lw=1, color='lightblue', alpha=0.5, label='O (adjusted)')
    plt.scatter(t, e, marker='o', fc='red', ls='-', lw=1, color='red', alpha=0.5, label='E')
    for i in range(len(t[(y_fit_diff2>0).ravel()])):
        if i==0:
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2, label='LTR boundary') 
        else:
            plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2) 
    for i in range(len(breakpoints)):
        if i==0:
            if np.isfinite( d[0:breakpoints[i]] ).sum() > 0:
                plt.plot( t[0:breakpoints[i]], np.tile( -np.nanmean( d[0:breakpoints[i]] ), breakpoints[i] ), ls='-', lw=3, color='gold')
            plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color=default_color, label='Breakpoint')
        else:
            if np.isfinite( d[breakpoints[i-1]:breakpoints[i]] ).sum() > 0:
                plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color=default_color)    
            plt.plot( t[breakpoints[i-1]:breakpoints[i]], np.tile( -np.nanmean( d[breakpoints[i-1]:breakpoints[i]] ), breakpoints[i]-breakpoints[i-1] ), ls='-', lw=3, color='gold')
        if i==len(breakpoints)-1:
            if np.isfinite( d[breakpoints[i]:] ).sum() > 0:
                plt.plot( t[breakpoints[i]:], np.tile( -np.nanmean( d[breakpoints[i]:] ), len(t)-breakpoints[i] ), ls='-', lw=3, color='gold', label='Adjustment')                
    plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)    
    plt.xlim(t[0],t[-1])                
    ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
    plt.grid(visible=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Temperature anomaly (from 1961-1990), $^{\circ}$C', fontsize=fontsize)
    plt.title( stationcode + ': depth=' + str(depth) + r' ( BEST ): $\rho$=' + str( f'{r[depth-1]:03f}' ), color=default_color, fontsize=fontsize)      
    fig.legend(loc='lower center', ncol=6, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)       
    plt.savefig(figstr, dpi=300)
    plt.close('all')                      

#------------------------------------------------------------------------------
# PLOT: LEK seasonal decadal mean
#------------------------------------------------------------------------------

if plot_seasonal == True:

    figstr = stationcode + '-' + 'seasonal.png'       
             
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(df_seasonal_fft.index, df_seasonal_fft['ONDJFM'], marker='s', ls='-', lw=1, color='blue', alpha=0.5, label='ONDJFM')
    plt.plot(df_seasonal_fft.index, df_seasonal_fft['AMJJAS'], marker='s', ls='-', lw=1, color='red', alpha=0.5, label='AMJJAS')         
    ylimits = plt.ylim()    
    if ~np.isnan(documented_change): plt.axvline(x=documented_change_datetime, ls='-', lw=2, color='gold', label='Documented change: ' + str(documented_change) )                   
    for i in range(len(t[(y_fit_diff2>0).ravel()])):
        if i==0: plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2, label='LTR boundary') 
        else: plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2) 
    for i in range(len(breakpoints)):
        if i==0: plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color=default_color, label='Breakpoint')
        else: plt.axvline( t[breakpoints[i]], ls='dashed', lw=3, color=default_color)    
    plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)      
    plt.xlim(t[0],t[-1])              
    ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
    plt.grid(visible=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Temperature anomaly (from 1961-1990), $^{\circ}$C', fontsize=fontsize)
    plt.title( stationcode, color=default_color, fontsize=fontsize)           
    fig.legend(loc='lower center', ncol=6, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)       
    plt.savefig(figstr, dpi=300)
    plt.close('all')       
    
#------------------------------------------------------------------------------
print('** END')
