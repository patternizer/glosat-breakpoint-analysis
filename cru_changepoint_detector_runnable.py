#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: cru_changepoint_detector_runnable.py
#------------------------------------------------------------------------------
#
# Version 0.3
# 20 November, 2021
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
from datetime import datetime

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

def calculate_adjustments(stationcode):
    
    #------------------------------------------------------------------------------
    import filter_cru_dft as cru_filter # CRU DFT filter
    import cru_changepoint_detector as cru # CRU changepoint detector
    #------------------------------------------------------------------------------
   
    #-----------------------------------------------------------------------------
    # SETTINGS
    #-----------------------------------------------------------------------------
                   
    #if stationcode == '103810':
    #    documented_change = 1908
    #elif stationcode == '685880':
    #    documented_change = 1939 
    #else:
    #    documented_change = np.nan
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
                               
    #------------------------------------------------------------------------------
    # LOAD: LEK global dataframe
    #------------------------------------------------------------------------------
         
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
    de = (df.groupby('year').mean().iloc[:,43:55]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,55:67]).reset_index()        

    # TRIM: to start of Pandas datetime range
        
    da = da[da.year >= 1678].reset_index(drop=True)
    de = de[de.year >= 1678].reset_index(drop=True)
    ds = ds[ds.year >= 1678].reset_index(drop=True)

    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)                   
    # Solve Y1677-Y2262 Pandas bug with Xarray:        
    # t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')      
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')  
            
    if len(ts_monthly) == 0:    
        print('STATION DATA: not available. Returning ...')
        breakpoints = []
        adjustments = []
    else:
        print('STATION DATA:', ts_monthly)
        
    # COMPUTE: 12-m MA
    
    a = pd.Series(ts_monthly).rolling(12, center=True).mean().values
    e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
    s = pd.Series(sd_monthly).rolling(12, center=True).mean().values

    # APPLY: mask

    mask = np.array(len(ts_monthly) * [True])
    t = t_monthly[mask]
    a = a[mask]
    e = e[mask]
    s = s[mask]

    diff_yearly = a - e
    
    # COMPUTE: CUSUM
        	
    c = np.nancumsum( diff_yearly )
    diff_yearly = diff_yearly

    x = ( np.arange(len(c)) / len(c) )
    y = c

    if np.isnan(documented_change):
        documented_change_datetime = np.nan
    else:        
        documented_change_datetime = pd.to_datetime('01-01-'+str(documented_change),format='%d-%m-%Y')
                    
    # CALL: cru_changepoint_detector

    y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
    
    # CALCULATE: intra-breakpoint fragment means
        
    y_means = []
    adjustments = []
    for j in range(len(breakpoints)+1):                
        if j == 0:              
            y_means = y_means + list( np.tile( -np.nanmean(diff_yearly[0:breakpoints[j]]), breakpoints[j] ) ) 
            adjustment = [ -np.nanmean(diff_yearly[0:breakpoints[j]]) ]
        if (j > 0) & (j<len(breakpoints)):
            y_means = y_means + list( -np.tile( np.nanmean(diff_yearly[breakpoints[j-1]:breakpoints[j]]), breakpoints[j]-breakpoints[j-1] )) 
            adjustment = [ -np.nanmean(diff_yearly[breakpoints[j-1]:breakpoints[j]]) ]
        if (j == len(breakpoints)):              
            y_means = y_means + list( -np.tile( np.nanmean(diff_yearly[breakpoints[-1]:]), len(diff_yearly)-breakpoints[-1] ) ) 
            adjustment = [ -np.nanmean(diff_yearly[breakpoints[-1]:]) ]
        adjustments.append(adjustment)
        
    y_means = np.array( y_means ) 
    adjustments = np.array(adjustments).ravel()

    # STATISTICS    

    rmse = np.sqrt( np.nanmean( diff_yearly**2.0 ) )
    mae = np.nanmean( np.abs( y_means ) )
    breakpoint_flags = np.array(len(ts_monthly) * [False])
    for j in range(len(breakpoints)):
        breakpoint_flags[breakpoints[j]] = True
        
    df = pd.DataFrame( {'time':t, 'breakpoint':breakpoint_flags, 'adjustment':-diff_yearly, 'segment_mean':y_means}, index=np.arange(len(t)) )          

    #------------------------------------------------------------------------------
    # WRITE: breakpoints and segment adjustments to CSV
    #------------------------------------------------------------------------------

    file_breakpoints = stationcode + '-' + 'breakpoints.csv'    
    file_adjustments = stationcode + '-' + 'adjustments.csv'    

    df_breakpoints = pd.DataFrame( {'breakpoint':t[breakpoints]}, index=np.arange(1,len(breakpoints)+1) )
    df_breakpoints.to_csv( file_breakpoints )
    df_adjustments = pd.DataFrame( {'adjustment':adjustments}, index=np.arange(1,len(adjustments)+1) )
    df_adjustments.to_csv( file_adjustments )

    # EXTRACT: seasonal components

    trim_months = len(ex_monthly)%12
    df = pd.DataFrame({'Tg':ex_monthly[:-1-trim_months]}, index=t_monthly[:-1-trim_months])         
    t_years = [ pd.to_datetime( str(df.index.year.unique()[i])+'-01-01') for i in range(len(df.index.year.unique())) ][1:] # years
    DJF = ( df[df.index.month==12]['Tg'].values + df[df.index.month==1]['Tg'].values[1:] + df[df.index.month==2]['Tg'].values[1:] ) / 3
    MAM = ( df[df.index.month==3]['Tg'].values[1:] + df[df.index.month==4]['Tg'].values[1:] + df[df.index.month==5]['Tg'].values[1:] ) / 3
    JJA = ( df[df.index.month==6]['Tg'].values[1:] + df[df.index.month==7]['Tg'].values[1:] + df[df.index.month==8]['Tg'].values[1:] ) / 3
    SON = ( df[df.index.month==9]['Tg'].values[1:] + df[df.index.month==10]['Tg'].values[1:] + df[df.index.month==11]['Tg'].values[1:] ) / 3
    ONDJFM = ( df[df.index.month==10]['Tg'].values[1:] + df[df.index.month==11]['Tg'].values[1:] + df[df.index.month==12]['Tg'].values + df[df.index.month==1]['Tg'].values[1:] + df[df.index.month==2]['Tg'].values[1:] + df[df.index.month==3]['Tg'].values[1:] ) / 6
    AMJJAS = ( df[df.index.month==4]['Tg'].values[1:] + df[df.index.month==5]['Tg'].values[1:] + df[df.index.month==6]['Tg'].values[1:] + df[df.index.month==7]['Tg'].values[1:] + df[df.index.month==8]['Tg'].values[1:] + df[df.index.month==9]['Tg'].values[1:] ) / 6
    df_seasonal = pd.DataFrame({'DJF':DJF, 'MAM':MAM, 'JJA':JJA, 'SON':SON, 'ONDJFM':ONDJFM, 'AMJJAS':AMJJAS}, index = t_years)     
    mask = np.isfinite(df_seasonal)
    df_seasonal_fft = pd.DataFrame(index=df_seasonal.index)
    df_seasonal_fft['DJF'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['DJF'].values[mask['DJF']], nfft)}, index=df_seasonal['DJF'].index[mask['DJF']])
    df_seasonal_fft['MAM'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['MAM'].values[mask['MAM']], nfft)}, index=df_seasonal['MAM'].index[mask['MAM']])
    df_seasonal_fft['JJA'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['JJA'].values[mask['JJA']], nfft)}, index=df_seasonal['JJA'].index[mask['JJA']])
    df_seasonal_fft['SON'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['SON'].values[mask['SON']], nfft)}, index=df_seasonal['SON'].index[mask['SON']])
    df_seasonal_fft['ONDJFM'] = pd.DataFrame({'ONDJFM':smooth_fft(df_seasonal['ONDJFM'].values[mask['ONDJFM']], nfft)}, index=df_seasonal['ONDJFM'].index[mask['ONDJFM']])
    df_seasonal_fft['AMJJAS'] = pd.DataFrame({'AMJJAS':smooth_fft(df_seasonal['AMJJAS'].values[mask['AMJJAS']], nfft)}, index=df_seasonal['AMJJAS'].index[mask['AMJJAS']])
    mask_fft = np.isfinite(df_seasonal_fft)
       
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
        plt.scatter(t, diff_yearly, marker='o', fc='lightblue', ls='-', lw=1, color='lightblue', alpha=0.5, label='O-E')
        for i in range(len(t[(y_fit_diff2>0).ravel()])):
            if i==0:
                plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2, label='LTR boundary') 
            else:
                plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2) 
        for i in range(len(breakpoints)):
            if i==0:
                plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color, label='Breakpoint')
            else:
                plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color)    
        plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
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
            if i==0: plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color, label='Breakpoint')
            else: plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color)    
        plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
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
                plt.plot( t[0:breakpoints[i]], np.tile( -np.nanmean(diff_yearly[0:breakpoints[i]]), breakpoints[i] ), ls='-', lw=3, color='gold')
                plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color, label='Breakpoint')
            else:
                plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color)    
                plt.plot( t[breakpoints[i-1]:breakpoints[i]], np.tile( -np.nanmean(diff_yearly[breakpoints[i-1]:breakpoints[i]]), breakpoints[i]-breakpoints[i-1] ), ls='-', lw=3, color='gold')
            if i==len(breakpoints)-1:
                plt.plot( t[breakpoints[i]:], np.tile( -np.nanmean(diff_yearly[breakpoints[i]:]), len(t)-breakpoints[i] ), ls='-', lw=3, color='gold', label='Adjustment')                
        plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
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
            if i==0: plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color, label='Breakpoint')
            else: plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color)    
        plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
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
        
    return t[breakpoints], adjustments
                      
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
    
    
