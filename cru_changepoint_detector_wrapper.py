#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: cru_changepoint_detector_wrapper.py
#------------------------------------------------------------------------------
#
# Version 0.1
# 24 August, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------
import cru_changepoint_detector as cru # CRU changepoint detector
#----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16
#stationcode = '037401'     # HadCET
#stationcode = '103810'     # Berlin-Dahlem (breakpoint: 1908)
stationcode = '685880'     # Durban/Louis Botha (breakpoint: 1939)
#documented_change = np.nan
#documented_change = 1908
documented_change = 1939 
file_cusum = 'DATA/cusum_' + stationcode + '_obs.csv'
file_breakpoints = stationcode + '-' + 'breakpoints.csv'    
figstr = stationcode + '-' + 'cusum-curve-linear-tree.png' 
    
#------------------------------------------------------------------------------
# LOAD: CUSUM timeseries from local expectation Kriging (LEK) analysis
#------------------------------------------------------------------------------
         
df = pd.read_csv( file_cusum, index_col=0 )
x = df.index.values
y = df.cu.values
mask = np.isfinite(y)
	
#------------------------------------------------------------------------------
# CALL: cru_changepoint_detector
#------------------------------------------------------------------------------

y_fit, y_fit_diff, breakpoints, depth, r, r2adj = cru.changepoint_detector(x[mask], y[mask])
df_breakpoints = pd.DataFrame({'breakpoint':breakpoints})
df_breakpoints.to_csv( file_breakpoints )

#------------------------------------------------------------------------------
# PLOT: breakpoints as a function of linear tree depth
#------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(15,10))
plt.axvline(x=documented_change, ls='--', lw=2, color='black', label='Documented change: ' + str(documented_change) )                   
plt.scatter(x[mask], y[mask], s=3, c='blue', zorder=3, label='CUSUM')
plt.scatter(x[mask], y_fit, s=3, c='red', zorder=4, label='LTR')
ylimits = plt.ylim()    
plt.fill_between(x[mask], 
     np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)), 
     np.abs(np.nanmean(y_fit_diff)) - 6.0*np.abs(np.nanstd(y_fit_diff)), 
     ls='-', lw=1, color='teal', alpha=0.1, zorder=1)
plt.plot(x[mask], [np.nan] + list(np.diff(y_fit)), ls='-', lw=2, color='teal', zorder=2,  label=r'$\delta$LTR: ' + r'$\mu\pm6\sigma$')    
for j in range(len(breakpoints)):        
    if (j%2 == 0) & (j<len(breakpoints)-1):
        plt.fill_betweenx(ylimits, breakpoints[j], breakpoints[j+1], facecolor='lightgrey', alpha=0.5, zorder=0)
    elif (j%2 != 0) & (j<len(breakpoints)-1):        
        plt.fill_betweenx(ylimits, breakpoints[j], breakpoints[j+1], facecolor='grey', alpha=0.5, zorder=0)         
    if j == 0:              
        plt.fill_betweenx(ylimits, x[mask][0], breakpoints[j], facecolor='grey', alpha=0.5, zorder=0)         
    if (j == len(breakpoints)-1) & (j%2 == 0):              
        plt.fill_betweenx(ylimits, breakpoints[j], x[mask][-1], facecolor='lightgrey', alpha=0.5, zorder=0)         
    if (j == len(breakpoints)-1) & (j%2 != 0):              
        plt.fill_betweenx(ylimits, breakpoints[j], x[mask][-1], facecolor='grey', alpha=0.5, zorder=0)      
plt.legend(loc='lower right', ncol=2, markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)       
plt.tick_params(labelsize=fontsize)    
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel('CUSUM', fontsize=fontsize)
plt.title( stationcode + ': depth=' + str(depth) + r' : $\rho$=' + str(np.round(r[depth-1],3)), color='black', fontsize=fontsize)           
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')    
                  
#------------------------------------------------------------------------------
print('** END')
