#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: linear_tree_regression.py
#------------------------------------------------------------------------------
#
# Version 0.3
# 3 November, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
from functools import reduce
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
#import seaborn as sns; sns.set()
from itertools import product
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import *
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from lineartree import LinearTreeClassifier, LinearTreeRegressor
#import basis_expansions
#from basis_expansions import LinearSpline
#from basis_expansions import (Binner,
#                              GaussianKernel,
#                              Polynomial, 
#                              LinearSpline, 
#                              CubicSpline,
#                              NaturalCubicSpline)
#from dftransformers import ColumnSelector, FeatureUnion, Intercept, MapFeature
#from simulation import (run_simulation_expreiment, 
#                        plot_simulation_expreiment, 
#                        make_random_train_test,
#                        run_residual_simulation)

#-----------------------------------------------------------------------------
# ML Linear Tree Regression
#-----------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=4,
                       n_informative=2, n_targets=1,
                       random_state=0, shuffle=False)
regr = LinearTreeRegressor(base_estimator=LinearRegression())
regr.fit(X, y)

#-----------------------------------------------------------------------------
# ML Linear Tree Classification
#-----------------------------------------------------------------------------

from sklearn.linear_model import RidgeClassifier
from lineartree import LinearTreeClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = LinearTreeClassifier(base_estimator=RidgeClassifier())
clf.fit(X, y)

#----------------------------------------------------------------------------------
import filter_cru_dft as cru_filter # CRU DFT filter
#import cru_changepoint_detector as cru # CRU changepoint detector
#----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

max_depth = 9 # in range [1,20]
min_separation = 60 # 5 yr
min_slope_change = 3 * (120/min_separation) 	# units: CUSUM / decade    
min_correlation_change = 0.001 

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

plot_ltr_correlation = True
plot_ltr_loop = True

nsmooth = 12                  # 1yr MA monthly
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

    matplotlib.rcParams['text.usetex'] = False
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

#-----------------------------------------------------------------------------
# METHODS
#-----------------------------------------------------------------------------

def factors(n):    
    '''
    https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
    '''
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def rc_subplots(n):    

    # DEDUCE: most rectangular array of subpolots (nrows=nr, ncolumns=nc)

    factor_vec = np.sort(np.array(list( factors(n) )))
    factor_diff = [ np.abs(factor_vec[i]-np.median(factor_vec)) for i in range(len(factor_vec)) ]
    nr = factor_vec[ factor_diff == np.min(factor_diff) ][0] 
    nc = int(np.max(factor_vec) / nr)

    return nc, nr

def linear_regression_ols(x,y):
    
    regr = linear_model.LinearRegression()
    # regr = TheilSenRegressor(random_state=42)
    # regr = RANSACRegressor(random_state=42)
    
    X = x.reshape(len(x),1)
    t = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
    regr.fit(X, y)
    ypred = regr.predict(t.reshape(-1, 1))
    slope = regr.coef_[0]
    intercept = regr.intercept_
        
    return t, ypred, slope, intercept

def adjusted_r_squared(x,y):
    
    X = x.reshape(len(x),1)
    model = sm.OLS(y, X).fit()
    R2adj = model.rsquared_adj

    return R2adj

def calculate_piecewise_regression( x, y, depth, min_separation ):
    
    # FIT: linear tree regressor
                
    max_bins = int( min_separation/3 ) 			# 1/3 of min_samples_leaf in range[10,120]
    if max_bins < 10: max_bins = 10
    if max_bins > 120: max_bins = 120
    min_samples_leaf = int( min_separation * (120/min_separation) ) # 1 decade

    mask = np.isfinite(y)
    x_obs = np.arange( len(y) ) / len(y)
    x_obs = x_obs[mask].reshape(-1, 1)
    y_obs = y[mask].reshape(-1, 1)		     
    
    lt = LinearTreeRegressor(
        base_estimator = LinearRegression(),
        min_samples_leaf = min_samples_leaf,
        max_bins = max_bins,
        max_depth = depth        
    ).fit(x_obs, y_obs)    
    y_fit = lt.predict(x_obs)           
        
    # FIT: decision tree regressor

    dr = DecisionTreeRegressor(   
       max_depth = depth
    ).fit(x_obs, y_obs)
    x_fit = dr.predict(x_obs)

    # COMPUTE: goodness of fit

    mask_ols = np.isfinite(y_obs) & np.isfinite(y_fit.reshape(-1,1))
    corrcoef = scipy.stats.pearsonr(y_obs[mask_ols], y_fit.reshape(-1,1)[mask_ols])[0]
    R2adj = adjusted_r_squared(y_obs, y_fit.reshape(-1,1))
    
    return x_fit, y_fit, corrcoef, R2adj
 
def calculate_breakpoints( y, y_fit, min_separation, min_slope_change ):
       
    # BREAKPOINT: detection ( using slopes )

    y_fit_diff1 = np.array([np.nan] + list(np.diff(y_fit)))
    y_fit_diff2 = np.array([np.nan, np.nan] + list(np.diff(y_fit, 2)))
    y_fit_diff2[ y_fit_diff2 < 1e-6 ] = np.nan
    idx = np.arange( len(y_fit_diff2) )[ np.abs(y_fit_diff2) > 0] - 1     
    slopes_all = np.zeros(len(y))
    slopes_all[:] = y_fit_diff1[:]
    slopes_all[ idx ] = np.nan
    slopes_all[ idx + 1] = np.nan
    slopes_all = slopes_all * min_separation # slope = Q-sum / decade if min_separation=120                  
    slopes = np.zeros(len(y))
    for i in range(len(y)):    
        if i==0:        
            slopes[0] = 0.0        
        else:        
            if np.isnan(slopes_all[i]):        
                slopes[i] = slopes[i-3]
            else:            
                slopes[i] = slopes_all[i]                
    slopes_diff = np.array( [0.0] + list(np.diff(slopes)) )
    breakpoints_all = np.arange(len(y))[ np.abs(slopes_diff) > min_slope_change ] - 1     
    breakpoints_diff = np.array( [breakpoints_all[0]] + list( np.diff(breakpoints_all) ) )
    breakpoints = breakpoints_all[ breakpoints_diff > min_separation ] # decade minimum breakpoint separation
                
    return y_fit_diff2, slopes, breakpoints    

def smooth_fft(x, span):  
    
    y_lo, y_hi, zvarlo, zvarhi, fc, pctl = cru_filter.cru_filter_dft(x, span)    
    x_filtered = y_lo

    return x_filtered
    
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

#t_yearly = np.arange( df_compressed.iloc[0].year, df_compressed.iloc[-1].year + 1)
#df_yearly = pd.DataFrame({'year':t_yearly})
#df = df_yearly.merge(df_compressed, how='left', on='year')
#dt = df.groupby('year').mean().iloc[:,0:12]
#dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
#dn = dt.copy()
#dn.iloc[:,0:] = dn_array
#da = (dt - dn).reset_index()
#de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
#ds = (df.groupby('year').mean().iloc[:,43:55]).reset_index()        

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
# FIT: linear tree regression (LTR) model
#------------------------------------------------------------------------------
                    
# DEDUCE: subplot number or rows and columns

nc, nr = rc_subplots( max_depth )
    
# DEDUCE: optimal tree depth

r = []
r2adj = []
for depth in range(1,max_depth+1):       
       
    x_fit, y_fit, corrcoef, R2adj = calculate_piecewise_regression( x, y, depth, min_separation )    
    
    r.append(corrcoef)
    r2adj.append(R2adj)    
               
r_diff = np.array( [np.nan] + list(np.diff(r)) )
max_depth_optimum = np.arange(1,max_depth+1)[ r_diff <= min_correlation_change ][0] - 1

#------------------------------------------------------------------------------
# BEST: case
#------------------------------------------------------------------------------

x_fit, y_fit, corrcoef, R2adj = calculate_piecewise_regression( x, y, max_depth_optimum, min_separation )    
y_fit_diff2, slopes, breakpoints = calculate_breakpoints( y, y_fit, min_separation, min_slope_change )

# CALCULATE: intra-breakpoint fragment means
        
y_means = []
adjustments = []
for j in range(len(breakpoints)+1):                
    if j == 0:              
        y_means = y_means + list( np.tile( -np.nanmean( d[0:breakpoints[j]] ), breakpoints[j] ) ) 
        adjustment = [ -np.nanmean( d[0:breakpoints[j]] ) ]
    if (j > 0) & (j<len(breakpoints)):
        y_means = y_means + list( np.tile( -np.nanmean( d[breakpoints[j-1]:breakpoints[j]] ), breakpoints[j]-breakpoints[j-1] )) 
        adjustment = [ -np.nanmean( d[breakpoints[j-1]:breakpoints[j]] ) ]
    if (j == len(breakpoints)):              
        y_means = y_means + list( np.tile( -np.nanmean( d[breakpoints[-1]:] ), len(d)-breakpoints[-1] ) ) 
        adjustment = [ -np.nanmean( d[breakpoints[-1]:] ) ]
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

# GRAPHVIZ: plot regression tree model (full) and to depth=3            
# lt.plot_model(max_depth=max_depth_optimum)

#------------------------------------------------------------------------------
# PLOT: correlation r and R2adj as a function of linear regression tree depth
#------------------------------------------------------------------------------

if plot_ltr_correlation == True: 
                
    figstr = stationcode + '-' + 'ltr-correlation.png'
                
    fig, ax = plt.subplots(figsize=(15,10))
    plt.axvline( x=max_depth_optimum, ls='--', lw=1, color=default_color)
    plt.plot(np.arange( 1, max_depth+1), r, marker='o', ms=6, ls='-', lw=1, color='teal', alpha=1, label=r'$\rho$')
    plt.plot(np.arange( 1, max_depth+1), r2adj, marker='o', ms=6, ls='-', lw=1, color='lightblue', alpha=1, label=r'$R^{2}_{adj}$')
    plt.plot( max_depth_optimum, r[max_depth_optimum-1], marker='o', ms=12, ls='-', lw=3, color='teal', alpha=0.5, label=r'$\rho$( BEST )')
    plt.plot( max_depth_optimum, r2adj[max_depth_optimum-1], marker='o', markersize=12, ls='-', lw=3, color='lightblue', alpha=0.5, label=r'$R^{2}_{adj}$( BEST )')
    ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
    plt.grid(visible=None)
    plt.xticks(np.arange(1,max_depth+1))
    ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
    ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
    plt.grid(visible=None)
    plt.tick_params(labelsize=fontsize)  
    plt.xlabel('Depth', fontsize=fontsize)
    plt.ylabel(r'Goodness of fit', fontsize=fontsize)
    fig.legend(loc='lower center', ncol=4, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.subplots_adjust(left=0.1, bottom=0.15, right=None, top=None, wspace=None, hspace=None)  
    plt.title( stationcode, color=default_color, fontsize=fontsize)           
    plt.savefig(figstr, dpi=300)
    plt.close('all')    
 
#------------------------------------------------------------------------------
# PLOT: LTR breakpoint detection LOOP
#------------------------------------------------------------------------------

if plot_ltr_loop == True: 

	fontsize_multi = 12
		
	figstr = stationcode + '-' + 'ltr-loop.png'
	fig, ax = plt.subplots(figsize=(15,10))

	for depth in range(1, max_depth+1):       

		x_fit, y_fit, corrcoef, R2adj = calculate_piecewise_regression( x, y, depth, min_separation )    
		y_fit_diff2, slopes, breakpoints = calculate_breakpoints( y, y_fit, min_separation, min_slope_change )

		plt.subplot(nr, nc, depth)
		if depth < max_depth:        
			if ~np.isnan(documented_change): plt.axvline(x=documented_change_datetime, ls='-', lw=2, color='gold')                   
			plt.plot(t, y.ravel(), color='blue', ls='-', lw=3)
			plt.plot(t, y_fit.ravel(), color='red', ls='-', lw=2)            
			plt.fill_between( t, slopes.ravel(), 0, color='lightblue', alpha=0.5)
		else:
			if ~np.isnan(documented_change): plt.axvline(x=documented_change_datetime, ls='-', lw=2, color='gold', label='Documented change: ' + str(documented_change) )                   
			plt.plot( t, y, color='blue', ls='-', lw=3, label='CUSUM (O-E)')
			plt.plot( t, y_fit, color='red', ls='-', lw=2, label='LTR fit')
			plt.fill_between( t, slopes, 0, color='lightblue', alpha=0.5, label='CUSUM / decade' )
		ylimits = plt.ylim()       
					
		if depth == max_depth_optimum:
			plt.title( stationcode + ': depth=' + str(depth) + r' ( BEST ): $\rho$=' + str( f'{r[depth-1]:03f}' ), color=default_color, fontsize=fontsize_multi)                   
			df_breakpoints = pd.DataFrame({'breakpoint':t[breakpoints]})
			df_breakpoints.to_csv(stationcode + '-' + 'breakpoints.csv')
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
		else:
			plt.title( stationcode + ': depth=' + str(depth) + r' : $\rho$=' + str( f'{r[depth-1]:03f}' ), color=default_color, fontsize=fontsize_multi)           
			for i in range(len(t[(y_fit_diff2>0).ravel()])):
				plt.axvline( t[(y_fit_diff2>0).ravel()][i], ls='-', lw=1, color=default_color, alpha=0.2) 
			for i in range(len(breakpoints)):
				plt.axvline( t[breakpoints[i]], ls='dashed', lw=2, color=default_color)     
		plt.axhline( y=0, ls='-', lw=1, color=default_color, alpha=0.2)                    
		ax.xaxis.grid(visible=None, which='major', color='none', linestyle='-')
		ax.yaxis.grid(visible=None, which='major', color='none', linestyle='-')
		ax.grid(visible=None)    
		plt.tick_params(labelsize=fontsize_multi)    
		plt.xlabel('Year', fontsize=fontsize_multi)
		plt.ylabel(r'CUSUM (O-E), $^{\circ}$C', fontsize=fontsize_multi)    
		fig.tight_layout()

	fig.legend(loc='lower center', ncol=5, markerscale=6, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize_multi)       
	fig.subplots_adjust(left=0.1, bottom=0.1, right=None, top=None, wspace=None, hspace=None)  
	plt.savefig(figstr, dpi=300)
	plt.close('all')    
                  
#------------------------------------------------------------------------------
print('** END')        
    
