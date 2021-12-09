import numpy as np
import scipy
from sklearn import linear_model
from sklearn.linear_model import *
import statsmodels.api as sm
from lineartree import LinearTreeClassifier, LinearTreeRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
                
def changepoint_detector(x,y):
    '''
    ------------------------------------------------------------------------------
    PROGRAM: cru_changepoint_detector.py
    ------------------------------------------------------------------------------
    Version 0.3
    11 November, 2021
    Michael Taylor
    https://patternizer.github.io
    patternizer AT gmail DOT com
    michael DOT a DOT taylor AT uea DOT ac DOT uk
    ------------------------------------------------------------------------------
    Uses linear tree regression fit to CUSUM series to detect changepoints.
    
    y		 	: timeseries
    x		 	: timeseries index (e.g. decimal year dates)
    
    RETURNS:
    
    y_fit 				: linear tree regression fit at dervied optimal tree depth    
    y_fit_diff1 		: 1st difference of y_fit
    y_fit_diff2 		: 2nd difference of y_fit
    slopes				: vector of LTR segment slope
    breakpoints 		: vector of breakpoints detected
    max_depth_optimum 	: optimal tree depth derived from goodness of fit and correlation
    r					: vector of Pearson correlation coefficients as a function of tree depth
    R2adj				: vector of adjusted R^2 as a function of tree depth    
    ------------------------------------------------------------------------------
    CALL SYNTAX: y_fit, y_fit_diff1, y_fit_diff2, slopes, breakpoints, max_depth_optimum, r, R2adj = changepoint_detector(x, y)
    
    ------------------------------------------------------------------------------
    '''

    #--------------------------------------------------------------------------
    # METHODS
    #--------------------------------------------------------------------------

    def linear_regression_ols(x,y):
    
        regr = linear_model.LinearRegression()
        # regr = TheilSenRegressor(random_state=42)
        # regr = RANSACRegressor(random_state=42)
    
        X = x.reshape(len(x),1)
        t = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
        regr.fit(X, y)
        ypred = regr.predict(t.reshape(-1, 1))
        ypred = []
        slope = regr.coef_[0]
        intercept = regr.intercept_     
        
        return t, ypred, slope, intercept
    
    def adjusted_r_squared(x,y):
        
        X = x.reshape(len(x),1)
        model = sm.OLS(y, X).fit()
        R2adj = model.rsquared_adj

        return R2adj

    #--------------------------------------------------------------------------
    # SETTINGS (pre-optimised for monthly timeseries)
    #--------------------------------------------------------------------------

    max_depth = 9 			# in range [1,20]
    min_samples_leaf = 120 	# 1 decade
    max_bins = 40 			# 1/3 of min_samples_leaf
    min_separation = 120 	# 1 decade
    min_slope_change = 6 	# units: CUSUM / decade

    # FORMAT: mask and reshape data

    mask = np.isfinite(y)
    x_obs = np.arange( len(y) ) / len(y)
    x_obs = x_obs[mask].reshape(-1, 1)
    y_obs = y[mask].reshape(-1, 1)		                

    # DEDUCE: optimal tree depth

    r = []
    r2adj = []
    for depth in range(1,max_depth+1):       
	       
        # FIT: linear tree regression (LTR) model

        lt = LinearTreeRegressor(
            base_estimator = LinearRegression(),
            min_samples_leaf = min_samples_leaf,
            max_bins = max_bins,
            max_depth = depth
        ).fit(x_obs, y_obs)            
        y_fit = lt.predict(x_obs)    
        
        # FIT: decision tree regressor ( optional )

        dr = DecisionTreeRegressor(   
            max_depth = depth
        ).fit(x_obs, y_obs)
        x_fit = dr.predict(x_obs)

        # COMPUTE: goodness of fit

        mask_ols = np.isfinite(y_obs) & np.isfinite(y_fit.reshape(-1,1))
        corrcoef = scipy.stats.pearsonr(y_obs[mask_ols], y_fit.reshape(-1,1)[mask_ols])[0]
        R2adj = adjusted_r_squared(y_obs, y_fit.reshape(-1,1))        	       
        r.append(corrcoef)
        r2adj.append(R2adj)    

    r_diff = np.array( [np.nan] + list(np.diff(r)) )
    max_depth_optimum = np.arange(1,max_depth+1)[ r_diff < 0.001 ][0] - 1
    if max_depth_optimum <= 2: max_depth_optimum = 3
            
    # FIT: LTR model for optimum depth and extract breakpoints
		
    lt = LinearTreeRegressor(
        base_estimator = LinearRegression(),
        min_samples_leaf = min_samples_leaf,
        max_bins = max_bins,
        max_depth = max_depth_optimum        
    ).fit(x_obs, y_obs)    
    y_fit = lt.predict(x_obs)            
    y_fit_diff = [0.0] + list(np.diff(y_fit))        
   
    # BREAKPOINT: detection ( using slopes )

    y_fit_diff1 = np.array([np.nan] + list(np.diff(y_fit)))
    y_fit_diff2 = np.array([np.nan, np.nan] + list(np.diff(y_fit, 2)))
    y_fit_diff2[ y_fit_diff2 < 1e-6 ] = np.nan
    idx = np.arange( len(y_fit_diff2) )[ np.abs(y_fit_diff2) > 0] - 1     
    slopes_all = np.zeros(len(y))
    slopes_all[:] = y_fit_diff1[:]
    slopes_all[ idx ] = np.nan
    slopes_all[ idx + 1] = np.nan
    slopes_all = slopes_all * min_separation # slope = Q-sum / decade ( min_separation=120 )
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
    if len(breakpoints_all) > 0:
        breakpoints_diff = np.array( [breakpoints_all[0]] + list( np.diff(breakpoints_all) ) )
        breakpoints = breakpoints_all[ breakpoints_diff > min_separation ] # minimum breakpoint separation = decade ( min_separation )
    else:
        breakpoints = []
                
    return y_fit, y_fit_diff1, y_fit_diff2, slopes, breakpoints, max_depth_optimum, r, R2adj           
#------------------------------------------------------------------------------



