import numpy as np
import pandas as pd
                
def cru_filter_dft(y, w):

    '''
    ------------------------------------------------------------------------------
    PROGRAM: filter_cru_dft.py
    ------------------------------------------------------------------------------
    Version 0.3
    6 August, 2021
    Michael Taylor
    https://patternizer.github.io
    patternizer AT gmail DOT com
    michael DOT a DOT taylor AT uea DOT ac DOT uk
    ------------------------------------------------------------------------------
    Uses a zero-padded Hamming window DFT filter design.
    
    y		: timeseries
    w		: smoothing window
    fc		: (derived) cut-off frequency = 1/w
    pctl	: (derived) percentile of low pass spectral variance
    lut	: pre-calculated look-up table to convert fc --> pctl
    zvarlo	: spectral variance of the low pass signal component
    zvarhi	: spectral variance of the high pass signal component
    y_lo	: low pass DFT filtered signal in the time domain
    y_hi	: high pass DFT filtered signal in the time domain    
    ------------------------------------------------------------------------------
    CALL SYNTAX: y_lo, y_hi, zvarlo, zvarhi, fc, pctl = dft_filter(y, w)
    ------------------------------------------------------------------------------
    '''

    #------------------------------------------------------------------------------
    # METHODS
    #------------------------------------------------------------------------------

    def nextpowerof2(x):
    
        if x > 1:
            for i in range(1, int(x)):
                if ( 2**i >= x ):
                    return 2**i
        else:
            return 1

    def dft(x):

        # Discrete Fourier Transform

        N = len(x)
        n = np.arange(N)
        k = n.reshape((N, 1))
        e = np.exp(-2j * np.pi * k * n / N)
        
        return np.dot(e, x)

    def idft(x):

        # Inverse Discrete Fourier Transform

        N = len(x)
        n = np.arange(N)
        k = n.reshape((N, 1))
        e = np.exp(2j * np.pi * k * n / N)
        
        return 1/N * np.dot(e, x)
    
    #------------------------------------------------------------------------------
    # COMPUTE: DFT low and high pass components --> y_lo, y_hi
    #------------------------------------------------------------------------------
    
    N = len(y)                                                      # signal length
    n = np.arange(N)
    Fs = 1                                                          # sampling rate
    Ts = N/Fs                                                       # sampling interval 
    f = n/Ts                                                        # frequencies [0,1] 
    freqA = f[0:int(N/2)]                                           # frequencies: RHS
    freqB = f[int(N/2):]-1                                          # frequencies: LHS
    freq = np.array(list(freqA) + list(freqB))                      # construct SciPy's fftfreq(len(y), d=1) function
          
    y_mean = np.nanmean(y)
    y = y - y_mean                                                  # cneter timeseries (NB: add back in later)
            
    z = dft(y)                                                      # DFT
    zamp = np.sqrt(np.real(z)**2.0 + np.imag(z)**2.0) / N           # ampplitudes
    zphase = np.arctan2( np.imag(z), np.real(z) )                   # phases

    zvar = zamp**2                                                  # variance of each harmonic
    zvarsum = np.sum(zvar)                                          # total variance of spectrum
    yvarsum = np.var(y)                                             # total variance of timeseries
    zvarpc = zvar / zvarsum                                         # percentage variance per harmonic
        
    if w >= 2:
        
        df = pd.read_csv( 'ml_optimisation.csv' )		      # Look up table (LUT): w --> fc and pctl
        fc = 1.0/w
        pctl = df[df['fc'] > fc]['pctl'].iloc[-1]        

    else:
                       
        pctl = 90                                                   # (default) if no fc provided
        
    zpeaks = zvarpc[ zvarpc > ( np.percentile(zvarpc, pctl) ) ]     # high variance peaks > p(pctl)
    zpeaks_idx = np.argsort( zpeaks )                               # peak indices
    znopeaks_idx = np.setdiff1d( np.argsort( zvarpc ), zpeaks_idx)  # remaining indices
    npeaks = len(zpeaks_idx)                                        # number of peaks
    zvarlo = np.sum( [zvarpc[i] for i in zpeaks_idx] )              # total percentage variance of low pass 
    zvarhi = np.sum( [zvarpc[i] for i in znopeaks_idx] )            # total percentage variance of high pass  

    if w < 2:
        
        fc = freq[ zpeaks_idx.max() ]                               # estimate low pass / high pass cut-off       
                 
    # FILTER DESIGN: low pass filter (Hamming window) with zero-padding
        	
    if N%2 == 0:    
        L = N+1                                                 # filter length (M+1)
    else:
        L = N                                                   # filter length (M+1)
    h_support = np.arange( -int((L-1)/2), int((L-1)/2)+1 )      # filter support
    h_ideal = ( 2*fc/Fs) * np.sinc( 2*fc*h_support/Fs )         # filter (ideal)
    h = np.hamming(L).T*h_ideal                                 # filter
        
    # ZERO-PAD: (next power of 2 > L+M-1) signal and impulse-response
        
    Ndft = nextpowerof2(L+N-1)
    yzp = list(y) + list(np.zeros(Ndft-N+1))
    hzp = list(h) + list(np.zeros(Ndft-L+1))    
        
    # COMPUTE: FFT of signal and filter in freq domain
        
    Y = dft(yzp)                                                # DFT signal 
    H = dft(hzp)                                                # DFT filter
        
    # COMPUTE: cyclic convolution (pairwise product) of signal and filter in freq domain
        
    Z = np.multiply(Y, H)
    y_filtered_lo = np.real( idft(Z)[int(N/2):N+int((N)/2)] )   # low pass signal    
    y_filtered_hi = y - y_filtered_lo                           # high pass signal
                    
    return y_filtered_lo + y_mean, y_filtered_hi, zvarlo, zvarhi, fc, pctl
#------------------------------------------------------------------------------
