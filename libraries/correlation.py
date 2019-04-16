import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import os; os.chdir('/northbend')
from libraries.taps import get_taps


def get_autocorrelation(values, rolling_ac_window=30, 
                    taps_left=30, taps_right=30, plot=False):
    
    ac = acf(values, unbiased=False, 
             nlags=values.shape[0], qstat=False, fft=None, 
             alpha=None, missing='none')
    rolling_ac = pd.Series(ac).rolling(rolling_ac_window).mean().values
    taps = get_taps(rolling_ac, taps_left, taps_right)
    taps_upper = taps['upper']
    taps_lower = taps['lower']
    
    upper_cycle = np.arange(taps['upper'].shape[0])[taps['upper']]
    lower_cycle = np.arange(taps['lower'].shape[0])[taps['lower']]   
    
    
    if plot:
        plt.plot(rolling_ac)
        plt.plot(np.arange(ac.shape[0])[taps_upper], rolling_ac[taps_upper], 'o', color='orange')
        plt.plot(np.arange(ac.shape[0])[taps_lower], rolling_ac[taps_lower], 'o', color='red')
    

    return {
            'autocor': rolling_ac,
            'upper_cycle': upper_cycle,
            'lower_cycle': lower_cycle
            }





# =============================================================================
# Get correlation on one and multiple windows
# =============================================================================

''' Correlation between two currencies on one window '''
def get_correlation(val1, val2, window, std_ratio=2):
    # Get Rolling Correlation on window over both currencies
    correlation_collection  = [np.nan] * window
    for i in range(window, val1.shape[0]):
        corr = np.corrcoef(val1[i - window: i], val2[i - window: i])[0, 1]
        correlation_collection.append(corr)
    return np.array(correlation_collection)


''' Correlation between two currencies on TWO window '''
def get_rolling_correlation_waves(val1, val2, windows = [15, 30, 60, 90]):
    correlation_collection = []
    for window in windows:
        # Get Rolling Correlation on window over both currencies
        corr_collection= [np.nan] * window    
        for i in range(window, val1.shape[0]):
            corr = np.corrcoef(val1[i - window: i], val2[i - window: i])[0, 1]
            corr_collection.append(corr)
        correlation_collection.append(corr_collection)
    # Comnvert to Df
    df = pd.DataFrame(np.array(correlation_collection).T, columns = windows)
    return df 




