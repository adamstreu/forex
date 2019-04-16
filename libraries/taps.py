import pandas as pd
import numpy as np
import os; os.chdir('/northbend')



def get_taps(values, left_interval, right_interval):
    
    '''
    Find taps (upper and lower) on values as boolean array
    '''

    # Get Upper Taps
    y = values
    flip = np.flip(y, axis=0)
    left = pd.Series(y).rolling(left_interval).max().values 
    right = pd.Series(flip).rolling(right_interval).max().values
    left_match = y == left
    right_match = flip == right
    match_upper= left_match & np.flip(right_match, axis=0)
    
    # Lower Taps      
    y = values
    flip = np.flip(y, axis=0)
    left = pd.Series(y).rolling(left_interval).min().values 
    right = pd.Series(flip).rolling(right_interval).min().values
    left_match = y == left
    right_match = flip == right
    match_lower= left_match & np.flip(right_match, axis=0)
    
    # Return
    return {
            'upper': match_upper,
            'lower': match_lower
           }






"""
def get_taps(candles, start, end, left_interval, right_interval):
    
    '''
    Uses bidhigh and ask low to find spike tops and bottoms
        sort of.  Really just finds his and lows for l, r, periods
    '''

    # Get Upper Taps
    y = candles.loc[start: end, 'bidhigh'].values
    flip = np.flip(y, axis=0)
    left = pd.Series(y).rolling(left_interval).max().values 
    right = pd.Series(flip).rolling(right_interval).max().values
    left_match = y == left
    right_match = flip == right
    match = left_match & np.flip(right_match, axis=0)
    taps_upper = candles.loc[match, 'bidhigh'].index.values
    
    # Lower Taps       
    y = candles.loc[start: end, 'asklow'].values
    y_flip = ((y[1:] - y[:-1]) * -1).cumsum() + y[0] 
    y_flip = np.insert(y_flip, 0, y[0])
    flip = np.flip(y_flip, axis=0)
    left = pd.Series(y_flip).rolling(left_interval).max().values 
    right = pd.Series(flip).rolling(right_interval).max().values
    left_match = y_flip == left
    right_match = flip == right
    match = left_match & np.flip(right_match, axis=0)
    taps_lower = candles.loc[match, 'asklow'].index.values
    
    # Return
    return {
            'taps_upper': taps_upper,
            'taps_lower': taps_lower
           }
    

"""
