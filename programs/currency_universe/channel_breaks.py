import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
from scipy.stats import rv_discrete
import os; os.chdir('/northbend')
from libraries.currency_universe import get_currencies
from libraries.transformations import get_groups
from classes.channel import Channel
from classes.wave import Wave

# Set Environment
pd.set_option('display.width', 1000)
pd.set_option('max.columns', 15)
np.warnings.filterwarnings('ignore')

'''
A Channel break might happen when the diff between the bollinger band and 
the cahnnel band grows
'''

###############################################################################
# Get instruments.  Call currency Matrix.
###############################################################################
if 1:

    # Get currency universe
    granularity = 'M1'
    _from = '2018-01-01T00:00:00Z'
    _to   = '2018-04-01T00:00:00Z'
    currency_dictionary = get_currencies(granularity, _from, _to)
    cur = currency_dictionary['currencies']
    curdiff = currency_dictionary['currencies_delta']
    ratios = currency_dictionary['ratios']
        
    # Create Diff DataFrame for currencies
    cur_diff = pd.DataFrame()
    for column in cur.columns:
        roll = cur[column].rolling(window=2) \
                           .apply(lambda x: (x[1] - x[0])).values
        cur_diff[column] = roll



###############################################################################
###############################################################################
if 1:    
    
    end = 600
    pred = 1200
    channel  = Channel(cur.loc[:end, 'aud'].values, std_ratio = 2)
    wave = Wave(cur.loc[:end, 'aud'].values)
    plt.plot(cur.loc[:end, 'aud'].values)
    plt.figure()
    plt.plot(channel.flattened)
    plt.plot(np.zeros(channel.flattened.shape[0]), color='black')
    plt.plot(np.zeros(channel.flattened.shape[0]) + channel.channel_deviation * 2, color='black')        
    plt.plot(np.zeros(channel.flattened.shape[0]) - channel.channel_deviation * 2, color='black')        
    plt.plot(wave.channel_wave)
    plt.figure()
    plt.plot(cur.loc[:pred, 'aud'].values)    
    plt.plot(cur.loc[:end, 'aud'].values)
    plt.plot(channel.c1(), color='black')
    plt.plot(channel.c3(), color='black')
    plt.plot(channel.c5(), color='black')
    plt.plot(wave.wave)
    
    
    
###############################################################################
# Better Autocorrelations
###############################################################################
if 0:
    
    def serial_corr(wave, lag=1):
        n = wave.shape[0]
        y1 = wave[lag:]
        y2 = wave[:n-lag]
        corr = np.corrcoef(y1, y2, ddof=0)[0, 1]
        return corr
        
    def autocorr(wave):
        lags = range(wave.shape[0]//2)
        corrs = [serial_corr(wave, lag) for lag in lags]
        return lags, corrs
    
    
    ac = autocorr(cur.loc[10000:10100, 'aud'].values)
    plt.plot(ac[1])
    
    
    