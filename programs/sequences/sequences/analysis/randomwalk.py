import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os; os.chdir('/sequences')
from libraries.oanda import get_candles
from libraries.indicators import up_down_mean_percentage


'''
Random Walk kind of stuff.
Probabilities of random walk kind of thing.
Ups and downs of window period

next.
For fun - see what happens if I sklearn normalize all data.

'''

# Parameters
###############################################################################
# Instrument
instrument  = 'EUR_USD'
granularity = 'S5'
_from       = '2018-01-01T00:00:00Z'
_to         = '2018-03-01T00:00:00Z'
# Windows
window = 720

# Call the candles
candles = get_candles(instrument, granularity, _from, _to)
# Call the up down mean percentage indicator
ud720 = up_down_mean_percentage(candles.midclose.values, 720)
ud360 = up_down_mean_percentage(candles.midclose.values, 45)
ud180 = up_down_mean_percentage(candles.midclose.values, 15)
# Collect Some results into a dataframe
df = pd.DataFrame(candles.loc[window::60, ['timestamp', 'midclose']])
df.index.names = ['ind']
df['up_mean_720'] = ud720['up_mean'][::60]
df['up_mean_360'] = ud720['up_mean'][::60]
df['up_mean_180'] = ud720['up_mean'][::60]
df['down_mean_720'] = ud720['down_mean'][::60]
df['down_mean_360'] = ud720['down_mean'][::60]
df['down_mean_180'] = ud720['down_mean'][::60]
df['up_perc_720'] = ud720['up_perc'][::60]
df['up_perc_360'] = ud720['up_perc'][::60]
df['up_perc_180'] = ud720['up_perc'][::60]
df['down_perc_720'] = ud720['down_perc'][::60]
df['down_perc_360'] = ud720['down_perc'][::60]
df['down_perc_180'] = ud720['down_perc'][::60]
df.to_csv('/Users/user/Desktop/updown.csv')




"""
# Get results for long series and append to collection
###############################################################################
for i in range(10000, 400000, 10000):
    print(i)
    start = i
    plt.figure(figsize=(14,4))
    plt.plot(ud['up_perc'][start:start+window], color='blue', label='up_perc', marker = '+')
    plt.plot(ud['down_perc'][start:start+window], color='orange', label='down_perc', marker = '+')
    plt.title('Percentage')
    plt.legend()
    plt.tight_layout()
    plt.figure(figsize=(14,4))
    plt.plot(ud['up_mean'][start:start+window], color='blue', label='up_mean', marker = '+')
    plt.plot(ud['down_mean'][start:start+window],  color='orange', label='down_mean', marker = '+')    
    plt.tight_layout()
    plt.title('Mean')
    plt.legend()
    plt.figure(figsize=(14,4))
    plt.plot(close_std[start:start+window*spacing:spacing], marker = '+')
    plt.title('Closing Values')
    plt.tight_layout()
    plt.show()
    time.sleep(2)
    
"""
    
'''
Look at the difference between starting ay 30000 and 80000.
at 30, it almost appears that up_mean predicts a jump up twice
at 80, there are still spikes but not as large.]
Can we chase this?
'''





'''
# Wraps in the channel stuff
###############################################################################
closings = up_mean[start:start+window*spacing:spacing]
closings_flat = horizontal_transform(closings)
channels = create_channels(closings_flat['closing'])
x1 = np.arange(closings.shape[0])
plt.figure(figsize=(14,4))
plt.plot(x1, closings)
plt.plot(x1, (channels['c2'] + closings_flat['linregress']) + closings[0])
plt.plot(x1, (channels['c3'] + closings_flat['linregress']) + closings[0])
plt.plot(x1, (channels['c4'] + closings_flat['linregress']) + closings[0])
plt.plot(x1, closings_flat['linregress'] + closings[0])
'''

    
    
    
    
    
    
    
    