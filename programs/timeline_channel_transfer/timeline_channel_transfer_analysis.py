import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import os; os.chdir('/northbend')
from libraries.outcomes import outcomes
from libraries.taps import taps
from classes.channel import Channel


'''
GEt channel and relaed satistics over many timelines.
Assume that movements are always rying to get back to a channel if they can.
When they can't so to speak they might be drifting new.
    and lots of other ways to look at it - typing this brief and fast
exploratory - for dumping tino tableua


intervals could be better  - closer together on front end than back


'''



###############################################################################
# Import and clean data adnd Set Environment
###############################################################################
if True:
    
    print('Getting Candles')
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15)
    # Get data
    folder = 'EUR_USD_M1_2014-01-11_2019-01-01'
    path   = '/Users/user/Desktop/' + folder + '/'
    candles = pd.read_csv(path + 'candles.csv')
    candles.set_index('location', drop=True, inplace=True)
    candles_backup = candles.copy()
    candles = candles.loc[:200000]



###############################################################################
# Iterate over candles positions, iterate over Channels, pickup stats.
###############################################################################
if True:
    
    # Set out what channel timeframes to collect
    intervals = np.linspace(30, 3000, 20).astype(int)    
    start = 30000
    stop = 60000
    # Run through Iterations collecting stats and outcomes
    up = []
    down = []
    channel_coll = []
    for location in candles.index.values[start: stop]:
        # Print for progress
        if location % 1000 == 0:
            print(location)
        # Get same sats on each interval at location
        for interval in intervals:
            values = candles.loc[location - interval:location, 'bidhigh'].values
            # Get Channel and staistics
            channel = Channel(values)
            channel_coll.append([location,
                                 interval, 
                                 candles.loc[location, 'timestamp'],
                                 candles.loc[location, 'bidhigh'],
                                 channel.slope,
                                 channel.channel_deviation,
                                 channel.position_distance,
                                 channel.position_distance_standard
                                 ])
        # Get outcomes at location for one simple distance for now 
        # distance = np.arange(1, 4) * channel.deviation
        distance = [.00350]
        outs = outcomes('short', candles, location, 15000, distance)            
        up.append(outs['loss'])
        down.append(outs['target'])
            
    # Set results DataFrames
    results_columns = ['location', 
                       'interval', 
                       'timestamp', 
                       'bidhigh',
                       'slope', 
                       'channel_deviation', 
                       'final_distance', 
                       'final_distance_std']
    results = pd.DataFrame(channel_coll, columns=results_columns)
    results.set_index(['location', 'interval'], inplace=True, drop=True)         
    # Ge Location Sums
    location_sums = results.groupby(by='location').sum()
    # Set outcomes Dataframe
    up = pd.DataFrame(up, columns=[.002])
    down = pd.DataFrame(down, columns=[.002])     
              

            
###############################################################################
# Export for tableau
###############################################################################            
out = (down < up).astype(int).values.ravel()
results['outcomes'] = np.repeat(out, len(intervals))




results['midclose_flat_std'] = results.bidhigh.values + results.channel_deviation.values

results['midclose_flat_std'] = results.bidhigh.values + results.channel_deviation.values

results['midclose_roll_std'] = results.bidhigh.values + results['bidhigh'].rolling(1500).std().values

results['midclose_roll_std_low'] = results.bidhigh.values - results['bidhigh'].rolling(1500).std().values

results['midclose_flat_std_low'] = results.bidhigh.values - results.channel_deviation.values

results['midclose_flat_std'] = results.bidhigh.values + results.channel_deviation.values

results.loc[31000:].to_csv('/Users/user/Desktop/multi.csv')





results.to_csv('/Users/user/Desktop/multi.csv')
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            