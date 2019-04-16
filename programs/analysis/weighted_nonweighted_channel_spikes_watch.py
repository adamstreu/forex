import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from scipy.optimize import leastsq
import os; os.chdir('/northbend')
from libraries.oanda           import get_candles
from classes.channel           import Channel
from libraries.outcomes        import outcomes
from libraries.transformations import pips_walk
from libraries.stats           import autocorrelation
from sklearn.preprocessing     import MinMaxScaler

'''
Runs through channels and graphs them all.
Purpose is to watch over time.

Looking for: a relationship between the closing values spiking, then 
the weighted closing values spiking. (weighted on volume)

If I want to get fancy I can add a diff grn or something to this.

'''

# Get Candles if required
###########################################################################
if False:
    # Candle Parameters
    instrument = 'EUR_USD'
    granularity = 'M1'
    _from = '2018-01-01T00:00:00Z'
    _to = '2019-01-01T00:00:00Z'
    # Get Candles
    candles = get_candles(instrument, granularity, _from, _to)



# Parameers
###########################################################################
channels_window  = 1500
outcomes_window  = int(channels_window / 4)
supports_window  = int(channels_window * 4)
support_bins     = 15
program_skip     = int(channels_window * .04) 
spike_by_channel = .5
spike_window     = 100



# Run Program  Get weights and Channels
###########################################################################
spike_collector = np.array([[0,0,0,0]])
for i in range(100500, #max(channels_window, supports_window), 
               candles.shape[0] - outcomes_window):

    # Get values
    start            = i
    closings         = candles.loc[start - channels_window +1: start, 
                                  'midclose'].values
    volumes          = candles.loc[start - channels_window +1: start, 
                                  'volume'].values
    outcomes_values = candles.loc[start: start + outcomes_window, 
                                  'midclose'].values
    
    # Get weights
    channel          = Channel(closings)
    weighted_closing = ((closings[1:] - closings[:-1]) / volumes[1:]).cumsum()
    w_channel        = Channel(weighted_closing)
    weighted_weird   = (closings[1:] - closings[:-1]).cumsum() / volumes[1:]
 
    # Analyze weighted channel spikes.  Add to collector
    if abs((w_channel.scaled[-spike_window:] \
            - w_channel.scaled[-1])).max() > spike_by_channel:
        spike_start     = int(channels_window - abs((w_channel.scaled[-spike_window:] - w_channel.scaled[-1])).argmax())
        spike_stop      = int(channels_window)
        spike_direction = int(np.sign((w_channel.scaled[-1] - w_channel.scaled[-spike_window:])[-abs((w_channel.scaled[-spike_window:] - w_channel.scaled[-1])).argmax()]))
        spike_collector = np.vstack([spike_collector, [i, 
                                                       spike_start, 
                                                       spike_stop, 
                                                       spike_direction]])   # = location drain, start, stop 
    
    # Get Supports
    support_values = candles.loc[start - supports_window: start, 'midclose'].values
    supports  = channel.get_supports(support_values)
    strengths = supports['strengths']
    supports  = supports['supports']
    supports_by_channel = (supports - channel.closings_c1[-1]) / (channel.closings_c7[-1] - channel.closings_c1[-1]).tolist()
    

    
    if i % program_skip == 0:
        # Plot Result
        ###########################################################################
        fig, ax = plt.subplots(2, 1, figsize=(20, 14 ))
        x1 = np.arange(channel.scaled.shape[0])
        x2 = np.arange(outcomes_values.shape[0]) + x1[-1]
        # Plot 1.  Channels and wave fit for closing and weighed closings    
        ax[0].plot(x1, channel.scaled, color='blue')
        ax[0].plot(w_channel.scaled, color='orange')
        ax[0].plot(x1, channel.c1, color='steelblue')
        ax[0].plot(x1, channel.c7, color='steelblue')    
        ax[0].plot(x1[:-1], w_channel.c1, color='navajowhite')
        ax[0].plot(x1[:-1], w_channel.c7, color='navajowhite')
        ax[0].plot(x2, np.zeros(x2.shape[0]), color='white')  
        # Plot supports
        for s in range(supports.shape[0]):
            if supports_by_channel[s] > 1:
                ax[0].plot(x2[0] + s * program_skip, 1, '^', color = 'green', markersize=15)
            elif supports_by_channel[s] < 0:
                ax[0].plot(x2[0] + s * program_skip, 0, 'v', color = 'green',  markersize=15)
            else:
                ax[0].plot(x2[0] + s * program_skip, supports_by_channel[s], '>', color='green', markersize=15)
        # Plot 2
        ax[1].plot(x1, channel.closings, color='skyblue')
        ax[1].plot(x1, channel.closings_c1, color='grey')
        ax[1].plot(x1, channel.closings_c7, color='grey')
        ax[1].plot(x2, outcomes_values, color = 'sandybrown')
        ax[1].plot(x2, np.ones(x2.shape[0]) * channel.closings_c1[-1], color='grey')
        ax[1].plot(x2, np.ones(x2.shape[0]) * channel.closings_c7[-1], color='grey')
        # Plot Spikes on closing with lines
        for s in range(spike_collector.shape[0]):
            if spike_collector[s, 1] > 0:
                if spike_collector[s, 3] == 1:
                    color = 'green'
                else:
                    color = 'red'
                ax[1].plot(np.arange(spike_collector[s, 1], spike_collector[s, 2]), 
                           channel.closings[spike_collector[s, 1]: spike_collector[s, 2]],
                           linewidth = 4, color = color )
        # Plot and print to keep in place
        plt.show()
        print(candles.loc[i, 'timestamp'])
        
    
    
    
    # Drain Spike collector
    ###########################################################################
    spike_collector[:, 0] -= 1
    spike_collector[:, 1] -= 1
    spike_collector[:, 2] -= 1
    


