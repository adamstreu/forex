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
Next:
    Sober up and summarize whayt I'm doing / where I am.
    Plot best results - make sure grouping makes sense.
    Make sure computing outcomes correctly
    Compare outcomes with channel width.  Anything?


Need to compre with channel width ? ? ? ? 
This is all - I am all on done after this.


did not collect largest spike timing anywhere on channel.
Rha;ts oo bad but wewre going without today.

Am I computing returns correctly.
Maybe not - cause I NEED to assume I can only buy so many at once.
So a ratio of 10:5 might be no different (except for spreads) than 2:1.
Need to investigate this and think it thorough when I asm sober.



Wednesday:
    
    Run program for 4 years( will ake a bunch of hours)


'''



# Get Candles if required
###########################################################################
if False:
    # Candle Parameters
    instrument = 'EUR_USD'
    granularity = 'M1'
    _from = '2017-01-01T00:00:00Z'
    _to = '2018-01-01T00:00:00Z'
    # Get Candles
    candles = get_candles(instrument, granularity, _from, _to)



# Main Iterative Program
###########################################################################
if False:
    # Parameters
    channels_window  = 7500
    outcomes_window  = int(channels_window / 4)
    supports_window  = int(channels_window * 4)
    support_bins     = 15
    program_skip     = int(channels_window * .04) 
    spike_windows    = np.array([50, 100, 200, 300, 400, 500])    
    # Instantiations
    results     = []
    long_target = []
    long_loss   = []
    short_target = []
    short_loss  = []
    for i in range(max(channels_window, supports_window), candles.shape[0] - outcomes_window):
        if i % 10000 == 0:  
            print('Percent complete: {:.2f}'.format(i / candles.shape[0]))
        
        # Get values
        start            = i
        closings         = candles.loc[start - channels_window +1: start, 'midclose'].values
        volumes          = candles.loc[start - channels_window +1: start, 'volume'].values
        outcomes_values  = candles.loc[start: start + outcomes_window, 'midclose'].values
        # Get weights
        channel          = Channel(closings)
        weighted_closing = ((closings[1:] - closings[:-1]) / volumes[1:]).cumsum()
        w_channel        = Channel(weighted_closing)
        weighted_weird   = (closings[1:] - closings[:-1]).cumsum() / volumes[1:]
        # weigthed spikes
        weighted_spikes = w_channel.scaled[-1] - w_channel.scaled[-spike_windows]
        # Append to results
        tmp =   [i,
                channel.channel_slope,
                channel.closings_slope,
                channel.closing_position,
                channel.channel_range,
                channel.largest_spike,
                channel.largest_spike_5,
                channel.within_range,
                candles.loc[i, 'spread'],
                candles.loc[i, 'volume'],
                channel.c1[-1],
                channel.c7[-1],
                channel.closings[-1],
                ]
        for each in weighted_spikes:
            tmp.append(each)
        results.append(tmp)
        

        # Set distance for outcome
        # distance = np.arange(1, 11) * (channel.channel_range / 6)
        distance = np.arange(1, 11) * .0015
        # Get long outcomes
        outs = outcomes('long', candles, i, channels_window * 2, distance, False)
        long_target.append([i] + outs['target'])
        long_loss.append([i] + outs['loss'])
        # get short outcomes
        outs = outcomes('short', candles, i, channels_window * 2, distance, False)
        short_target.append([i] + outs['target'])
        short_loss.append([i] + outs['loss'])
    
    
    # Assemble Dataframes
    results_columns = ['location',
                       'channel_slope', 
                       'closings_slope',
                       'channel_closing_position',
                       'channel_range',
                       'largest_spike',
                       'largest_spike_5',
                       'within_range',
                       'spread',
                       'volume',
                       'c1',
                       'c7',
                       'closing_value']
    for spike in spike_windows:
        results_columns.append('weighted_spike_' + str(spike))

    # Assemble results    
    results       = pd.DataFrame(np.array(results), columns = results_columns)
    results       = results.set_index('location', drop=True)
    results.index = results.index.astype(int)

    # Assemble Outcomes
    long_target  = pd.DataFrame(np.array(long_target))
    long_loss    = pd.DataFrame(np.array(long_loss))
    short_target = pd.DataFrame(np.array(short_target))
    short_loss   = pd.DataFrame(np.array(short_loss))
    # Set indexes
    long_target  = long_target.set_index(0, drop=True)
    long_loss    = long_loss.set_index(0, drop=True)
    short_target = short_target.set_index(0, drop=True)
    short_loss   = short_loss.set_index(0, drop=True)   
    # Correct Indexes
    long_target.index  = long_target.index.rename('location')
    long_loss.index    = long_loss.index.rename('location')
    short_target.index = short_target.index.rename('location')
    short_loss.index   = short_loss.index.rename('location')
    # Set index type
    long_target.index  = long_target.index.astype(int)
    long_loss.index    = long_loss.index.astype(int)
    short_target.index = short_target.index.astype(int)
    short_loss.index   = short_loss.index.astype(int)
    








# Cycle through everything and get returns.  Must change a number of things for short too long
###############################################################################
# Filters
filter_closing_position = .5
filter_slope            = 0
# Iterators
filter_spikes           = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
outcome_direction       = 'long'
# Initialize
analysis_collection = []
for spike_window in spike_windows:
    for filter_spike in filter_spikes:
        for t in long_target.columns:
            for l in long_target.columns:
                if t >= l:
                    # Assemble filter Conditions
                    cond1 = results['weighted_spike_' + str(spike_window)] > filter_spike  
                    cond2 = results.closings_slope > filter_slope
                    cond3 = results.channel_closing_position < filter_closing_position
                    filters = cond1 & cond2 & cond3
                    
                    # Create Groups
                    condition_index = filters[filters].index.values
                    if condition_index.shape[0] > 0:
                        groups = condition_index[np.insert(condition_index[1:] > condition_index[:-1] + spike_window, 0, True )]
                        
                        # Calculate winning percentages
                        if outcome_direction == 'long':
                            filters_wins = (long_target.loc[filters, t] < long_loss.loc[filters, l]).mean()
                            groups_wins     = (long_target.loc[groups, t] < long_loss.loc[groups, l]).mean()
                        else:
                            filters_wins = (short_target.loc[filters, t] < short_loss.loc[filters, l]).mean()
                            groups_wins     = (short_target.loc[groups, t] < short_loss.loc[groups, l]).mean()
                        # Spread
                        filters_spread = candles.loc[filters[filters].index, 'spread'].sum()    
                        groups_spread  = candles.loc[groups, 'spread'].sum()
                        # Wins and loss
                        filters_winnings = filters_wins * distance[t - 1]
                        filters_losses = (1 - filters_wins) * distance[l - 1]
                        groups_winnings = groups_wins * distance[t - 1]
                        groups_losses = (1 - groups_wins) * distance[l - 1]
                        # returns
                        filters_return = (filters.sum() * (filters_winnings - filters_losses)) - filters_spread
                        groups_return  = (groups.shape[0] * (groups_winnings - groups_losses)) - groups_spread
                        # Append Results to tmp
                        analysis_collection.append([spike_window,
                                                   filter_spike, 
                                                   t,
                                                   l, 
                                                   filters_wins,
                                                   groups_wins,
                                                   filters.sum(),
                                                   groups.shape[0],
                                                   filters_return,
                                                   groups_return])
columns    = ['spike',
              'filter', 
              'target', 
              'loss',
              'filters_wins',
              'groups_wins',
              'filters',              
              'groups', 
              'f_return',
              'g_return'] 
collection = np.array(analysis_collection)
analysis = pd.DataFrame(collection, columns = columns)
analysis['return_per_filters'] = analysis.f_return / analysis.filters
analysis['return_per_group'] = analysis.g_return / analysis.groups
            


# Filter Analysis
###############################################################################
analysis.sort_values('g_return', ascending=False)['g_return'].plot(style='o')
plt.show()
print(analysis.sort_values('g_return', ascending=False).head())
print()
print(analysis[analysis.groups < 50].sort_values('g_return', ascending=False).head())



# Plot maye the best returns





































"""

# Filter on some conditions.  Create Groups.  
###############################################################################
# Create Filters
filter_closing_position = .5
filter_slope            = 0
filter_spike_window     = 2  # [0, 1, 2, 3]  (currently)
filter_spike            = .50  # This is an unsigned value
outcome_direction       = 'long'  # long or short
filter_group_length     = spike_windows[filter_spike_window]
# Assemble filter Conditions
cond1 = results['weighted_spike_' + str(spike_windows[filter_spike_window])] \
      > filter_spike  # or less than the negative
cond2 = results.closings_slope > filter_slope
cond3 = results.channel_closing_position < filter_closing_position
filters = cond1 & cond2 & cond3


# Create Groups.  
###############################################################################
condition_index = filters[filters].index.values
groups = condition_index[np.insert(condition_index[1:] \
       > condition_index[:-1] + filter_group_length, 0, True )]


# Print stats for filters:
###############################################################################
print()
print('Matching Outcomes: {}'.format(filters.sum()))
print('Matching Groups  : {}'.format(groups.shape[0]))
print()




# Calculate Winning Percentages and Return.  Print analysis
###############################################################################
print('\t\tWinning % filters / Groups.\t\tReturn on filters / groups\n')
for t in long_target.columns:
    for l in long_target.columns:
        if t >= l:
            # Calculate winning percentages
            if outcome_direction == 'long':
                filters_wins = (long_target.loc[filters, t] < long_loss.loc[filters, l]).mean()
                groups_wins     = (long_target.loc[groups, t] < long_loss.loc[groups, l]).mean()
            else:
                filters_wins = (short_target.loc[filters, t] < short_loss.loc[filters, l]).mean()
                groups_wins     = (short_target.loc[groups, t] < short_loss.loc[groups, l]).mean()
            # Spread
            filters_spread = candles.loc[filters[filters].index, 'spread'].sum()    
            groups_spread  = candles.loc[groups, 'spread'].sum()
            # Wins and loss
            filters_winnings = filters_wins * distance[t - 1]
            filters_losses = (1 - filters_wins) * distance[l - 1]
            groups_winnings = groups_wins * distance[t - 1]
            groups_losses = (1 - groups_wins) * distance[l - 1]
            # returns
            filters_return = (filters.sum() * (filters_winnings - filters_losses)) - filters_spread
            groups_return  = (groups.shape[0] * (groups_winnings - groups_losses)) - groups_spread
            msg = '{}:  {} / {} \t\t {:.2f} \t {:.2f}  \t\t\t {:.4f} \t {:.4f}'
            print(msg.format(outcome_direction, 
                             t, 
                             l, 
                             filters_wins, 
                             groups_wins, 
                             filters_return,
                             groups_return))



# Plot findings
###############################################################################
y_groups = np.zeros(candles.shape[0])
y_filters = y_groups.copy()
y_groups[groups] = 1
y_filters[filters[filters].index.values] = .5
x = np.arange(y_groups.shape[0])
plt.figure(figsize = (8, 3))
plt.plot(x, y_groups, '+')
plt.plot(x, y_filters, '+')

"""







'''
# Combine with desired outcomes.  Push to Tableau  
###############################################################################
tab = results.copy()
for t in long_target.columns:
    for l in long_target.columns:
        tab['long_{}_{}'.format(t, l)]  = (long_target[t]  < long_loss[l]).astype(int)
        tab['short_{}_{}'.format(t, l)] = (short_target[l] < short_loss[l]).astype(int)
tab.to_csv('/Users/user/Desktop/tab.csv')
'''
