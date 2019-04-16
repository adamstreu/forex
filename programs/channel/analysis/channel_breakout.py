import numpy as np
import pandas as pd
import os; os.chdir('/forex')
from libraries.transformations import get_groups
from libraries.oanda           import get_candles
from classes.channel           import channel


'''

THERE ARE STILL ISSUES.
RUN A BUNCH OF GRAPH AND YOU WILL SEE.

    BARS MIGHT NOT BE RIGHT WHEN BOTH EQUAL 501,
    RANGE I FOUND A NEGATIVE
    CLOSING PIOSITIONS MIGHT NOT BE ACCURATE







Next:
    Get a program goin that pick out channel outbreaks and
    puts a 3:1 bid on directions:
        top breakout
        

Purpose:
    See if I can find any reliable patterns associated with breaking out of
    the tops or bottoms of a channel


Next analysis:
    remove results where bars are both equal to whaterver...(neccesary ? )
    breakouts different lweves
    filter or not on slope
    only get results Tues - thursday
    tot_ex = groups * ex
    
    not important: graph group spaacing make sure were reasonable



Analyze new breakout results - Preliminaries:
    Double check (by graphing and logic) that correct results are bing provided
    Check some different window lenghts, granularity, etc.
    Anything there?
    If not:
        can we filter (tableau?)
        can we add a few extra things in.
    If results are that different than before:
        can we throw some extra shit in and see what mk can do?
        

If I want to work with good, model patterns:
    do not use if there are any large jumps in closing values (throws off channel)
        any one vlaue _x_ greater than average value
    channels slope needs to be close to 0
    bimodal ? 
    closing position is at a peak
    go only in direction of slope
    how well can we fit a sin wave to it
    well defined trough beween bimodal (sya, almost half ? ? ?)
    Need some stats on how well the channels fit the pattern
    

Analysis thoughts:
    expected values seem to be better using all breakouts instead of groups.
        Maybe harder to manage though.
    Mondays seem like good days
    
'''



# Parameters
############################################################################### 
# Instrument
instrument  = 'EUR_USD'
granularity = 'M1'
_from =       '2016-01-01T00:00:00Z'
_to   =       '2018-01-01T00:00:00Z'
# Window Pareters
window = 500
search_interval = window
# Instantiation
results = []
bars = []
candles = get_candles(instrument, granularity, _from, _to)


# Calculate Window and Channel Results
############################################################################### 
for i in range(window, candles.shape[0] - search_interval):
    # Print progress.
    if i % 10000 == 0:  
        print('Percent complete: {:.2f}'.format(i / candles.shape[0]))
    # Prepare Slice candles for channel and outcome_interval
    closings = candles.loc[i - window: i, 'midclose'].values
    outcome_interval= candles.loc[i: i + search_interval, 'midclose'].values
    # Fetch channel transformation on window.  Append to results
    c = channel(closings)
    results.append([i,
                    c.channel_slope,
                    c.closings_slope,
                    c.closing_position,
                    c.channel_range,
                    ])
    # Get bar outcomes on window
    bars.append([i] + c.outcomes(outcome_interval))
    
# Collect all window results into dataframe
results_columns = ['location',
                   'channel_slope', 
                   'closings_slope',
                   'channel_closing_position',
                   'channel_range', 
                   ]
results = pd.DataFrame(np.array(results), columns = results_columns)
results = results.set_index('location', drop=True)
results.index = results.index.astype(int)
# Assemble bars into dataframe
bars_columns = ['location', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3']
bars = pd.DataFrame(np.array(bars), columns=bars_columns)
bars = bars.set_index('location', drop=True)
bars.index = bars.index.astype(int)


# Analysis
############################################################################### 
'''
Currenlty using GROUP start instead of all\
'''
filter_on_slope  = True
analysis         = []
group_gap_length = window 
top_filter       = .7
bottom_filter    = .3
# Get all values from results where closing position is outside of
top_breakouts    = results[results.channel_closing_position > top_filter].index
bottom_breakouts = results[results.channel_closing_position < bottom_filter].index
top_groups = get_groups(top_breakouts, group_gap_length)
bottom_groups = get_groups(bottom_breakouts, group_gap_length)
# Filter on slope
if filter_on_slope:
    slopes_up   = (results.closings_slope > 0).index
    slopes_down = (results.closings_slope < 0).index
    top_groups  = np.intersect1d(top_groups, slopes_up)
    bottom_groups  = np.intersect1d(bottom_groups, slopes_down)
    top_breakouts  = np.intersect1d(top_breakouts, slopes_up)
    bottom_breakouts  = np.intersect1d(bottom_breakouts, slopes_down)
for d in ['d1', 'd2', 'd3'] :
    for u in ['u1', 'u2', 'u3']:
        # Assess top breakouts
        breakout_results = bars.loc[top_groups]
        top_down_wins    = (breakout_results[d] < breakout_results[u]).mean()
        top_up_wins      = (breakout_results[d] > breakout_results[u]).mean()
        # Assess bottom breakouts
        breakout_results = bars.loc[bottom_groups]
        bottom_down_wins = (breakout_results[d] < breakout_results[u]).mean()
        bottom_up_wins   = (breakout_results[d] > breakout_results[u]).mean()
        # Colect results into DataFrame
        analysis.append([d, 
                         u, 
                         top_down_wins, 
                         top_up_wins, 
                         bottom_down_wins, 
                         bottom_up_wins])
# Assemble analysis into dataframes
analysis_columns = ['down', 
                    'up', 
                    'top_down_wins', 
                    'top_up_wins', 
                    'bottom_down_wins',
                    'bottom_up_wins']
analysis = pd.DataFrame(np.array(analysis), columns=analysis_columns)
analysis = analysis.apply(pd.to_numeric, errors='ignore')
# Calculate expected values
analysis['top_down_ex']    = (analysis.top_down_wins    * analysis.down.str[1].astype(int)) - (analysis.up.str[1].astype(int)   * (1 - analysis.top_down_wins))
analysis['top_up_ex']      = (analysis.top_up_wins      * analysis.up.str[1].astype(int))   - (analysis.down.str[1].astype(int) * (1 - analysis.top_up_wins))
analysis['bottom_down_ex'] = (analysis.bottom_down_wins * analysis.down.str[1].astype(int)) - (analysis.up.str[1].astype(int)   * (1 - analysis.bottom_down_wins))
analysis['bottom_up_ex']   = (analysis.bottom_up_wins   * analysis.up.str[1].astype(int))   - (analysis.down.str[1].astype(int) * (1 - analysis.bottom_up_wins))
# Print Analysis
# Print breakouts quantityie and groups
msg = '\nTop and Bottom Filters: {}\t{}'
print(msg.format(top_filter, bottom_filter))
msg = 'Top breakout count:     {} with {} groups.'
print(msg.format(top_breakouts.shape[0], len(top_groups)))
msg = 'Bot breakout count:     {} with {} groups.\n'
print(msg.format(bottom_breakouts.shape[0], len(bottom_groups)))
print(analysis)




'''

for i in top_groups:
    plt.figure()
    closings = candles.loc[i - window: i, 'midclose'].values
    outcome_interval= candles.loc[i: i + search_interval, 'midclose'].values
    # Fetch channel transformation on window.  Append to results
    c = channel(closings)
    print(i)
    c.plot_closings(outcome_interval)
    plt.show()
    
    
'''













"""
# Add analytical results back onto results df
############################################################################### 
# add groups to results for plotting locations of top / bottom breakouts
results['top_group']        = np.zeros((results.shape[0]))
results['bottom_group']     = np.zeros((results.shape[0]))
results['top_breakouts']    = np.zeros((results.shape[0]))
results['bottom_breakouts'] = np.zeros((results.shape[0]))
results.loc[top_groups, 'top_group']              = .75
results.loc[top_breakouts, 'top_breakouts']       = .8
results.loc[bottom_groups, 'bottom_group']        = .25
results.loc[bottom_breakouts, 'bottom_breakouts'] = .3
"""


