'''

TONIGHT - CAN WE JUST DO THEN RELEASE SOMETHING SIMILAR FOR RATIOS ALONE.
            COULD WORK ON GETTING SOMETHING THAT WORKS FOR SMALLER INTERVALS
            GO BACKWARDS AND USE ADDITIONAL INDICATORS.

        GET SOMETHING WE CAN DEPLOY....
        
'''


# =============================================================================
# Imports
# =============================================================================
if 0:     
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from multiprocessing import Process
    from multiprocessing import Manager
    import os; os.chdir('/northbend')
    from libraries.transformations import get_groups
    from libraries.outcomes import get_outcomes_multi
    from libraries.outcomes import plot_outcomes
    from libraries.currency_universe import get_currencies
    from libraries.oanda import get_candles
    # Multiprocessing wrapped
    from libraries.waves import waves_wrapper
    from libraries.waves import get_rolling_rank
    # Multiprocessing unwrapped
    from libraries.waves import get_rolling_mean_pos_std
    from libraries.waves import get_channel_mean_pos_std
    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15)
    np.warnings.filterwarnings('ignore')   
    
    

# =============================================================================
# Call currency universe.  Multiprocessing Units.
# =============================================================================
if 0:       
    
    # Parameters    
    pair = 'EUR_USD'
    granularity = 'M1'
    _from = '2015-01-01T00:00:00Z'
    _to   = '2018-01-01T00:00:00Z'
    
    # Get Candles
    candles = get_candles(pair, granularity, _from, _to)
    ratio = candles['midclose']



# =============================================================================
# Get Outcomes
# =============================================================================
if 0:     
    
    targets = np.arange(.0005, .01 , .0015).round(6)
    outcomes = get_outcomes_multi(ratio.values, targets)
    up = outcomes['up']
    down = outcomes['down']
    # Build Results DataFrame
    long = up < down
    short = down < up
    minimums = pd.DataFrame(np.minimum(up.values, down.values), 
                                    columns = up.columns, index = up.index)
    # Print mean minimums per target
    msg = '\nMean Minimums Per Target\n------------------------\n{}\n'
    print(msg.format(minimums[minimums!=ratio.shape[0]].dropna().mean()))


    # Some plotting verification for outcomes
    if False:
        location = 1051492
        plot_outcomes(location, ratio, up, down, targets, minimums) 
        


# =============================================================================
# Get Indicators
# =============================================================================
if 0:      

    # Parameters
    windows = np.array([15, 30, 60, 90, 120, 240, 480, 960])
    bins = 60
    # Channel Indicators
    channel_stats = get_channel_mean_pos_std(ratio.values, windows)
    channel_std   = channel_stats['std']
    channel_pos   = channel_stats['pos']
    channel_mean  = channel_stats['mean']
    slopes        = channel_stats['slope']
    # Rolling mean, pos, std and Slope
    rolling_stats = get_rolling_mean_pos_std(ratio, windows)
    rolling_std   = rolling_stats['std']
    rolling_pos   = rolling_stats['pos']
    rolling_mean  = rolling_stats['mean']
    # Rolling Channel Difference
    roll_chann_diff_mean =  rolling_mean - channel_mean
    roll_chann_diff_pos  =  rolling_pos - channel_pos
    roll_chann_diff_std  =  rolling_std - channel_std



# =============================================================================
# Score Binned Indicators Frequency
# =============================================================================
if 0:          

    def get_indicator_bins_frequency(indicator_df, outcomes):
        coll = []
        indexes = []
        for window in indicator_df.columns.tolist():
            indicator = indicator_df.loc[window:, window].values
            arange = np.arange(indicator_df.shape[0])
            hist = np.histogram(indicator, bins=bins)
            for i in range(bins):
                start = hist[1][i]
                # end = hist[1][i+1]
                cond1 = indicator >= start
                cond2 = True # indicator <= end
                index = cond1 & cond2
                index = np.insert(index, 0, [False] * window)
                group_count = get_groups(arange[index], 100).shape[0]
                shape = index.sum()
                long_mean = outcomes.loc[index].mean().tolist()
                for each in list(zip(outcomes.columns.tolist(), long_mean)):
                    coll.append([window, each[0], i, shape, group_count, each[1]])
                    indexes.append(index)
                    
        # Binned Freq Results Analysis DF
        columns = ['window', 'target', 'bin', 'shape', 'groups', 'perc']
        coll = np.array(coll)
        results = pd.DataFrame(coll.T, columns).T
        # Export and return
        results.to_csv('/Users/user/Desktop/ratio.csv')
        # Set Location and final currency index]
        return {
                'results': results,
                'indexes': indexes
                }
        
        
    ind_score = get_indicator_bins_frequency(slopes, long)
    slope_scored = ind_score['results']
    slope_indexes = ind_score['indexes']    
    
    ind_score = get_indicator_bins_frequency(roll_chann_diff_pos, long)
    pos_diff_scored = ind_score['results']
    pos_diff_indexes = ind_score['indexes']    

    '''
    for i in range(45, len(slope_indexes)):
        for j in range(45, len(slope_indexes)):   
    '''
            
    indexes_to_use = slope_scored.loc[(slope_scored.target == 0.002) & (slope_scored.window == 90)].index.values
    combined = []
    for i in indexes_to_use:
        for j in indexes_to_use:
            print(i, j)
            intersection = slope_indexes[i] & pos_diff_indexes[j]
            outs = long.loc[intersection].mean().tolist()
            combined.append([i, j, intersection.sum()] + outs)
    columns = ['slope', 'pos_diff', 'shape'] + targets.tolist()
    combined = pd.DataFrame(combined, columns = columns)
    
    
    
    

    
    
    
    
    
    if False: 
        location = 233
        plt.plot(indexes[location], 'o')
        print(long.loc[indexes[location]].mean())
        print(indexes[location].sum())
        print(results.loc[location])    


# =============================================================================
# Export
# =============================================================================
if False:
    
    ratio
    
    
    results.to_csv('/Users/user/Desktop/ratio.csv')
    ratio.to_pickle('/Users/user/Desktop/3_year_ratio_slope_outcomes/ration.pkl')
    channel_std.to_pickle('/Users/user/Desktop/3_year_ratio_slope_outcomes/channel_std.pkl')
    channel_pos.to_pickle('/Users/user/Desktop/3_year_ratio_slope_outcomes/channel_pos.pkl')
    channel_mean.to_pickle('/Users/user/Desktop/3_year_ratio_slope_outcomes/channel_mean.pkl')
    slopes.to_pickle('/Users/user/Desktop/3_year_ratio_slope_outcomes/slopes.pkl')
    up.to_pickle('/Users/user/Desktop/3_year_ratio_slope_outcomes/up.pkl')
    down.to_pickle('/Users/user/Desktop/3_year_ratio_slope_outcomes/down.pkl')
    candles.to_pickle('/Users/user/Desktop/3_year_ratio_slope_outcomes/candles.pkl')



















    
    
    
