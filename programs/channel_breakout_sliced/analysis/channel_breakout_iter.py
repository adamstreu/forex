import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os; os.chdir('/forex')
from libraries.transformations import get_groups
from libraries.oanda           import get_candles
from classes.channel           import Channel
from libraries.outcomes        import outcomes

'''
to do:
    finish deploymenty (order placement)
    get one or two currencies going
    
    geta  follow system for each that allows me to plot a group by ex scatter
    
'''


def get_results_bars(candles, window, search_interval):
    '''    
    This function has been verified with the channel model.
    Results and outcomes collected by it seem to return the correct dfs.
    '''    
    # Instantiation
    results = []
    long = []
    short = []
    for i in range(window, candles.shape[0] - search_interval):
        # Print progress.
#        if i % 10000 == 0:  
#            print('Percent complete: {:.2f}'.format(i / candles.shape[0]))
        # Prepare Slice candles for channel and outcome_interval
        closings = candles.loc[i - window: i, 'midclose'].values
        # Fetch channel transformation on window.  Append to results
        channel = Channel(closings)
        results.append([i,
                        channel.channel_slope,
                        channel.closings_slope,
                        channel.closing_position,
                        channel.channel_range,
                        channel.largest_spike,
                        channel.largest_spike_5,
                        channel.within_range
                        ])
        # Get Outcomes
        average_channel_distance = ((channel.closings_c7[-1] \
                                 - channel.closings_c1[-1]) / 6)
        distance = np.arange(1, 21) * average_channel_distance
        outs = outcomes(candles, i, search_interval, distance )
        long.append([i] + outs['long_target'] + outs['long_loss'])
        short.append([i] + outs['short_target'] + outs['short_loss'])
    # Collect all window results into dataframe
    results_columns = ['location',
                       'channel_slope', 
                       'closings_slope',
                       'channel_closing_position',
                       'channel_range',
                       'largest_spike',
                       'largest_spike_5',
                       'within_range'
                       ]
    results = pd.DataFrame(np.array(results), columns = results_columns)
    results = results.set_index('location', drop=True)
    results.index = results.index.astype(int)
    # Assemble long and short into dataframe and return
    target_columns = []
    loss_columns = []
    for i in range(int((len(long[0]) - 1) / 2)):
       target_columns.append('t'+str(i + 1))
       loss_columns.append('l' + str(i + 1))
    columns = ['location'] + target_columns + loss_columns
    long = pd.DataFrame(np.array(long), columns=columns)
    long = long.set_index('location', drop=True)
    long.index = long.index.astype(int)
    short = pd.DataFrame(np.array(short), columns=columns)
    short = short.set_index('location', drop=True)
    short.index = short.index.astype(int)    
    return results, long, short


def analyze_results(top_or_bottom, results, outcomes, _filter, group_interval, 
                    use_groupings, position_filter = 100, range_filter = -1, 
                    filter_on_slope=False):
    ''' More or less verified '''
    df = []
    res = results.copy()
    if top_or_bottom == 'top':
        breakouts = res[(res.channel_closing_position > _filter)].index
    elif top_or_bottom == 'bottom':
        breakouts = res[(res.channel_closing_position < _filter)].index
    # Get only first instance in a group of breakouts (defined on interval)
    if use_groupings:
        breakouts = get_groups(breakouts, group_interval)
    # for all combinations of up down outcome pairs
    for t in outcomes.filter(regex='t').columns :
        for l in outcomes.filter(regex='l').columns:
            # Assess bottom breakout outcomes on columns pair
            breakout_results = outcomes.loc[breakouts].copy()
            wins = (breakout_results[t] < breakout_results[l]).mean()
            # Colect results into DataFrame
            df.append([t, 
                       l, 
                       wins
                       ])
    # Assemble analysis into dataframes
    columns = ['target',
               'loss', 
               'wins']
    df = pd.DataFrame(np.array(df), columns=columns)
    df = df.apply(pd.to_numeric, errors='ignore')
    # Calculate expected values
    df['ex']    = (df.wins * df.target.str[1:].astype(int)) \
                - (df.loss.str[1:].astype(int)  * (1 - df.wins))
    df['total'] = df['ex'] * breakouts.shape[0]
    return df, breakouts


if __name__ == '__main__':
    
    
    # Parameters
    ###########################################################################
    # iterations 
    iterations         = 100
    top_filter         = np.linspace(.5, 4, iterations)
    bottom_filter      = np.linspace(-3, .5, iterations) 
    filter_slice_width = top_filter[1] - top_filter[0]
    # Filters 
    position_filter    = 30
    range_filter       = [0, 1]
    closings_slope     = [-1, 1]
    # Group outcomes 
    use_groups         = True
    group_interval     = 250
    # Analysis
    column_to_track    = 'total'
    
    granularity     = 'M1'
    _from           = '2016-01-01T00:00:00Z'
    _to             = '2018-01-01T00:00:00Z'
    # Windows 
    window          = 500
    search_interval = window * 5
    
    instruments = ['EUR_USD', 'EUR_AUD', 'AUD_CAD', 'EUR_CHF',
                   'EUR_GBP', 'GBP_CHF', 'GBP_USD', 'NZD_USD', 'USD_CAD',
                   'USD_CHF', 'EUR_NZD', 'EUR_SGD', 'EUR_CAD', 'USD_SGD', 
                   'GBP_AUD', 'AUD_USD', 'GBP_NZD']
    
    for instrument in instruments:
        print('/n/n-------instrument-------------'''.format(instrument))
        
    
        # Grab candles and calculate windows (results and bars)
        ###########################################################################
        # Fetch candlesr
        candles = get_candles(instrument, granularity, _from, _to)
        # Fetch results and outcomes on windows
        results, long, short = get_results_bars(candles, 
                                                window, 
                                                search_interval)
        # Iterate breakouts (seperate on top and bottom)
        ###########################################################################
        ''' Verified.  Everything seems to be working / collecting properly '''
        # Initializtions
        bottom_breakouts = []
        top_breakouts = []   
        for i in range(iterations):
    #        if i % 20 == 0: print(i)
            # Collect analysis on filters,best results for both top and bottom.
            analysis = analyze_results('bottom', 
                                   results, 
                                   long,  
                                   bottom_filter[i], 
                                   group_interval=group_interval,
                                   use_groupings=use_groups,
                                   position_filter = position_filter, 
                                   range_filter = range_filter, 
                                   )
            placements = analysis[1]
            analysis = analysis[0]
            # Original version - Keep.  Append data for the follow series
            bottom_breakouts.append(analysis[column_to_track].values.tolist() \
                                    + [int(placements.shape[0])])
            analysis = analyze_results('top', 
                                       results, 
                                       short,  
                                       top_filter[i], 
                                       group_interval=group_interval,
                                       use_groupings=use_groups,
                                       position_filter = position_filter, 
                                       range_filter = range_filter, 
                                       )
            placements = analysis[1]
            analysis = analysis[0]
            # Original version - Keep.  Append data for the follow series
            top_breakouts.append(analysis[column_to_track].values.tolist() \
                                        + [int(placements.shape[0])])
        # Assemble dataframe barsfor follows
        columns= []
        for each in analysis.loc[:, ['target', 'loss']].values.tolist():
            columns.append(str(each[0]) + str(each[1])) 
        columns.append('groups')
        bottom_breakouts       = pd.DataFrame(np.array(bottom_breakouts), 
                                              columns=columns)
        bottom_breakouts.index = bottom_filter
        top_breakouts          = pd.DataFrame(np.array(top_breakouts), 
                                              columns=columns)
        top_breakouts.index    = top_filter
        # REmove group count and Get best columns
        top_breakouts_groups    = top_breakouts.pop('groups')
        bottom_breakouts_groups = bottom_breakouts.pop('groups')
        bottom_target           = bottom_breakouts.max().idxmax()[: bottom_breakouts.max().idxmax().find('l')]
        bottom_loss             = bottom_breakouts.max().idxmax()[bottom_breakouts.max().idxmax().find('l'):]
        bottom_filter_final     = bottom_breakouts[bottom_target + bottom_loss].idxmax()
        top_target              = top_breakouts.max().idxmax()[: top_breakouts.max().idxmax().find('l')]
        top_loss                = top_breakouts.max().idxmax()[top_breakouts.max().idxmax().find('l'):]
        top_filter_final        = top_breakouts[top_target + top_loss].idxmax()
    
    
     
        # Get bars and channel range and breakouts for final filters
        ###########################################################################
        top_analysis = analyze_results('top', 
                                       results, 
                                       short,  
                                       top_filter_final, 
                                       group_interval=group_interval,
                                       use_groupings=use_groups,
                                       position_filter = position_filter, 
                                       range_filter = range_filter, 
                                       )
        top_breakout_index = top_analysis[1]
        top_analysis = top_analysis[0] 
        bottom_analysis = analyze_results('bottom', 
                                          results, 
                                          long,  
                                          bottom_filter_final, 
                                          group_interval=group_interval,
                                          use_groupings=use_groups,
                                          position_filter = position_filter, 
                                          range_filter = range_filter, 
                                          )
        bottom_breakout_index = bottom_analysis[1]
        bottom_analysis = bottom_analysis[0] 
        
        # Plot
        ###########################################################################
        # Plot this beautiful arrangement
        combined = top_breakouts.append(bottom_breakouts)[top_breakouts.append(bottom_breakouts) > 0]
        combined.iloc[: ,:-1].plot(figsize=(14,5), title='Follows')#, ylim=0)plt.show()
        combined.iloc[:, -1].plot(color='black')
        # Print
        ###########################################################################
        # Print Parameters just for a reminder 
        print()
        print('Using Groups: {}'.format(use_groups))
        print('start:        {}'.format(str(candles.head(1).timestamp)))
        # Top
        print('\n--- Top Breakouts--- ')
        print('Target:    {}'.format(top_target))
        print('Loss:      {}'.format(top_loss))
        print('Filter:    {}'.format(top_filter_final))
        print('Win %:     {}'.format(top_analysis.loc[top_analysis.total.idxmax()].wins))
        print('Ex. value: {}'.format(top_analysis.loc[top_analysis.total.idxmax()].ex))
        print('Total:     {}'.format(top_analysis.loc[top_analysis.total.idxmax()].total))
        print('Bars:      {}'.format(int(long[[top_target, top_loss]].min(axis=1).mean())))
        print('Breakouts: {}'.format(top_breakout_index.shape[0]))
        # Bottom
        print('\n--- Bottom Breakouts--- ')
        print('Target:    {}'.format(bottom_target))
        print('Loss:      {}'.format(bottom_loss))
        print('Filter:    {}'.format(bottom_filter_final))
        print('Win %:     {}'.format(bottom_analysis.loc[bottom_analysis.total.idxmax()].wins))
        print('Ex. value: {}'.format(bottom_analysis.loc[bottom_analysis.total.idxmax()].ex))
        print('Total:     {}'.format(bottom_analysis.loc[bottom_analysis.total.idxmax()].total))
        print('Bars:      {}'.format(int(long[[bottom_target, bottom_loss]].min(axis=1).mean())))
        print('Breakouts: {}'.format(bottom_breakout_index.shape[0]))    
        
    