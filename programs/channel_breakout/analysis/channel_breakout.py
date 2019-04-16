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

        
    
    
    
        Goal:
      x  Ungrouped and sliced
      x  Maybe Only consider channels that fit the norm
      x  Get the ex or throughput value high as possible while still keeping enough groups.
      x  Keep throughput high.  Don't trust my 20:1 results much.
      x  Group what is left (before and after ml to see if spread is ok)
      
   

  
Analysis:
        
             ml did nothing (see below)        
        
    >>>>>    In fact, we will say right now that there is (for now)
             no better filtering of stats to be done.
        
    >>>>>    Throughput on u1d6 is excellent.
             And wins are newarly 1/2        
    
    

    


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
        if i % 10000 == 0:  
            print('Percent complete: {:.2f}'.format(i / candles.shape[0]))
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
    top_filter         = np.linspace(.5, 20, iterations)
    bottom_filter      = np.linspace(-20, .5, iterations) 
    filter_slice_width = top_filter[1] - top_filter[0]
    # Filters 
    position_filter    = 30
    range_filter       = [0, 1]
    closings_slope     = [-1, 1]
    # Group outcomes 
    use_groups         = False
    group_interval     = 80
    # Analysis
    column_to_track    = 'ex'
    

    # Grab candles and calculate windows (results and bars)
    ###########################################################################
    if False:
        # Instrument
        instrument      = 'EUR_USD'
        granularity     = 'M5'
        _from           = '2014-01-01T00:00:00Z'
        _to             = '2018-01-01T00:00:00Z'
        # Windows 
        window          = 125
        search_interval = window * 4
        # Fetch candlesr
        candles = get_candles(instrument, granularity, _from, _to)
        # Fetch results and outcomes on windows
        results, long, short = get_results_bars(candles, 
                                                window, 
                                                search_interval)


    # Iterate breakouts (seperate on top and bottom)
    ###########################################################################
    bottom_breakouts = []
    top_breakouts = []
    for i in range(iterations):
        if i % 20 == 0: print(i)
        # Bottom Series
        try:
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
            bottom_breakouts.append(analysis[column_to_track].values.tolist() \
                                    + [int(placements.shape[0])])
        except Exception as e:
            print('Bottom: {}'.format(e))
            bottom_breakouts.append([bottom_filter[i]] + np.zeros(400).tolist())
        # Top Series
        try:
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
        except Exception as e:
            print('Bottom: {}'.format(e))
            top_breakouts.append([top_filter[i]] + np.zeros(400).tolist())
    # Assemble dataframe bars for follows
    columns= []
    for each in analysis.loc[:, ['target', 'loss']].values.tolist():
        columns.append(str(each[0]) + str(each[1])) 
    columns.append('groups')
    bottom_breakouts       = pd.DataFrame(np.array(bottom_breakouts), 
                                          columns=columns)
    top_breakouts          = pd.DataFrame(np.array(top_breakouts), 
                                          columns=columns)
    bottom_breakouts.index = bottom_filter
    top_breakouts.index    = top_filter
    # Remove group count and Get best columns
    top_breakouts_groups    = top_breakouts.pop('groups')
    bottom_breakouts_groups = bottom_breakouts.pop('groups')
    bottom_target           = bottom_breakouts.max().idxmax()[: bottom_breakouts.max().idxmax().find('l')]
    bottom_loss             = bottom_breakouts.max().idxmax()[bottom_breakouts.max().idxmax().find('l'):]
    bottom_filter_final     = bottom_breakouts[bottom_target + bottom_loss].idxmax()
    top_target              = top_breakouts.max().idxmax()[: top_breakouts.max().idxmax().find('l')]
    top_loss                = top_breakouts.max().idxmax()[top_breakouts.max().idxmax().find('l'):]
    top_filter_final        = top_breakouts[top_target + top_loss].idxmax()

    
    
    # Get bars and channel range and breakouts for final filters
    #########################################################################
    top_analysis = analyze_results('top', 
                                   results, 
                                   short,  
                                   top_filter_final, 
                                   group_interval=group_interval,
                                   use_groupings=use_groups,
                                   position_filter = position_filter, 
                                   range_filter = range_filter)
    top_breakout_index = top_analysis[1]
    top_analysis = top_analysis[0] 
    bottom_analysis = analyze_results('bottom', 
                                      results, 
                                      long,  
                                      bottom_filter_final, 
                                      group_interval=group_interval,
                                      use_groupings=use_groups,
                                      position_filter = position_filter, 
                                      range_filter = range_filter)
    bottom_breakout_index = bottom_analysis[1]
    bottom_analysis = bottom_analysis[0] 
    
    
    # Plot
    ###########################################################################
    # Plot this beautiful arrangement
    combined = top_breakouts.append(bottom_breakouts)[top_breakouts.append(bottom_breakouts) > 0]
    combined.iloc[: ,:-1].plot(figsize=(14,5), title='Follows')#, ylim=0)plt.show()
    # Plot breakouts
    breakouts = pd.DataFrame(results.index)
    breakouts['top'] = 0
    breakouts['bottom'] = 0
    breakouts.loc[top_breakout_index, 'top'] = .4
    breakouts.loc[bottom_breakout_index, 'bottom'] = .2
    breakouts.set_index('location', drop=True, inplace=True)
    breakouts.plot(figsize=(14,8), style='+')
    # Show
    plt.show()    
    
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
    




'''

want to beat

--- Top Breakouts--- 
Target:    t19
Loss:      l1
Filter:    1.1717171717171717
Win %:     0.2157676348547718
Ex. value: 3.3153526970954363
Total:     799.0000000000001
Bars:      128
Breakouts: 241

--- Bottom Breakouts--- 
Target:    t19
Loss:      l1
Filter:    -0.06565656565656575
Win %:     0.20125786163522014
Ex. value: 3.0251572327044025
Total:     962.0
Bars:      128
Breakouts: 318


'''






















'''
    
    


Further analysis (after first is deployed):
    posiiton closout could be sided (coming from a direction), or in a range.
    interested in some more mk attempts after things get better.
    plot win loss dist over time on groups / breakuots on on e (best) filet /d/u
    Collect samples of each du on each iter on each ....
    Evaluate performance of channl construction
    Remove outcomes where not in. (Or say lose after _x_.)
    Do we need to worry about the other two direciton?
    Can I better the reulsts filt3ering on ogth3re parametss?
    better to use higest functioning periodic results (?)
        nah fuck that do way down the line probabily never.
    Now that results and channels are improved, perhaps some windows
        thrown at a ml will stick.    
    Later:
    do not use if there are any large jumps in closing values (throws off 
        any one vlaue _x_ greater than average value
    channels slope needs to be close to 0
    bimodal ? 
    closing position is at a peak
    go only in direction of slope
    how well can we fit a sin wave to it
    well defined trough beween bimodal (sya, almost half ? ? ?)
    Need some stats on how well the channels fit the pattern
    are particulr days good?
    
    
    
    
# To plot a bunch at once for evaluation:
for i in group_index:
    closings = candles.loc[i - window: i, 'midclose'].values
    outcome_interval= candles.loc[i: i + search_interval, 'midclose'].values
    # Fetch channel transformation on window.  Append to results
    c = Channel(closings)
    c.plot_closings(outcome_interval)
    print('\n--- collected ---\n')
    print('i: {}'.format(i))
    print(results.loc[i])
    print(bars.loc[i])
    plt.show()


'''




'''
# A very quick ML thing.
# Using only results from breakouts. (not grouped)

import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC as svc
x = results.loc[top_breakout_index].copy()
y = (short.loc[top_breakout_index, 't9'] < short.loc[top_breakout_index, 'l1']).astype(int)


x.reset_index(inplace=True, drop=True)
y.index = x.index
row = int(x.shape[0] * .8)
x_train = x.loc[: row]
x_test  = x.loc[row :]
y_train = y.loc[: row]
y_test  = y.loc[row :]

scaler  = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

weights = sklearn.utils.class_weight.compute_class_weight('balanced', 
                                                          np.array([0,1]), 
                                                          y_train)

logreg = LogisticRegression(class_weight={0: weights[0], 1: weights[1]})
logreg.fit(x_train, y_train)
predictions = logreg.predict(x_test)
print(classification_report(y_test, predictions))

logreg = svc()
logreg = KNeighborsClassifier(n_neighbors=2)
logreg = DecisionTreeClassifier(random_state=0)
logreg = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                       hidden_layer_sizes=(5, 2), random_state=1)
'''










