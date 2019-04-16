import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from functools import reduce
import os; os.chdir('/northbend')
from libraries.oanda           import get_candles
from classes.channel           import Channel
from libraries.outcomes        import outcomes

import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC as svc


'''
Method:
    Call window
    Create Channel
    Gather Outcomes
    Analyse 
        large step
    Make Predictions
    return to results.  are there any further correlations? 
    graph distribution of predictions
    calculate return inclusing spread, commisions, etc



Need to figure out actual bididing strategy:
    jsut do the top instruments (50:1 leverage)
    Need to choose enough groups to make enough return on money.
    
    



analysis paramters:
   slice width 
   
outcomes parameters:
    fixed or relative to channel
    midclose or bid/ask scheme
    how large to go (20 is too large)
    

Filter results on:
    days

    
    
filter channel statistics:
    what side is slice hitting on
    closing position is at a peak
    go only in direction of slope
    


To do next on channel:
    Can we get support lines?
    Evaluate channel construction
    Add more stats
    what kind of distribution is channel.
        what best models it.
        can we classify it into a few groups
        bimodal, etc
        how well can we fit a sin wave to it, a normal, 
    random walk over time
        fixed steps by distance.  create new





I am trying to find a range of channel defined closing values to place bids at.
I want to filter past those closing values to try and seperate predictions.
So far nothing is making any predictions clearer.  I don;t have a proper model 
of the channel yet.  
    

    WHAT?
    
  - usa:
    - US30_USD
    - SPX500_USD
    - NAS100_USD
  - commodities:
    - NATGAS_USD
    - BCO_USD
    - XAU_USD
  - bonds:
    - DE10YB_EUR
    - USB30Y_USD
    - USB10Y_USD
    
    
    
'''    
    
def get_results_bars(candles, window, search_interval, peaks_window, distance):
    # Instantiation
    results     = []
    long_target = []
    long_loss   = []
    short_target = []
    short_loss  = []
    peaks = []
    start = max(window, peaks_window)
    for i in range(start, candles.shape[0] - search_interval):
        # Print progress.
        if i % 10000 == 0:  
            print('Percent complete: {:.2f}'.format(i / candles.shape[0]))
        
        # Fetch channel transformation on window.  Append to results
        channel = Channel(candles, i, window)
        results.append([i,
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
                        channel.closings[-1]
                        ])
    
        # Get Peaks
        peaks_collection = channel.get_supports(peaks_window)
        for peak in peaks_collection:
            peaks.append([i, peak])
            
        # Set distance for outcome
        if type(distance) == 'str':
            distance = np.arange(1, 11) * (channel.channel_range / 6)
        # Get long outcomes
        outs = outcomes('long', candles, i, search_interval, distance, False)
        long_target.append([i] + outs['target'])
        long_loss.append([i] + outs['loss'])
        # get short outcomes
        outs = outcomes('short', candles, i, search_interval, distance, False)
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
                       'closing_value'
                       ]
    results      = pd.DataFrame(np.array(results), columns = results_columns)
    long_target  = pd.DataFrame(np.array(long_target))
    long_loss    = pd.DataFrame(np.array(long_loss))
    short_target = pd.DataFrame(np.array(short_target))
    short_loss   = pd.DataFrame(np.array(short_loss))
    peaks        = pd.DataFrame(np.array(peaks), columns=['location', 'peaks'])
    # Set indexes
    results      = results.set_index('location', drop=True)
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
    results.index      = results.index.astype(int)
    long_target.index  = long_target.index.astype(int)
    long_loss.index    = long_loss.index.astype(int)
    short_target.index = short_target.index.astype(int)
    short_loss.index   = short_loss.index.astype(int)
    # Return
    return {'results': results, 
            'long_target': long_target,
            'long_loss': long_loss, 
            'short_target': short_target, 
            'short_loss': short_loss, 
            'peaks': peaks}




def analyze_iterations_over_slices(candles, filters, results, 
                                   target, loss, distance):
    ex               = []
    groups           = []
    bars             = []
    throughput       = []
    wins             = []
    total            = []
    indexes          = []
    purchased_shares = 8000
    purchase_price   = candles.midclose.mean()
    
    for i in range(filters.shape[0] - 1):
        tmp_ex = []
        tmp_bars = []
        tmp_throughput = []
        tmp_groups = []
        tmp_wins = []
        tmp_total = []
        lower = results.channel_closing_position > filters[i]
        upper = results.channel_closing_position < filters[i + 1]
        position_index = results.loc[lower & upper].index
        indexes.append(position_index)
        for t in target.columns:
            for l in loss.columns:
                spread = candles.loc[position_index, 'spread'].values \
                       * purchased_shares                
                outcomes = (target.loc[position_index, t] \
                         < loss.loc[position_index, l]).values.astype(float)
                if type(distance) == str:
                    distance_r = results.loc[position_index, 
                                           'channel_range'].values / 6
                    outcomes[outcomes == 0] = -1 * distance_r[outcomes == 0]
                    outcomes[outcomes == 1] = distance_r[outcomes == 1] * t/l
                    outcomes *= purchased_shares
                    outcomes -= spread
                else:
                    outcomes[outcomes == 0] = -1 * distance[l - 1] * purchased_shares
                    outcomes[outcomes == 1] = distance[t - 1] * purchased_shares
                    outcomes -= spread
                win_perc = (outcomes > 0).mean()
                expected = outcomes.mean() #(win_perc *  (ratio + 1)) - 1
                local_bars = np.minimum(target.loc[position_index,t].values, 
                                        loss.loc[position_index, 
                                                 l].values).mean()   
                # Append values
                if outcomes.shape[0] != 0:
                    tmp_ex.append(expected)
                    tmp_bars.append(local_bars)
                    tmp_wins.append(win_perc)
                    tmp_throughput.append(expected / local_bars)
                    tmp_groups.append(position_index.shape[0])
                    tmp_total.append(outcomes.cumsum()[-1])
                else:
                    tmp_ex.append(0)
                    tmp_bars.append(0)
                    tmp_wins.append(0)
                    tmp_throughput.append(0)
                    tmp_groups.append(0)
                    tmp_total.append(0)
        ex.append(tmp_ex)
        throughput.append(tmp_throughput)
        bars.append(tmp_bars)
        groups.append(tmp_groups)
        wins.append(tmp_wins)
        total.append(tmp_total)
    # Create Columns
    columns = []
    for t in target.columns:
        for l in loss.columns:
            columns.append('T' + str(t)+ '_L' + str(l))
    # Create DataFrames and set index
    ex         = pd.DataFrame(np.array(ex), columns=columns)
    throughput = pd.DataFrame(np.array(throughput), columns=columns)
    bars       = pd.DataFrame(np.array(bars), columns=columns)
    groups     = pd.DataFrame(np.array(groups), columns=columns)
    wins       = pd.DataFrame(np.array(wins), columns=columns)
    total      = pd.DataFrame(np.array(total), columns=columns)
    indexes    = pd.DataFrame(np.array(indexes))
    # Set Indexes to filter
    ex         = ex.set_index(filters[:-1], drop=True)
    throughput = throughput.set_index(filters[:-1], drop=True)
    bars       = bars.set_index(filters[:-1], drop=True)
    groups     = groups.set_index(filters[:-1], drop=True)
    wins       = wins.set_index(filters[:-1], drop=True)
    total      = total.set_index(filters[:-1], drop=True)
    indexes    = indexes.set_index(filters[:-1], drop=True)
    # Return
    return {'ex': ex, 
            'throughput': throughput,
            'bars': bars,
            'groups': groups,
            'wins': wins,
            'total': total,
            'indexes': indexes}
    
    
def export_to_tableau(export, name):
    totab = []
    for i in export['ex'].index:
        for c in export['ex'].columns:
            tmp = [i, c]
            for df in [export['ex'], 
                       export['throughput'], 
                       export['bars'], 
                       export['groups'], 
                       export['wins'], 
                       export['total']]:
                tmp.append(df.loc[i, c])
            totab.append(tmp)
    columns = ['filter', 
              'column', 
              'ex', 
              'throughput',
              'bars',
              'groups',
              'wins',
              'total']
    totab = pd.DataFrame(np.array(totab), columns=columns)
    totab = totab.set_index('filter', drop=True)
    totab = totab.apply(pd.to_numeric, errors='ignore')
    totab.to_csv('/Users/user/Desktop/' + str(name) + '_analysis.csv')
    '''    
    totab.to_csv('/Users/user/Desktop/m1_w500_y1_top/' + str(instrument) + 'totab.csv')
    up_indexes.to_csv('/Users/user/Desktop/m1_w500_y1_top/' + str(instrument) + '_indexes.csv')
    results.to_csv('/Users/user/Desktop/m1_w500_y1_top/' + str(instrument) + '_results.csv')
    up.to_csv('/Users/user/Desktop/m1_w500_y1_top/' + str(instrument) + '_up.csv')
    down.to_csv('/Users/user/Desktop/m1_w500_y1_top/' + str(instrument) + '_down.csv')
    '''
    return None
    

def plot_outcome_distribution(results, top_outcomes, bottom_outcomes):
    a = pd.DataFrame(results.index.values)
    a['top_outcomes'] = -1
    a['bottom_outcomes'] = -1
    a.set_index(0, inplace=True, drop=True)
    a.index = a.index.rename('location')
    a.loc[top_index, 'top_outcomes']       = top_outcomes.outcomes
    a.loc[bottom_index, 'bottom_outcomes'] = bottom_outcomes.outcomes
    a.loc[a[a.top_outcomes == 0].index, 'top_outcomes']        = .8
    a.loc[a[a.bottom_outcomes == 1].index, 'bottom_outcomes']  = .5
    a.loc[a[a.bottom_outcomes == 0].index, 'bottom_outcomes']  = .3
    a.loc[a[a.top_outcomes == -1].index, 'top_outcomes']       = 0
    a.loc[a[a.bottom_outcomes == -1].index, 'bottom_outcomes'] = 0
    a.plot(figsize=(14, 3), style='o')
    return None


def quick_ml(df, results, filter_slice, target_df, loss_df, target, loss):
    # Create x and y
    ind = df['indexes'].loc[filter_slice][0].values
    x = results.loc[ind].copy()
    y = (target_df.loc[ind, target] < loss_df.loc[ind, loss]).astype(int)
    # create sets
    x.reset_index(inplace=True, drop=True)
    y.index = x.index
    row = int(x.shape[0] * .8)
    x_train = x.loc[: row]
    x_test  = x.loc[row :]
    y_train = y.loc[: row]
    y_test  = y.loc[row :]
    # scale values
    scaler  = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)
    # weuights ? 
    weights = sklearn.utils.class_weight.compute_class_weight('balanced', 
                                                              np.array([0,1]), 
                                                              y_train)
    # model
    logreg = LogisticRegression(class_weight={0: weights[0], 1: weights[1]})
    logreg = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                           hidden_layer_sizes=(500, 4), random_state=1)
    logreg.fit(x_train, y_train)
    predictions = logreg.predict(x_test)
    print(classification_report(y_test, predictions))
    # Print
    logreg = svc()
    logreg = KNeighborsClassifier(n_neighbors=2)
    logreg = DecisionTreeClassifier(random_state=0)
    return None



###############################################################################
# Request and Calculate candles, Results, outcomes, peaks
###############################################################################
    
if False:
    # Candles
    instrument      = 'EUR_USD'
    granularity     = 'M1'
    _from           = '2017-01-01T00:00:00Z'
    _to             = '2018-01-01T00:00:00Z'
    # Outcomes
    distance        = 'relative'# np.arange(1, 11) * .0005 # or 'relative'
    # Windows 
    window          = 1500
    search_interval = window * 2
    # Peaks
    peaks_window    = 2000
    
    # Fetch candles, Results and Outcomes
    candles      = get_candles(instrument, granularity, _from, _to)
    get_results  = get_results_bars(candles, window, search_interval, 
                                    peaks_window, distance)
    long_target  = get_results['long_target']
    long_loss    = get_results['long_loss']
    short_target = get_results['short_target']
    short_loss   = get_results['short_loss']
    peaks        = get_results['peaks']
    results      = get_results['results']

    

###############################################################################
# Filters on results 
###############################################################################
if False:
    # Start with fresh material
    long_target  = get_results['long_target']
    long_loss    = get_results['long_loss']
    short_target = get_results['short_target']
    short_loss   = get_results['short_loss']
    peaks        = get_results['peaks']
    results      = get_results['results']
    # Filter Results Immediately on known parameters
    within_range_filter  = results[results.within_range > .75].index.values
    channel_range_filter = results[results.channel_range > 0].index.values
    volume_high_filter   = candles[candles.volume < 10000].index.values
    spread_filter        = candles[candles.spread < .0003].index.values
    # Apply intersection of filters to all dataframes
    keep_index   = reduce(np.intersect1d, (within_range_filter, 
                                           channel_range_filter, 
                                           volume_high_filter,
                                           spread_filter))
    # Apply to df's
    results      = results.loc[keep_index]
    long_target  = long_target.loc[keep_index]
    long_loss    = long_loss.loc[keep_index]
    short_target = short_target.loc[keep_index]
    short_loss   = short_loss.loc[keep_index]
    peaks        = peaks.loc[keep_index]
    # Print results
    print('Items Filtered out: {}'.format(get_results['results'].shape[0] \
                                          - keep_index.shape[0]))
    print('As percentage:      {}'.format((1 - (keep_index.shape[0] \
                                          / get_results['results'].shape[0]))))




###############################################################################
# Analyze Bottom and Top breakouts ( using long and short positions ) 
######################################### ######################################
'''
Both Total and E[x] are in real dollars.
This might not be ideal (at least for E[x]),  as it does not 
neccesarily reflect the relationship betweem risk and return.

Spread is in real dollars, and we are using spread so I am keeping everything
in the same units.
'''
slice_width   = .1
top_slices    = np.round(np.arange(.5, 5, slice_width), 4)
bottom_slices = np.round(np.arange(-5, .5, slice_width), 4)
top           = analyze_iterations_over_slices(candles, 
                                               top_slices, 
                                               results, 
                                               short_target, 
                                               short_loss,
                                               distance)
bottom        = analyze_iterations_over_slices(candles, 
                                               bottom_slices, 
                                               results, 
                                               long_target, 
                                               long_loss,
                                               distance)
# Esport for graphical analysis
export_to_tableau(top,'top')
export_to_tableau(bottom, 'bottom')



###############################################################################
# Collect best top and bottom filters, targets, losses
###############################################################################
# Filter results by following parameters

wins_low_filter         = .4
minimum_groups_per_year = 15
maximum_groups_per_year = 100
# Set Group filter.
candle_years            = (pd.to_datetime(candles.timestamp.values[-1]) \
                        - pd.to_datetime(candles.timestamp.values[0])).days
candle_years            /= 365
groups_high_filter      = int(maximum_groups_per_year * candle_years)
groups_low_filter       = int(minimum_groups_per_year * candle_years)





"""

###############################################################################
# Collect variables for winning Parameters
###############################################################################
# Top winning Parameters
top_filter           = 1.5
top_target_column    = 4
top_loss_column      = 1
# bottom winning Parameters
bottom_filter        = -2.6
bottom_target_column = 10
bottom_loss_column   = 6
# Set indexes
top_index     = top['indexes'].loc[top_filter][0].values
bottom_index  = bottom['indexes'].loc[bottom_filter][0].values
# Create winning DataFrames
top_outcomes    = results.loc[top_index]
bottom_outcomes = results.loc[bottom_index]
top_outcomes['outcomes']    = (short_target.loc[top_index, 
                               top_target_column] \
                            < short_loss.loc[top_index, 
                                             top_loss_column]).astype(int)
bottom_outcomes['outcomes'] = (long_target.loc[bottom_index, 
                               bottom_target_column] \
                            < long_loss.loc[bottom_index, 
                                            bottom_loss_column]).astype(int)
# Only keep till no longer needed

# Export Outcomes
top_outcomes.to_csv('/Users/user/Desktop/top_outcomes.csv')
bottom_outcomes.to_csv('/Users/user/Desktop/bottom_outcomes.csv')






###############################################################################
# Plot Distribution and Cumulative Returns
###############################################################################
# Plot outcome distribution
plot_outcome_distribution(results, top_outcomes, bottom_outcomes)
plt.figure(figsize=(14,2))
plt.plot(candles.midclose.values)
# Plot top and bottom outcome cumulative returns



###############################################################################
# Real quick ML stuff
###############################################################################
print('\nTop:')
quick_ml(top, results, top_filter, short_target, short_loss, 
         top_target_column, top_loss_column)
print('\nBottom:')
quick_ml(bottom, results, bottom_filter, long_target, long_loss,    
         bottom_target_column, bottom_loss_column)









"""










'''
###############################################################################
# Peaks analysis (top for now - at position - actign as repulsive force)
###############################################################################
# Get loc of peaks within epsilon
avg_range = results.channel_range.mean() / 6
for ind in peaks.index.values:
    if abs(candles.loc[ind, 'midclose'] - peaks.loc[ind, 'peaks']) < avg_range / 2:
        peaks.loc[ind, 'top_breakout'] = candles.loc[ind, 'midclose']
peaks_index = peaks[peaks.top_breakout > 0].index.values



# right away - is there any line up with outcomes and filter?
for t in range(1, len(short_target.columns) + 1):
    for l in range(1, len(short_loss.columns) + 1):
        out = (short_target.loc[peaks_index, t] < short_loss.loc[peaks_index, l]).mean()
        without = (short_target[t] < short_loss[l]).mean()
        print('{}, {}: {:.2f}\t{:.2f}'.format(t, l, out, without))        


results['peaks'] = 0
results.loc[peaks_index, 'peaks'] = peaks.loc[peaks_index, 'peaks']

# How many each filter has of peaks
for i in top['indexes'].index.values:
    print(str(i) +'   '+ str(np.intersect1d(top['indexes'].loc[i][0].values, peaks_index).shape[0])
            + '    ' + str(top['indexes'].loc[i][0].values.shape[0]) )
'''









