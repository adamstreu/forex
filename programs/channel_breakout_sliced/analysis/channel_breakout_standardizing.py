import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import os; os.chdir('/forex')
from libraries.transformations import get_groups
from libraries.oanda           import get_candles
from classes.channel           import Channel
from libraries.outcomes        import simple_outcomes


'''
to do next:
    
    Choose good filter / gran / window for a number of instruments
    
    id slice size ok?
    can we build the
    what side is it hitting range from?
    
    
    
    
    
    


        
        
        
        
        
        
        
        
        
'''

def get_results_bars(candles, window, search_interval):
    # Instantiation
    results = []
    up = []
    down = []
    for i in range(window, candles.shape[0] - search_interval):
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
                        channel.within_range
                        ])
        # Get Outcomes
        distance = np.arange(1, 11) * (channel.channel_range / 6)
        outs = simple_outcomes(candles, i, search_interval, distance )
        up.append([i] + outs['up'])
        down.append([i] + outs['down'])
    # Assemble columns
    results_columns = ['location',
                       'channel_slope', 
                       'closings_slope',
                       'channel_closing_position',
                       'channel_range',
                       'largest_spike',
                       'largest_spike_5',
                       'within_range'
                       ]
    up_columns = ['location']
    down_columns = ['location']
    for i in range(len(up[0]) - 1):
       up_columns.append('u'+str(i + 1))
       down_columns.append('d' + str(i + 1))
    # Assemble Dataframes
    results = pd.DataFrame(np.array(results), columns = results_columns)
    up = pd.DataFrame(np.array(up), columns=up_columns)
    down = pd.DataFrame(np.array(down), columns=down_columns)
    # Correct Indexes
    results = results.set_index('location', drop=True)
    up = up.set_index('location', drop=True)
    down = down.set_index('location', drop=True)
    results.index = results.index.astype(int)
    up.index = up.index.astype(int)    
    down.index = down.index.astype(int)
    # Return
    return results, up, down



if True:
    # Instrument
    instrument      = 'EUR_USD'
    granularity     = 'M1'
    _from           = '2014-06-01T00:00:00Z'
    _to             = '2018-01-01T00:00:00Z'
    # Windows 
    window          = 175
    search_interval = window * 5
    # Fetch candlesr
    candles = get_candles(instrument, granularity, _from, _to)
    # Fetch results and outcomes on windows
    results, up, down = get_results_bars(candles, 
                                            window, 
                                            search_interval)


# This is for top (short positions) only.  It is ungrouped and sliced
iterations         = 100
position_filter    = np.round(np.linspace(0, 10, iterations), 3)
ex = []
groups = []
bars = []
throughput = []
wins = []
total = []
indexes = []
for i in range(position_filter.shape[0] - 1):
    tmp_ex = []
    tmp_bars = []
    tmp_throughput = []
    tmp_groups = []
    tmp_wins = []
    tmp_total = []
    lower = results.channel_closing_position > position_filter[i]
    upper = results.channel_closing_position < position_filter[i + 1]
    position_index = results.loc[lower & upper].index
    indexes.append(position_index)
    for u in up.columns:
        for d in down.columns:
            # Calculate values
            win_perc =  (down.loc[position_index, d] < up.loc[position_index, u]).mean()
            expected = (int(d[1:]) * win_perc) - (int(u[1:]) * (1 - win_perc))
            local_bars = np.minimum(down.loc[position_index, d].values, up.loc[position_index, u].values).mean()
            # Append values
            tmp_ex.append(expected)
            tmp_bars.append(local_bars)
            tmp_wins.append(win_perc)
            tmp_throughput.append(expected / local_bars)
            tmp_groups.append(position_index.shape[0])
            tmp_total.append(expected * position_index.shape[0])
    ex.append(tmp_ex)
    throughput.append(tmp_throughput)
    bars.append(tmp_bars)
    groups.append(tmp_groups)
    wins.append(tmp_wins)
    total.append(tmp_total)
# Create Columns
columns = []
for u in up.columns:
    for d in down.columns:
        columns.append(str(u)+str(d))
# Create DataFrames and set index
ex         = pd.DataFrame(np.array(ex), columns=columns)
throughput = pd.DataFrame(np.array(throughput), columns=columns)
bars       = pd.DataFrame(np.array(bars), columns=columns)
groups     = pd.DataFrame(np.array(groups), columns=columns)
wins       = pd.DataFrame(np.array(wins), columns=columns)
total      = pd.DataFrame(np.array(total), columns=columns)
indexes    = pd.DataFrame(np.array(indexes))
# Stt Indexes
ex         = ex.set_index(        position_filter[:-1], drop=True)
throughput = throughput.set_index(position_filter[:-1], drop=True)
bars       = bars.set_index(      position_filter[:-1], drop=True)
groups     = groups.set_index(    position_filter[:-1], drop=True)
wins       = wins.set_index(      position_filter[:-1], drop=True)
total      = total.set_index(     position_filter[:-1], drop=True)
indexes    = indexes.set_index(   position_filter[:-1], drop=True)


# Graph spreads
ind = indexes.loc[1.919][0].values
plt.figure(figsize=(16, 2))
a = pd.DataFrame(results.index.tolist(), columns = ['ind'])
a['outcome'] = 0.1
a.set_index('ind', inplace=True, drop=True)
b = (down.loc[ind, 'd6'] \
              < up.loc[ind, 'u1']).astype(float).values
b[b == 0] = .65
a.loc[ind, 'outcome'] = b
plt.plot(a, '+')
print('Mean: {}'.format((1 / a.mean()).values[0]))
print('Std:  {}'.format((1 / a.std()).values[0]))

# Print Results maybe


# Colect for export to tableau
totab = []
for i in ex.index:
    for c in ex.columns:
        tmp = [i, c]
        for df in [ex, throughput, bars, groups, wins, total]:
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
totab.to_csv('/Users/user/Desktop/top.csv')









































'''
# How do we do with a closings slope?
results_backup = results.copy()
up_backup      = up.copy()
down_backup    = down.copy()
res = results.loc[ind]
u   = up.loc[ind]
d   = down.loc[ind]
# index = res[res.closings_slope < 0].index.values
index = results[results.channel_range > 0].index.values
(d.loc[index, 'd6'] < d.loc[index, 'u1']).mean()
'''

'''
# Results with outcomes in group to tab
restotab = results.loc[indexes.loc[1.616][0].values]
'''





















''' Here JUST FOR FUN.  DON'T GET STUCK ON IT. '''
import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC as svc
x = results.loc[ind].copy()
y = (down.loc[ind, 'd6'] < up.loc[ind, 'u1']).astype(int)


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
logreg = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                       hidden_layer_sizes=(5, 2), random_state=1)
logreg.fit(x_train, y_train)
predictions = logreg.predict(x_test)
print(classification_report(y_test, predictions))

logreg = svc()
logreg = KNeighborsClassifier(n_neighbors=2)
logreg = DecisionTreeClassifier(random_state=0)





