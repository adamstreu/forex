import time
import numpy as np
import pandas as pd
import random
from itertools import product
import pickle
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from scipy                     import signal
from scipy                     import stats
import warnings
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.neural_network    import MLPClassifier
from sklearn.decomposition     import PCA
from sklearn.metrics           import classification_report 
from sklearn.metrics           import precision_score
from sklearn.preprocessing     import MinMaxScaler          as minmaxscaler
from sklearn.preprocessing     import StandardScaler        as standardscaler
from sklearn.ensemble          import ExtraTreesClassifier  as etc
import os; os.chdir('/forex')
from libraries.transformations import horizontal_transform
from libraries.transformations import create_channels
from libraries.transformations import get_groups
from libraries.plotting        import plot_channels
from libraries.outcomes        import up_down_outcomes
from libraries.oanda           import get_candles

warnings.filterwarnings("ignore")#, category=DeprecationWarning)


'''
Flatten and scale window.
Keep histogram of values as a frame.
That's it.

HAving addeed the -1 score all my calculations will be off.
But that;s ok cause they already were cause I wasn;t calculating bars correct.

Now I don't care about my calcs.  I can;t move on with this type of this till
the classfication report shows something good.
'''


def convert_candles_by_timing(candles, skip_seconds):
    times_collection = [pd.to_datetime(_from)]
    final_time = pd.to_datetime(_to)
    while times_collection[-1] <= final_time:
        times_collection.append(pd.to_datetime(times_collection[-1]) + pd.Timedelta(seconds = skip_seconds))
    # Convert times lists to df and fill it with candles values
    times = pd.DataFrame(times_collection, columns = ['timestamp'])
    times = pd.merge(left=times, right=candles, how='left', on='timestamp')
    # Drop rows befroe first non null row
    times = times.loc[times.timestamp >= candles.timestamp.values[0]]
    times = times.fillna(method='ffill')
    times.reset_index(inplace = True, drop=True)
    return times



# Parameters
############################################################################### 
# Instrument
instrument          = 'EUR_USD'
granularity         = 'S15'         
_from               = '2015-01-01T00:00:00Z'
_to                 = '2018-01-01T00:00:00Z'
# Window Parameters
large_window        = 500
small_window        = 60 # How many gran seconds in one window
bin_window          = 100
large_skip          = small_window * large_window 
reduce_frames       = int(large_skip * .01)
bins                = np.linspace(0, 1, bin_window + 1)
search_window       = large_skip
# Outcomes
outcome_width       = np.array([.001, .002, .003, .004, .006, .008, .009, .01])
outcome_width       = np.array([.005, .001, .002, .003, .004, .005])
# Instantiation
results             = []
bars_up             = []
bars_down           = []


''' Get Candles.  Backfill empty seconds.  Set outcome width by candles. '''
# Import Candles.  Create new candle set based on even time spacing
candles = get_candles(instrument, granularity, _from, _to)
# candles = convert_candles_by_timing(candles, 5)
# Cut off candles at base to make for even division into large skip
candles = candles[(candles.shape[0] - int(candles.shape[0] / large_skip) * large_skip):]
candles = candles.reset_index(drop=True)
# Remove days when markets are not open
# Set outcome width
#outcome_width *= candles.midclose.mean()


# Call each frame.  Scale all values based on large * small window.
for i in range(large_skip, candles.shape[0] - search_window, reduce_frames):
    if i % 100000 == 0: print('{:.2f}'.format(i / candles.shape[0]))
    closings = candles.loc[i - large_skip + 1: i, 'midclose'].values
    closings_flattened = horizontal_transform(closings)
    scaled = minmaxscaler().fit_transform(closings_flattened['closing'].reshape(-1, 1))
    frame = np.empty((bin_window, large_window))
    frame_count = 0
    for j in range(small_window, scaled.shape[0] + 2, small_window):
        hist = np.histogram(scaled[j - bin_window: j], bins=bins)
        frame[:, frame_count] = hist[0][-1::-1]
        frame_count += 1
    results.append(frame.reshape(1, -1).tolist()[0])

    # Get outcomes at each position
    search = min(i + search_window, candles.shape[0])
    closings_outcomes = candles.loc[i: search, 'midclose'].values
    up_down = up_down_outcomes(closings_outcomes, outcome_width, search_window)
    bars_up.append(up_down['up'])
    bars_down.append(up_down['down'])
    
# Ready Dataframes for analysis
results = pd.DataFrame(np.array(results))
# Create Outcomes Columns 
columns_up, columns_down = [], []
for i in range(len(bars_up[0])):
    columns_up.append('u' + str(i))
    columns_down.append('d' + str(i))
# Create one bars df
bars_up = pd.DataFrame(np.array(bars_up), columns=columns_up)
bars_down = pd.DataFrame(np.array(bars_down), columns=columns_down)
bars = bars_down.join(bars_up, lsuffix='left')
# Make sure values are all floats (except instrument)
results = results.apply(pd.to_numeric, errors='ignore')
bars = bars.apply(pd.to_numeric, errors='ignore')
# Keep maybe for modeling
bars_to_use = bars.copy()
results_to_use = results.copy()






# Channel Statistics ML
################################################################################
predictions   = []
test_values   = []
probabilities = []
preds_columns = []
analysis      = []
bars_columns = list(product(columns_down, columns_up))
models = [('log', LogisticRegression(solver = 'saga',
                                                multi_class = 'ovr',
                                                max_iter = 100,
                                                # n_jobs = -1,
                                                C = 100000,
                                                fit_intercept = False)),          
          ('nn', MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
               beta_1=0.9, beta_2=0.999, early_stopping=False,
               epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
               learning_rate_init=0.001, max_iter=200, momentum=0.9,
               nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
               solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
               warm_start=False))]
for model in models:
    for bc in bars_columns:
        if bc[0][1] == bc[1][1]:
            print('---------{}------------'.format(bc))
            x = results.copy()
            # Assign outcome of -1 if both up and down not found
            not_ind = bars[(bars[bc[0]] == search_window) & 
                           (bars[bc[1]] == search_window)].index
            outs    = (bars[bc[0]] > bars[bc[1]]).astype(int)     
            outs.loc[not_ind] = -1
            # Gather statistics on outcomes
            down_width = outcome_width[int(bc[0][1:])]
            up_width   = outcome_width[int(bc[1][1:])] 
            down0     = (bars[bc[0]] == 0).sum()
            up0       = (bars[bc[1]] == 0).sum()
            dropped   = not_ind.shape[0]
            downperc  = (outs == 0).mean()
            upperc    = (outs == 1).mean()
            # Create Split index
            train_percent = .8
            start = int(x.shape[0] * train_percent)
            stop = x.shape[0]
            train_ind = np.arange(start)
            test_ind  = np.arange(start, stop)
            np.random.shuffle(train_ind)
            np.random.shuffle(test_ind)
            # Split all df's on same index
            x_train    = x.loc[train_ind]
            x_test     = x.loc[test_ind]
            y_train    = outs.loc[train_ind]
            y_test     = outs.loc[test_ind]
            # Standardizet values on both periods
            try:
#                scaler  = standardscaler()    
#                x_train = scaler.fit_transform(x_train) 
#                x_test  =  scaler.transform(x_test)                
                # Train and Test Model on current month
                mod              = model[1]
                mod.fit(x_train, y_train)
                preds = mod.predict(x_test).astype(int)  
                predictions.append(preds.tolist())
                test_values.append(y_test.tolist())
                preds_columns.append('{}_{}_{}'.format(model[0], bc[0], bc[1]))
                # Get Scores
                score      = precision_score(y_test.values.astype(str), 
                                             preds.astype(str), 
                                             average=None)
                # Get bars (bad logic)
                cond1 = (bars[bc[0]] != search_window + 1)
                cond2 = (bars[bc[1]] != search_window + 1)
                d_bars = int(bars.loc[cond1, bc[0]].mean())
                u_bars = int(bars.loc[cond2, bc[1]].mean()) 
                # collect Results Statistics
                down_wins     = y_test[(preds == 0) & (y_test == 0)].shape[0]
                down_losses   = y_test[(preds == 0) & (y_test == 1)].shape[0]
                up_wins       = y_test[(preds == 1) & (y_test == 1)].shape[0]
                up_losses     = y_test[(preds == 1) & (y_test == 0)].shape[0]
                try:
                    down_win_perc = down_wins / (down_wins + down_losses)
                except:
                    down_win_perc = 0
                try:
                    up_win_perc   = up_wins / (up_wins + up_losses)
                except:
                    up_win_perc = 0
                down_ex = down_width * down_win_perc - up_width * (1 - down_win_perc)
                up_ex   = up_width * up_win_perc - down_width * (1 - up_win_perc)    
                down_tot = down_ex * (down_wins + down_losses)
                up_tot   = up_ex * (up_wins + up_losses)     
                down_throughput  = down_ex / d_bars
                up_throughput    = up_ex   / u_bars
                down_total_throughput  = down_tot / d_bars
                up_total_throughput    = up_tot   / u_bars
                down_rat   = down_width / up_width
                up_rat     = up_width / down_width
                down_ratio = down_rat * down_win_perc - (1 * (1 - down_win_perc)) 
                up_ratio   = up_rat   * up_win_perc - (1 * (1 - up_win_perc))   
                down_ratio_total = down_ratio * (down_wins + down_losses)
                up_ratio_total   = up_ratio * (up_wins + up_losses)   
    
                print(classification_report(y_test, preds))
                '''
                # Print progress
                msg = '{}\t{}\t{}\t{}\t{}\t{}\t{}'
                print(msg.format(model[0], 
                                 bc, 
                                 score,
                                 (y_test == 0).sum(),
                                 (y_test == 1).sum(),
                                 (preds == 0).sum(),
                                 (preds == 1).sum()))
                '''
                '''
                # Save File to location
                file = '{}{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                file = file.format(tmp_file_location, instrument, model[0], df[0], 
                               window, granularity, round(down_width, 5), round(up_width, 5),
                               today)
                pickle.dump(mod, open(file, 'wb'))
                # Collect statistics into analysis table
                '''
                analysis.append([bc[0], 
                                 bc[1], 
                                 down_width,
                                 up_width,
                                 d_bars,
                                 u_bars,
                                 down_wins,
                                 down_losses,
                                 up_wins     ,  
                                 up_losses    , 
                                 down_win_perc ,
                                 up_win_perc ,
                                 down_ex ,
                                 up_ex   ,
                                 down_tot ,
                                 up_tot   ,
                                 down_throughput,  
                                 up_throughput   ,
                                 down_total_throughput,  
                                 up_total_throughput,                                
                                 down_ratio_total,
                                 up_ratio_total
                                 ])
            except Exception as e:
                print('{} Didnt work: {}'.format(bc, e))


# Collect Prediction Statistics into DataFrame
###############################################################################
analysis_columns = [
                    'down',
                    'up',
                    'down_width',
                    'up_width',
                    'd_bars',
                    'u_bars',
                    'down_wins',
                    'down_losses',
                    'up_wins'     ,  
                    'up_losses'    , 
                    'down_win_perc' ,
                    'up_win_perc' ,
                    'down_ex' ,
                    'up_ex'   ,
                    'down_tot' ,
                    'up_tot'   ,
                    'down_throughput',  
                    'up_throughput'   ,
                    'down_total_throughput',  
                    'up_total_throughput'   ,  
                    'down_ratio_total',
                    'up_ratio_total',                              
                     ]
analysis = pd.DataFrame(np.array(analysis), columns = analysis_columns) 
analysis = analysis.apply(pd.to_numeric, errors='ignore')
predictions = pd.DataFrame(np.array(predictions).T, columns = np.array(preds_columns))
test_values = pd.DataFrame(np.array(test_values).T, columns = preds_columns)










