import datetime
import time
import numpy as np
import pandas as pd
import random
from itertools import product
import pickle
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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
from programs.channel.functions.channels import channel_statistics
from itertools import product


"""
For portfolio building:
    runs through all listed currencies, 
    saves best _x_ models accorind to some _column_.
    
    Run commented code at end to move saved models into the models direcotry
        that then main program will access.

"""


# Parameters
############################################################################### 
# Instrument
instruments = ['EUR_USD', 'EUR_AUD', 'AUD_CAD', 'EUR_CHF',
               'EUR_GBP', 'GBP_CHF', 'GBP_USD', 'NZD_USD', 'USD_CAD',
               'USD_CHF', 'EUR_NZD', 'EUR_SGD', 'EUR_CAD', 'USD_SGD', 
               'GBP_AUD', 'AUD_USD', 'GBP_NZD']
granularity = 'M5'
# All periods evaluated over last month
_from = str(datetime.datetime.now() - pd.Timedelta(weeks=4))
_from = _from.replace(' ', 'T')[:-7] + 'Z'
_to   = '2030-01-01T00:00:00Z'
today = str(datetime.datetime.now())[:str(datetime.datetime.now()).find(' ')]
# Window Pareters
window = 250
search_outcomes = window
periods_per_year = 12
# Outcomes
outcome_width = np.array([.001, .002, .003, .004, .006, .008, .009, .01])
# Instantiation
results = []
bars = []
history = []
bars_ratio = []
bars_up = []
bars_down = []
distances = []
outcomes = []
hist_peaks = []
to_portfolio = []

tmp_file_location = '/forex/programs/channel/tmp/'
top_values = 5
down_column_to_evaluate = 'down_total_throughput'
    




for instrument in instruments:    
    
    granularity = 'M5'
    _from =       '2015-12-01T00:00:00Z'
    _to   =       '2016-01-01T00:00:00Z'
    # Window Pareters
    window = 250
    search_outcomes = window
    periods_per_year = 12
    # Outcomes
    outcome_width = np.array([.001, .002, .003, .004, .006, .008, .009, .01])
    # Instantiation
    results = []
    bars = []
    history = []
    bars_ratio = []
    bars_up = []
    bars_down = []
    distances = []
    outcomes = []
    hist_peaks = []
    
    # Import Candles.  Set outcomes as percnet of midclose
    candles = get_candles(instrument, granularity, _from, _to)
    outcome_width *= candles.midclose.mean()
    # Pad  / truncate histogram peaks to i places
    pad = lambda a,i : a[0:i] if len(a) > i else a + [0] * (i-len(a))
    
    
    # Window Analysis
    ############################################################################### 
    # Call each window.  Transform and collect all resultsi
    for i in range(window, candles.shape[0] - search_outcomes):
        # Prepare DWindow for closing values and outcome winodow
        closings = candles.loc[i - window: i, 'midclose'].values
        search = min(i + search_outcomes, candles.shape[0])
        closings_outcomes = candles.loc[i: search, 'midclose'].values
        # Flatten closing values 
        closings_flat = horizontal_transform(closings)
        # Scale Flattened closing values to be between 0 and 1    
        mms = minmaxscaler()
        mms.fit(closings_flat['closing'].reshape(-1, 1))
        scaled = mms.transform(closings_flat['closing'].reshape(-1, 1)).ravel()
        # Create channels from flattened and scaled closing values
        channels = create_channels(scaled)
        # Calculate Outcomes Range from original closing values
        c6 = mms.inverse_transform(channels['c6'].reshape(-1, 1)).ravel()
        c4 = mms.inverse_transform(channels['c4'].reshape(-1, 1)).ravel()
        top    = c6[-1] + closings_flat['linregress'][-1] 
        bottom = c4[-1] + closings_flat['linregress'][-1] 
        _range = top - bottom
        # Calculate up down outcomes and bars 
        distance = np.array(outcome_width)
        up_down = up_down_outcomes(closings_outcomes, distance, search_outcomes)
        bars_up.append(  [str(instrument)] + [str(i)] + up_down['up'])
        bars_down.append([str(instrument)] + [str(i)] + up_down['down'])
        # Collect Historgram for window and iot's peaks for results
        hist = np.histogram(scaled, 150)
        history.append([str(instrument)] + [str(i)] + list(hist[0])) 
        hist = np.histogram(scaled, bins=10)
        # Collect histogram peaks into results
        if hist[0][0] > hist[0][1]:
            keep = [True]
        else:
            keep = [False]
        for h in range(1, hist[0].shape[0]-1):
            if hist[0][h] > hist[0][h+1] and hist[0][h] > hist[0][h-1]:
                keep.append(True)
            else:
                keep.append(False)
        if hist[0][-1] > hist[0][-2]:
            keep.append(True)
        else:
            keep.append(False)   
        keep = np.array(keep)
        x = (hist[1] + ((hist[1][1] - hist[1][0]) / 2))[:-1]
        # Collect Data
        results.append([instrument,
                        i,
                        mms.data_range_[0],
                        closings.std(),
                        closings.mean(),
                        stats.kurtosis(closings),
                        stats.skew(closings),
                        closings_flat['slope'],
                        mms.scale_[0],
                        scaled.std(),
                        scaled.mean(),
                        stats.kurtosis(scaled),
                        stats.skew(scaled),
                        candles.loc[i - window: i, 'volume'].values.mean(),
                        candles.loc[i - window: i, 'volume'].values[-1], 
                        channels['breakout'],
                        channels['slope'],
                        _range,
                        channels['closing_position'],
                        channels['c1'].mean(),
                        channels['c2'].mean(),
                        channels['c3'].mean(),
                        channels['c4'].mean(),
                        channels['c5'].mean(),
                        channels['c6'].mean(),
                        channels['c7'].mean(),
                        channels['d01'],
                        channels['d12'],
                        channels['d23'],
                        channels['d34'],
                        channels['d45'],
                        channels['d56'],
                        channels['d67'],
                        channels['d78']
                        ] + pad(list(x[keep]), 3))        
        # Track program run.  Print verification graphs.
        if i % 10000 == 0:
            complete = i - window
            complete /= (candles.shape[0] - (search_outcomes + window)) / 100
            msg = '{} is {}% complete at location {}.\t{}\t'
            print(msg.format(instrument, int(complete), i, _range))
            # Print a plot every, say, 10000 candles
            # plot_channels(candles, i, window)
    # In place of multithreading
    collect_results  = results
    collect_history  = history
    collect_outcomes = outcomes
    collect_bars_up = bars_up
    collect_bars_down = bars_down
    # Results Configuration
    ###############################################################################
    # Assemble columns for results and outcomes and historygrams
    columns = ['instrument',
               'location',
               'closings_range',
               'closings_std', 
               'closings_mean',
               'closings_kurt', 
               'closings_skew',
               'closings_slope',
               'scaler', 
               'scaled_std', 
               'scaled_mean',
               'scaled_kurt',
               'scaled_skew',
               'volume_mean',
               'volume_final', 
               'breakout',
               'channel_slope',
               'channel_range',
               'channel_position',
               'c1',
               'c2',
               'c3',
               'c4',
               'c5',
               'c6',
               'c7',
               'd01',
               'd12',
               'd23',
               'd34',
               'd45',
               'd56',           
               'd67',
               'd78',
               'peak1',
               'peak2',
               'peak3',]
    # Put together dataframe of results, history, outcomes
    columns_up  = ['instrument', 'location']
    columns_down  = ['instrument', 'location']
    for i in range(len(collect_bars_up[0]) - 2):
        columns_up.append('u' + str(i))
        columns_down.append('d' + str(i))
    history = pd.DataFrame(np.array(collect_history))
    results = pd.DataFrame(np.array(collect_results), columns=columns)
    bars_up = pd.DataFrame(np.array(bars_up), columns=columns_up)
    bars_down = pd.DataFrame(np.array(bars_down), columns=columns_down)
    bars = pd.merge(bars_up, bars_down, how='left', 
                    left_on=['instrument', 'location'], 
                    right_on=['instrument', 'location'])
    # Make sure values are all floats (except instrument)
    results = results.apply(pd.to_numeric, errors='ignore')
    history = history.apply(pd.to_numeric, errors='ignore')
    bars = bars.apply(pd.to_numeric, errors='ignore')
    # Rearange by location ( keep in order )
    results = results.sort_values(['location', 'instrument'])
    bars = bars.sort_values(['location', 'instrument'])
    history = history.sort_values([1, 0])
    # Reset Indexes
    results = results.reset_index(drop=True)
    history = history.reset_index(drop=True)
    bars    = bars.reset_index(drop=True)
    bars    = bars.replace(0, search_outcomes + 1)
    # Add A location indexer for later analysis
    location_index = results.location
    # Drop location and Instrument Columns
    history = history.drop([0,1], axis=1)
    results = results.drop(['instrument', 'location'], axis=1)
    bars = bars.drop(['instrument', 'location'], axis=1)
    # Add Additional Columns
    results['hist_kurtosis'] = history.kurtosis(axis=1)
    results['hist_skew'] = history.skew(axis=1)
    results['both_slopes'] = results.channel_slope / results.closings_slope
    # Tranfer to backups for ml section - odd i know
    bars_to_use = bars.copy()
    history_to_use = history.copy()
    results_to_use = results.copy()
    
    
    # Channel Statistics ML
    ################################################################################
    # Preliminaries
    year = (pd.to_datetime(candles.timestamp.values[-1]) - pd.to_datetime(candles.timestamp.values[0])).days / 365
    periods = int(round(periods_per_year * year, 0))
    locations = np.arange(results_to_use.shape[0] - results_to_use.shape[0] % periods).reshape((periods, -1))
    analysis = []
    bars_columns = list(product(columns_down[2:], columns_up[2:]))
    # Prepare models for inspection
    models = [('log', LogisticRegression(solver = 'liblinear',
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
                   warm_start=False)),
              # ('etc', etc()),
              # ('svm', SVC())
              ]
    l = 0
    # Limit Dataframe to period locations
    results = results_to_use.loc[locations[l]].copy()
    bars    = bars_to_use.loc[locations[l]].copy()
    history = history_to_use.loc[locations[l]].copy()   
    dfs = [('results', results), 
           ('history', history)]
    for df in dfs:
        for model in models:
            for bc in bars_columns:
                if bc[0][1] == bc[1][1]:
                    print('---------{}------------'.format(bc))
                    x = df[1].copy()
                    not_ind = bars[(bars[bc[0]] == search_outcomes) & 
                    (bars[bc[1]] == search_outcomes)].index
                    outs    = (bars[bc[0]] > bars[bc[1]]).astype(int)     
                    outs.loc[not_ind] = -1
                    # calculate outcomes column.  Drop double zeros from df and outs
#                    not_ind = bars[(bars[bc[0]] == search_outcomes + 1) & 
#                                   (bars[bc[1]] == search_outcomes + 1)].index
#                    outs    = (bars[bc[0]] > bars[bc[1]]).astype(int)     
#                    outs    = outs[~np.in1d(outs.index.values, not_ind)]
#                    x     = x[~np.in1d(x.index.values, not_ind)]
#                    x = x.reset_index(drop=True)
#                    outs = outs.reset_index(drop=True)

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
                    # Standardize values on both periods
                    try:
                        scaler  = standardscaler()    
                        x_train = scaler.fit_transform(x_train) 
                        x_test  =  scaler.transform(x_test)                
                        # Train and Test Model on current month
                        mod              = model[1]
                        mod.fit(x_train, y_train)
                        predictions      = mod.predict(x_test).astype(int)  
                        # Get Scores
                        score      = precision_score(y_test.values.astype(str), 
                                                     predictions.astype(str), 
                                                     average=None)
                        # Get bars (bad logic)
                        cond1 = (bars[bc[0]] != search_outcomes + 1)
                        cond2 = (bars[bc[1]] != search_outcomes + 1)
                        d_bars = int(bars.loc[cond1, bc[0]].mean())
                        u_bars = int(bars.loc[cond2, bc[1]].mean()) 
                        # collect Results Statistics
                        down_wins     = y_test[(predictions == 0) & (y_test == 0)].shape[0]
                        down_losses   = y_test[(predictions == 0) & (y_test == 1)].shape[0]
                        up_wins       = y_test[(predictions == 1) & (y_test == 1)].shape[0]
                        up_losses     = y_test[(predictions == 1) & (y_test == 0)].shape[0]
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
            
                        print(classification_report(y_test, predictions))
                        '''
                        # Print progress
                        msg = '{}, {}, {}, {}:\t{}'
                        print(msg.format(l, df[0], model[0], bc, score))
                        
                        # Save File to location
                        file = '{}{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                        file = file.format(tmp_file_location, instrument, model[0], df[0], 
                                       window, granularity, round(down_width, 5), round(up_width, 5),
                                       today)
                        pickle.dump(mod, open(file, 'wb'))
                        '''
                        # Collect statistics into analysis table
                        analysis.append([instrument, 
                                         l,
                                         model[0], 
                                         df[0], 
                                         bc[0], 
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
                                         up_ratio_total])#,                                
                                         #file])
                    
                    except Exception as e:
                        print('{} Didnt work: {}'.format(bc, e))
                        # Collect statistics into analysis table
                        analysis.append([instrument,
                                         l,
                                         model[0], 
                                         df[0], 
                                         bc[0], 
                                         bc[1], 
                                         down_width,
                                         up_width,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,  
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0])#,
#                                         0
#                                         ])

    
    # Collect Prediction Statistics into DataFrame
    ################################################################################
    analysis_columns = ['instrument',
                        'periods',
                        'model',
                        'df',
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
                        'up_ratio_total']#,
#                        'file_location',
#                         ]
    analysis = pd.DataFrame(np.array(analysis), columns = analysis_columns) 
    analysis = analysis.apply(pd.to_numeric, errors='ignore')
    analysis['down_ratio'] = analysis.down_width / analysis.up_width
    analysis['up_ratio'] = analysis.up_width / analysis.down_width
           
    
    
    """
    # Get top _x_ performing models for previous period and Ready for portfolio
    ################################################################################
    #up_column_to_evaluate   = 'up_ratio_total'
    today = str(datetime.datetime.now())[:str(datetime.datetime.now()).find(' ')]

    
    # Filter top results and keep neccesary columns
    keeps = analysis.sort_values(down_column_to_evaluate, ascending = False).head(
            top_values)[['model', 'df', 'down_width', 
                         'up_width', 'file_location']].values.tolist()
    
    # Arrange as dictionary for export to portfolio
    for keep in keeps:
        tmp = {'instrument': instrument, 'model': keep[0], 
               'df': keep[1], 'window': window, 'down_width': round(keep[2], 5), 
               'up_width': round(keep[3], 5), 'direction': 0, 
               'filter': down_column_to_evaluate, 'analysis_date': today, 
               'file_location': keep[4]}
        to_portfolio.append(tmp)
    
    # Remove all unused files from tmp directory
    saved_files = []
    saved_files.append([port['file_location'] for port in to_portfolio])
    saved_files = saved_files[0]
    in_directory = os.listdir(tmp_file_location)
    to_remove = []
    for file in in_directory:
        if (tmp_file_location + file) not in saved_files:
            to_remove.append(tmp_file_location + file)
    [os.remove(file) for file in to_remove]        
    
    
# Prepare Data for transfer to portfolio
################################################################################
# Change file =_location listing in portfolio
for file in to_portfolio:
    replacement = 'models/' + str(down_column_to_evaluate) + '_'
    file['file_location'] = file['file_location'].replace('tmp/', replacement) 
# Print for transfer to portfolio
print()    
[print((str(line)+',')) for line in to_portfolio]
print()

'''
USE WITH CAUTION ! - - - Move all files in tmp to models directory.
new_location = '/forex/programs/channel/models/'
new_location += str(down_column_to_evaluate) + '_'
for file in os.listdir(tmp_file_location):
    old_file = tmp_file_location + file
    new_file = new_location      + file
    os.rename(old_file, new_file)
'''

"""
