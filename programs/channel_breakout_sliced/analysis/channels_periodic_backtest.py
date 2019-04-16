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


'''
Methods:
    There are three different analysiss that I want to do:
        Channel:
            Take what I have, use the correct models and timing to generate 
            predictions using ml.
        Bimodal - maybe scipy.stats.rv_histogram
        Periodic - 
            still a problem - i wouldn't have complete results searching for
            outcomes at en of every month as i do now.


Next:
    
    What correlates better - down ratio total or throiuput total or expected value total
        In fact, can we just do a totally seperate program based on both of thse?
        The portfolio created is very, very different.
        Need to analyze best return a over time and reliability of correlation (
        especialyl at the higher performing values) (most important here.)
        
        
        
        
        
        
    if iteration works ok - is it better with relative or fixed outcomes range?
    change closings to closings - closings[0] when putting those results 
        in the results
    
    Can try different scaler types (provided by sklearn)
    
    this is all taking a very long time:
        Need to engage more cores ( by time - shoudl be easy to do actually.
        Need to collect data differently if possible.
        Only after the other stuff is done though of course.
        
        
    Whayt about adding slopes for different windows?
    What about tkaing varying windows over smae time line and adding their 
    probabilites together ?
    for insatance, take minute marks.
    
    
    Transfer outcomes to up df and down df then
            create outcomes from the combinations.
    
    WE want to find a bimodal distribution heading in a  direction(slope)
        with end conditions that call for a reverse (or continuation ? )
            end conditions: channel position
            bimodal: with maybe weight / curtosis on one side?
        Can look for these discreeetly or just plug them into a model.
    
    
    
    Stats:
        
        local slope up or down
            (did it last touch the above channel or the below channel)
    
        Heikin Ashi (try with)
    
    Analysis:
        
        
        use sklearn for outside parameters to test for bimodal
            a whole investigation into bimodal or not
            
        Analyze outcomes by model probability
        

               
    Further Research:
        mean squared displacment and brownian motion stuff
    

    
'''


# Parameters
############################################################################### 
# Instrument
instruments = ['EUR_USD', 'EUR_AUD', 'AUD_CAD', 'EUR_CHF',
               'EUR_GBP', 'GBP_CHF', 'GBP_USD', 'NZD_USD', 'USD_CAD',
               'USD_CHF', 'EUR_NZD', 'EUR_SGD', 'EUR_CAD', 'USD_SGD', 
               'GBP_AUD', 'AUD_USD', 'GBP_NZD']
instrument  = instruments[0]
granularity = 'M5'
_from =       '2015-01-01T00:00:00Z'
_to   =       '2018-01-01T00:00:00Z'
# Window Pareters
window = 500
search_outcomes = window
periods_per_year = 12
# Outcomes
outcome_width = [.1, .25, .5, .75 , 1, 1.25, 1.5, 2, 2.5, 3, 3.5]
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
for l in range(locations.shape[0] - 1):
    # Limit Dataframe to period locations
    results = results_to_use.loc[locatiobarns[l]].copy()
    bars    = bars_to_use.loc[locations[l]].copy()
    history = history_to_use.loc[locations[l]].copy()   
    # Get dataframes for next month
    results_next      = results_to_use.loc[locations[l + 1]].copy()
    bars_next         = bars_to_use.loc[locations[l + 1]].copy()
    history_next      = history_to_use.loc[locations[l + 1]].copy()
#     Remove those outcomes from the period that aare not resolved In period.
    dfs = [('results', results, results_next), 
           ('history', history, history_next)]
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
                        not_ind = bars[(bars[bc[0]] == search_outcomes + 1) & 
                                       (bars[bc[1]] == search_outcomes + 1)].index
                        outs    = (bars[bc[0]] > bars[bc[1]]).astype(int)     
                        outs    = outs[~np.in1d(outs.index.values, not_ind)]
                        x     = x[~np.in1d(x.index.values, not_ind)]
                        x = x.reset_index(drop=True)
                        outs = outs.reset_index(drop=True)
                     Gather statistics on outcomes
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
                    
                        scaler  = standardscaler()    
                        x_train = scaler.fit_transform(x_train) 
                        x_test  =  scaler.transform(x_test) 
                    x_next  = scaler.transform(df[2])
                    y_next  = (bars_next[bc[0]] > bars_next[bc[1]]).astype(int)
                     Train and Test Model.  Print Predictions Results                   
                     Train and Test Model on current month
                    mod              = model[1]
                    mod.fit(x_train, y_train)
                    predictions      = mod.predict(x_test).astype(int)  
                    predictions_next = mod.predict(x_next).astype(int)
                    # Get Scores
                    score      = precision_score(y_test.values.astype(str), 
                                                 predictions.astype(str), 
                                                 average=None)
                    print(classification_report(y_test, predictions))
                    """
                    score_next = precision_score(y_next.values.astype(str), 
                                                 predictions_next.astype(str), 
                                                 average=None)
                    # Get bars (bad logic)
                    cond1 = (bars[bc[0]] != search_outcomes + 1)
                    cond2 = (bars[bc[1]] != search_outcomes + 1)
                    d_bars = int(bars.loc[cond1, bc[0]].mean())
                    u_bars = int(bars.loc[cond2, bc[1]].mean()) 
                    cond1 = (bars_next[bc[0]] != search_outcomes + 1)
                    cond2 = (bars_next[bc[1]] != search_outcomes + 1)
                    d_bars_next = int(bars_next.loc[cond1, bc[0]].mean())
                    u_bars_next = int(bars_next.loc[cond2, bc[1]].mean()) 
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
                    # For Results Stitistics on next period
                    down_wins_next              = y_next[(predictions_next == 0) & (y_next == 0)].shape[0]
                    down_losses_next            = y_next[(predictions_next == 0) & (y_next == 1)].shape[0]
                    up_wins_next                = y_next[(predictions_next == 1) & (y_next == 1)].shape[0]
                    up_losses_next              = y_next[(predictions_next == 1) & (y_next == 0)].shape[0]
                    try:
                        down_win_perc_next      = down_wins_next / (down_wins_next + down_losses_next)
                    except:
                        down_win_perc_next      = 0
                    try:
                        up_win_perc_next        = up_wins_next / (up_wins_next + up_losses_next)
                    except:
                        up_win_perc_next        = 0
                    down_ex_next                = down_width * down_win_perc_next - up_width * (1 - down_win_perc_next)
                    up_ex_next                  = up_width * up_win_perc_next - down_width * (1 - up_win_perc_next)    
                    down_tot_next               = down_ex_next * (down_wins_next + down_losses_next)
                    up_tot_next                 = up_ex_next * (up_wins_next + up_losses_next)     
                    down_throughput_next        = down_ex_next / d_bars_next
                    up_throughput_next          = up_ex_next   / u_bars_next
                    down_total_throughput_next  = down_tot_next / d_bars_next
                    up_total_throughput_next    = up_tot_next   / u_bars_next
                    down_ratio_next             = down_rat * down_win_perc_next - (1 * (1 - down_win_perc_next)) 
                    up_ratio_next               = up_rat   * up_win_perc_next - (1 * (1 - up_win_perc_next))                      
                    down_ratio_total_next       = down_ratio_next * (down_wins_next + down_losses_next)
                    up_ratio_total_next         = up_ratio_next * (up_wins_next + up_losses_next)  
                    # Print progress
                    msg = '{}, {}, {}, {}:\t{}\t{}'
                    print(msg.format(l, df[0], model[0], bc, score, score_next))
                    
                    # Collect statistics into analysis table
                    analysis.append([l,
                                     model[0], 
                                     df[0], 
                                     bc[0], 
                                     bc[1], 
                                     down_width,
                                     up_width,
                                     d_bars,
                                     u_bars,
                                     d_bars_next,
                                     u_bars_next, 
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
                                     up_total_throughput   ,                                
                                     down_wins_next,
                                     down_losses_next,
                                     up_wins_next     ,  
                                     up_losses_next    , 
                                     down_win_perc_next ,
                                     up_win_perc_next ,
                                     down_ex_next ,
                                     up_ex_next   ,
                                     down_tot_next ,
                                     up_tot_next   ,
                                     down_throughput_next  ,
                                     up_throughput_next   ,
                                     down_total_throughput_next,
                                     up_total_throughput_next,
                                     down_ratio_total,
                                     up_ratio_total,
                                     down_ratio_total_next,
                                     up_ratio_total_next                                     
                                     ])
                        except Exception as e:
                            print('{} Didnt work: {}'.format(bc, e))
                            # Collect statistics into analysis table
                            analysis.append([l,
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
                                             0,
                                             0
                                             ])
    

# Collect Prediction Statistics into DataFrame
################################################################################
analysis_columns = ['periods',
                'model',
                'df',
                'down',
                'up',
                'down_width',
                'up_width',
                'd_bars',
                'u_bars',
                'd_bars_next',
                'u_bars_next', 
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
                'down_wins_next',
                'down_losses_next',
                'up_wins_next'     ,  
                'up_losses_next'    , 
                'down_win_perc_next' ,
                'up_win_perc_next' ,
                'down_ex_next' ,
                'up_ex_next'   ,
                'down_tot_next' ,
                'up_tot_next'   ,
                'down_throughput_next'  ,
                'up_throughput_next'   ,
                'down_total_throughput_next',
                'up_total_throughput_next',
                'down_ratio_total',
                'up_ratio_total',
                'down_ratio_total_next',
                'up_ratio_total_next'
                 ]
analysis = pd.DataFrame(np.array(analysis), columns = analysis_columns) 
analysis = analysis.apply(pd.to_numeric, errors='ignore')

      
# Analyze Periodic Results
###############################################################################
'''
Way harder than it needs to be.
Just sort by the top 5 and grab the relavent (or all) columns.
'''
top_values = 5
down_column_to_evaluate = 'down_ratio_total'
up_column_to_evaluate   = 'up_ratio_total'
up_index =       []
down_index =     []
up_current =     []
down_current =   []
down_follows =   []
up_follows =     []
# Periodic Analysis - the good stuff.
for i in range(analysis.periods.max() +  1):
    anal = analysis[analysis.periods == i]
    up_current.append(anal.sort_values(up_column_to_evaluate,     ascending = False).head(top_values)[up_column_to_evaluate].values.tolist()) 
    down_current.append(anal.sort_values(down_column_to_evaluate, ascending = False).head(top_values)[down_column_to_evaluate].values.tolist()) 
    up_index   = anal.sort_values(up_column_to_evaluate,   ascending = False).head(top_values)[['model', 'df', 'down', 'up']].values.tolist()
    down_index = anal.sort_values(down_column_to_evaluate, ascending = False).head(top_values)[['model', 'df', 'down', 'up']].values.tolist()
    # Use filtered results to grab model prediction score form next period
    tmp_up = []
    tmp_down = []
    # anal = analysis[analysis.periods == i + 1]
    for j in range(top_values):
        try:
            tmp_up.append(float(anal.loc[(anal.model == up_index[j][0]) & \
                                         (anal.df == up_index[j][1]) & \
                                         (anal.down == up_index[j][2]) & \
                                         (anal.up == up_index[j][3]), 
                                         up_column_to_evaluate + '_next']))
        except:
            tmp_up.append(0)
        try:
            tmp_down.append(float(anal.loc[(anal.model == down_index[j][0]) & \
                                           (anal.df == down_index[j][1]) & \
                                           (anal.down == down_index[j][2]) & \
                                           (anal.up == down_index[j][3]), 
                                           down_column_to_evaluate + '_next']))
        except:
            tmp_down.append(0)
    up_follows.append(tmp_up)
    down_follows.append(tmp_down)    
# Creata Dataframesa
up_current   = pd.DataFrame(np.array(up_current))
down_current = pd.DataFrame(np.array(down_current))
up_follows   = pd.DataFrame(np.array(up_follows))
down_follows = pd.DataFrame(np.array(down_follows))
avg = ((down_follows.mean().mean()) + (up_follows.mean().mean())) / 2
# Print basic results
print('Current:\n----------------')
print('Up:   {}'.format(up_current.mean().values))
print('Up:   {}'.format(up_current.mean().mean()))
print('Down: {}'.format(down_current.mean().values))
print('Down: {}'.format(down_current.mean().mean()))
print('\nFollows:\n----------------')
print('Up:   {}'.format(up_follows.mean().values))
print('Up:   {}'.format(up_follows.mean().mean()))
print('Down: {}'.format(down_follows.mean().values))
print('Down: {}'.format(down_follows.mean().mean()))
print('\nCombined Expectation:\n----------------')
print(avg)


"""
