#!python3
import numpy as np
import pandas as pd
from itertools import product
from scipy                     import stats
from sklearn.linear_model      import LogisticRegression
from sklearn.neural_network    import MLPClassifier
from sklearn.metrics           import precision_score
from sklearn.preprocessing     import MinMaxScaler          as minmaxscaler
from sklearn.preprocessing     import StandardScaler        as standardscaler
import os; os.chdir('/forex')
from libraries.transformations import horizontal_transform
from libraries.transformations import create_channels




def channel_statistics(closing_values, window_length, candles): 
    print('gathring channel statistics')
    results = []
    history = []    
    # Pad  / truncate histogram peaks to i places
    pad = lambda a,i : a[0:i] if len(a) > i else a + [0] * (i-len(a))
    # Window Analysis
    ############################################################################### 

    # Prepare DWindow for closing values and outcome winodow
    closings = closing_values
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
    # Collect Historgram for window and iot's peaks for results
    hist = np.histogram(scaled, 150)
    history.append(list(hist[0])) 
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
    results.append([mms.data_range_[0],
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
                    candles.volume.mean(),
                    candles.volume.values[-1], 
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
    # Results Configuration
    ###############################################################################
    # Assemble columns for results and outcomes and historygrams
    columns = ['closings_range',
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
    history = pd.DataFrame(np.array(history))
    results = pd.DataFrame(np.array(results), columns=columns)
    # Make sure values are all floats (except instrument)
    results = results.apply(pd.to_numeric, errors='ignore')
    history = history.apply(pd.to_numeric, errors='ignore')
    # Rearange by location ( keep in order )
    results = results.reset_index(drop=True)
    history = history.reset_index(drop=True)
    # Add Additional Columns
    results['hist_kurtosis'] = history.kurtosis(axis=1)
    results['hist_skew'] = history.skew(axis=1)
    results['both_slopes'] = results.channel_slope / results.closings_slope
    return {'history': history.values.tolist()[0], 'results': results.values.tolist()[0]} 
    
    
    
    
    
    
    
def channel_ml(candles, results_to_use, history_to_use, bars_to_use, 
               window, search_outcomes, outcome_width):
    # Channel Statistics ML
    ################################################################################
    # Preliminaries
    year = round((pd.to_datetime(candles.timestamp.values[-1]) - pd.to_datetime(candles.timestamp.values[0])).days / 365, 0)
    periods = int(12  * year)
    locations = np.arange(results_to_use.shape[0] - results_to_use.shape[0] % periods).reshape((periods, -1))
    analysis = []
    
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

    bars_columns = list(product(bars_to_use.filter(regex='d').columns.tolist(), 
                                bars_to_use.filter(regex='u').columns.tolist()))
    
    
    
    
    for l in range(locations.shape[0] - 1):
        # Limit Dataframe to period locations
        results = results_to_use.loc[locations[l]].copy()
        bars    = bars_to_use.loc[locations[l]].copy()
        history = history_to_use.loc[locations[l]].copy()   
        # Get dataframes for next month
        results_next      = results_to_use.loc[locations[l + 1]].copy()
        bars_next         = bars_to_use.loc[locations[l + 1]].copy()
        history_next      = history_to_use.loc[locations[l + 1]].copy()
        
        # Remove those outcomes from the period that aare not resolved In period.
    #    bars_ind = np.tile(bars.index.values.reshape(-1, 1), bars.columns.shape[0])
    #    bars.values[((bars + bars_ind) > bars.index.values[-1]).values] = search_outcomes + 1
        
        dfs = [('results', results, results_next), 
               ('history', history, history_next)]
        for df in dfs:
            for model in models:
                for bc in bars_columns:
                    x = df[1].copy()
                    # calculate outcomes column.  Drop double zeros from df and outs
                    not_ind = bars[(bars[bc[0]] == search_outcomes + 1) & 
                                   (bars[bc[1]] == search_outcomes + 1)].index
                    outs    = (bars[bc[0]] > bars[bc[1]]).astype(int)     
                    outs    = outs[~np.in1d(outs.index.values, not_ind)]
                    x     = x[~np.in1d(x.index.values, not_ind)]
                    x = x.reset_index(drop=True)
                    outs = outs.reset_index(drop=True)
                    # Gather statistics on outcomes
                    down_width = outcome_width[int(bc[0][1:])]
                    up_width   = outcome_width[int(bc[1][1:])] 
#                    down0     = (bars[bc[0]] == 0).sum()
#                    up0       = (bars[bc[1]] == 0).sum()
#                    dropped   = not_ind.shape[0]
#                    downperc  = (outs == 0).mean()
#                    upperc    = (outs == 1).mean()
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
                        x_next  = scaler.transform(df[2])
                        y_next  = (bars_next[bc[0]] > bars_next[bc[1]]).astype(int)
                        # Train and Test Model.  Print Predictions Results                   
                        # Train and Test Model on current month
                        mod              = model[1]
                        mod.fit(x_train, y_train)
                        predictions      = mod.predict(x_test).astype(int)  
                        predictions_next = mod.predict(x_next).astype(int)
                        # Get Scores
                        score      = precision_score(y_test.values.astype(str), 
                                                     predictions.astype(str), 
                                                     average=None)
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
    
    
    
    





if __name__  == '__main__':
    pass