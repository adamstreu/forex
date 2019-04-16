 .'''
Next and Misc.


TODAY:
    
    Also - we have that nice max - abs(min) thing to work on.

    Also - Rolling autocorrelation - find best fit _x_ in _y_ windows
    
    Can we try this with the -2, -1, 0, 1, 2 column?
    
    how high / low will it go before hitting same
    
    Focusing on diffierence in high, open, low for outcomes.
    Focusing on short range changes - within a few bars where possible 
    relative targets
    
    rolling volume
    
    
    change indicator - the location where outcomes change from
        one binary to the other ( as they go in pairs
        diff for different targets.)

    get candles for currency.  
    How do high and low compare for currency ?

Misc:
    
    Get open and spread form universe    
    
    Shorter Windows
    5M granularity
    Filter by day and hour
    Relative Outcomes
    
    ensemble classifiers ( for example, try bagging)
    cross-validation

    timeseries
     and non-binary outcomes.
    run larger window intervals  and targets   
    
    get more indicators
        volume
        some standard ones
        get another currency
        save all to file
        get larger targets.

    
    Clustering on preds
        spectral, onclasssmv, isolation forest
    individual inddcator prediction probabilities compilation on new ml
    
    covariance estimtors, and etc on sklearn.
        
    filter them on indicators (: eur slopes up and usd slopes down)
    
    
    
    
    
ml:
        
    Will have to get spread from currency univerwse at some point    
    
        does adding a third currency prob help at all ? 
        Can I use a linear regression with praameters then with a supervised 
        get descent and final ditributions for ration both long and short

    what are the local conditioins around those points in which the 
        price moves up

    Relative outcomes based on ....
    Can i get a score somehow on 'changes' ? 
    Linear prediction on changes values ? - rolling changes ? 
    A 'Simple' Probability of move
    Add boundary distribution conditions to filters 
    taps - hit previous or turn around ? ? ? ? ? ?
    
            
Starategies:

 
    ml for each currency then combined by % or etc.
    binned frequency probability score per indicator
    a few waves at a time and thier small binned combination score.





Byu running ml on just eur_hml, then just roll_pos, then getting their probs
    hten running those through i have gotten better results thatn anything 
    so far on a a single currency.





BUMPS WE KNOW:
eur_roll_chann_diff_pos > 1.5 gives a two point bump to eur_long      hitting 15% of population
eur_roll_chann_diff_pos < -1.5 gives a two point bump to eur_short    hitting 15% of population

'''







# ===========================================================================
# Imports
# ===========================================================================
if 0:  
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from multiprocessing import Process
    from multiprocessing import Manager
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import EllipticEnvelope
    from sklearn.cluster import SpectralClustering
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import validation_curve
    from sklearn.model_selection import train_test_split    
    from sklearn.metrics import classification_report as cr    
    import os; os.chdir('/northbend')
    from libraries.transformations import get_groups
    from libraries.outcomes import get_outcomes_multi
    from libraries.currency_universe import get_currencies
    from libraries.indicators import stochastic_oscillator
    from libraries.indicators import get_indicator_bins_frequency
    # Multiprocessing wrapped
    from libraries.indicators import waves_wrapper
    from libraries.indicators import get_rolling_rank
    # Multiprocessing unwrapped
    from libraries.indicators import get_rolling_mean_pos_std
    from libraries.indicators import get_channel_mean_pos_std
    from libraries.indicators import get_rolling_currency_correlation
    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15)
    np.warnings.filterwarnings('ignore')   
    plt.rcParams['figure.figsize'] = [14, 6]
    

# ===========================================================================
# Call currency universe.  Multiprocessing Units.
# ===========================================================================
if 0:    
    
    ''' 
    Need to use because of bug in response / oanda / mutli:
    env no_proxy = '*' ipython 
    '''
   
    # Parameters    
    granularity = 'M1'
    _from = '2015-11-20T00:00:00Z'
    _to   = '2016-01-01T00:00:00Z'

    # Call Currency Matrix with different Columns
    def get_universe(arg, procnum, return_dict):
        print('Get Universe {}'.format(procnum))
        print('Get Universe {}'.format(arg))
        currency_dictionary = get_currencies(granularity, _from, _to, arg)
        return_dict[str(arg)] = currency_dictionary    

    # Parameters and Mutliprocessing setup
    args = ['midclose', 'midlow', 'midhigh', 'volume']
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    # Call processes, join, wait to complete, etc.
    for i in range(4):
        p = Process(target=get_universe, args=(args[i], i, return_dict))
        p.start()
        jobs.append(p)
    for job in jobs:
        job.join()

    # Relabel Values from Multiprocessing Return Dict.
    print('\nAssembling dictionaries\n')
    cur     = return_dict['midclose']['currencies']
    curdiff = return_dict['midclose']['currencies_delta']
    ratios  = return_dict['midclose']['ratios']
    cur_set = return_dict['midclose']['currency_set']
    high    = return_dict['midhigh']['currencies']
    low     = return_dict['midlow']['currencies']
    volume  = return_dict['volume']['currencies']
    volume_difference = return_dict['volume']['currencies_delta'] 
    timestamp = return_dict['volume']['timestamps'] 
    timestamp = pd.DataFrame(timestamp)
    a = timestamp.values
    timestamp['weekday'] = pd.DatetimeIndex(a.ravel()).weekday.values
    timestamp['hour'] = pd.DatetimeIndex(a.ravel()).hour.values
    

# ===========================================================================
# Set Parameters 
# ===========================================================================
if 0:   
    
    # Currencies and instruments
    currencies = ['aud', 'cad', 'eur', 'gbp', 'nzd', 'usd']
    aud = cur.loc[:, 'aud']
    cad = cur.loc[:, 'cad']
    eur = cur.loc[:, 'eur']
    gbp = cur.loc[:, 'gbp']
    nzd = cur.loc[:, 'nzd']
    usd = cur.loc[:, 'usd']
    eur_usd = ratios.loc[:, 'eur_usd']
    aud_cad = ratios.loc[:, 'aud_cad']
    ratio = ratios.loc[:, 'eur_usd']
    
    # Indicators
    windows = np.array([60, 120, 240, 480, 960, 1500, 1750, 2000]) 
    windows = np.array([5, 10, 15, 30, 45, 60, 90, 120])
    bins = 60

    # Outcomes
    currency_targets = np.arange(.00005, .00061 , .00008).round(6)
    currency_targets = np.linspace(.00005, .001, 8).round(6)
    currency_targets = np.linspace(.00005, .0005, 8).round(6)
    ratio_targets    = np.arange(.0005, .0045, .0005)
    max_search_interval = 3000

# ===========================================================================
# Get Outcome
# ===========================================================================
if 0:
    
    # EUR
    outcomes = get_outcomes_multi(eur, currency_targets, max_search_interval)
    eur_up = outcomes['up']
    eur_down = outcomes['down']
    eur_long = eur_up < eur_down
    eur_short = eur_down < eur_up
    mins = np.minimum(eur_up.values, eur_down.values)
    eur_minimums = pd.DataFrame(mins, 
                                columns = eur_up.columns,
                                index = eur_up.index)
    print(eur_minimums[eur_minimums != eur.shape[0]].dropna().mean())

    # USD
    outcomes = get_outcomes_multi(usd, currency_targets, max_search_interval)
    usd_up = outcomes['up']
    usd_down = outcomes['down']
    usd_long = usd_up < usd_down
    usd_short = usd_down < usd_up
    mins = np.minimum(usd_up.values, usd_down.values)
    usd_minimums = pd.DataFrame(mins, 
                                columns = usd_up.columns,
                                index = usd_up.index)
    print(usd_minimums[usd_minimums != usd.shape[0]].dropna().mean())
    
    # Aud
    outcomes = get_outcomes_multi(aud, currency_targets, max_search_interval)
    aud_up = outcomes['up']
    aud_down = outcomes['down']
    aud_long = aud_up < aud_down
    aud_short = aud_down < aud_up
    mins = np.minimum(aud_up.values, aud_down.values)
    aud_minimums = pd.DataFrame(mins, 
                                columns = aud_up.columns,
                                index = aud_up.index)
    print(aud_minimums[aud_minimums != aud.shape[0]].dropna().mean())

    # Cad
    outcomes = get_outcomes_multi(cad, currency_targets, max_search_interval)
    cad_up = outcomes['up']
    cad_down = outcomes['down']
    cad_long = cad_up < cad_down
    cad_short = cad_down < cad_up
    mins = np.minimum(cad_up.values, cad_down.values)
    cad_minimums = pd.DataFrame(mins, 
                                columns = cad_up.columns,
                                index = cad_up.index)
    print(cad_minimums[cad_minimums != cad.shape[0]].dropna().mean())
    
    # eur_usd
    outcomes = get_outcomes_multi(eur_usd, ratio_targets, max_search_interval)
    eur_usd_up = outcomes['up']
    eur_usd_down = outcomes['down']
    eur_usd_long = eur_usd_up < eur_usd_down
    eur_usd_short = eur_usd_down < eur_usd_up
    mins = np.minimum(eur_usd_up.values, eur_usd_down.values)
    eur_usd_minimums = pd.DataFrame(mins, 
                                columns = eur_usd_up.columns,
                                index = eur_usd_up.index)
    print(eur_usd_minimums[eur_usd_minimums != eur_usd.shape[0]].dropna().mean())
    
    
    # Aud CAd
    outcomes = get_outcomes_multi(aud_cad, ratio_targets, max_search_interval)
    aud_cad_up = outcomes['up']
    aud_cad_down = outcomes['down']
    aud_cad_long = aud_cad_up < aud_cad_down
    aud_cad_short = aud_cad_down < aud_cad_up
    mins = np.minimum(aud_cad_up.values, aud_cad_down.values)
    aud_cad_minimums = pd.DataFrame(mins, 
                                columns = aud_cad_up.columns,
                                index = aud_cad_up.index)
    print(aud_cad_minimums[aud_cad_minimums != aud_cad.shape[0]].dropna().mean())


# ===========================================================================
# Get Indicators
# ===========================================================================
if 0:
    
    # =================== EUR =======================================
    # Channel Indicators
    eur_channel_stats = get_channel_mean_pos_std(eur.values, windows)
    eur_channel_std   = eur_channel_stats['std']
    eur_channel_pos   = eur_channel_stats['pos']
    eur_channel_mean  = eur_channel_stats['mean']
    eur_slopes        = eur_channel_stats['slope']
    # Rolling mean, pos, std and Slope
    eur_rolling_stats = get_rolling_mean_pos_std(eur, windows)
    eur_rolling_std   = eur_rolling_stats['std']
    eur_rolling_pos   = eur_rolling_stats['pos']
    eur_rolling_mean  = eur_rolling_stats['mean']
    # Rolling Channel Difference
    eur_roll_chann_diff_mean =  eur_rolling_mean - eur_channel_mean
    eur_roll_chann_diff_pos  =  eur_rolling_pos - eur_channel_pos
    eur_roll_chann_diff_std  =  eur_rolling_std - eur_channel_std
    # HML
    eur_measure = high.eur - eur
    eur_hml = waves_wrapper(eur_measure, windows, get_rolling_rank)
    # Volume
    
    # Stochastic Oscilattor
    k = 90
    d = 240
    eur_so_1 = stochastic_oscillator(eur.values, high.eur.values, 
                                   low.eur.values, 9, 18)
    eur_so_2 = stochastic_oscillator(eur.values, high.eur.values, 
                                   low.eur.values, 90, 240)
    eur_so = pd.DataFrame(np.c_[eur_so_1, eur_so_2], 
                          columns = [0, 1],
                          index = eur_slopes.index)    
    # Standardize slopes
    slope_vals = StandardScaler().fit_transform(eur_slopes.fillna(0))
    eur_slope_std     = pd.DataFrame(slope_vals,
                                     columns = eur_slopes.columns,
                                     index = eur_slopes.index)
    # Standardize std
    slope_vals = MinMaxScaler().fit_transform(eur_rolling_std)
    eur_rolling_std_std     = pd.DataFrame(slope_vals,
                                     columns = eur_rolling_std.columns,
                                     index = eur_rolling_std.index)
    

    # =================== USD =======================================
    # Channel Indicators
    usd_channel_stats = get_channel_mean_pos_std(usd.values, windows)
    usd_channel_std   = usd_channel_stats['std']
    usd_channel_pos   = usd_channel_stats['pos']
    usd_channel_mean  = usd_channel_stats['mean']
    usd_slopes        = usd_channel_stats['slope']
    # Rolling mean, pos, std and Slope
    usd_rolling_stats = get_rolling_mean_pos_std(usd, windows)
    usd_rolling_std   = usd_rolling_stats['std']
    usd_rolling_pos   = usd_rolling_stats['pos']
    usd_rolling_mean  = usd_rolling_stats['mean']
    # Rolling Channel Difference
    usd_roll_chann_diff_mean =  usd_rolling_mean - usd_channel_mean
    usd_roll_chann_diff_pos  =  usd_rolling_pos - usd_channel_pos
    usd_roll_chann_diff_std  =  usd_rolling_std - usd_channel_std    
    # HML
    usd_measure = high.usd - usd
    usd_hml = waves_wrapper(usd_measure, windows, get_rolling_rank)
    # Volume
    # Stochastic Oscilattor
    k = 90
    d = 240
    usd_so_1 = stochastic_oscillator(usd.values, high.usd.values, 
                                   low.usd.values, 9, 18)
    usd_so_2 = stochastic_oscillator(usd.values, high.usd.values, 
                                   low.usd.values, 90, 240)
    usd_so = pd.DataFrame(np.c_[usd_so_1, usd_so_2], 
                          columns = [0, 1],
                          index = usd_slopes.index)  
    # Standardize slopes
    slope_vals = StandardScaler().fit_transform(usd_slopes)
    usd_slope_std = pd.DataFrame(slope_vals,
                                 columns = usd_slopes.columns,
                                 index = usd_slopes.index)
    # Standardize std
    slope_vals = MinMaxScaler().fit_transform(usd_rolling_std)
    usd_rolling_std_std = pd.DataFrame(slope_vals,
                                       columns = usd_rolling_std.columns,
                                       index = usd_rolling_std.index)


    # =================== Ratio =====================================
    # Channel Indicators
    ratio_channel_stats = get_channel_mean_pos_std(ratio.values, windows)
    ratio_channel_std   = ratio_channel_stats['std']
    ratio_channel_pos   = ratio_channel_stats['pos']
    ratio_channel_mean  = ratio_channel_stats['mean']
    ratio_slopes        = ratio_channel_stats['slope']
    # Rolling mean, pos, std and Slope
    ratio_rolling_stats = get_rolling_mean_pos_std(ratio, windows)
    ratio_rolling_std   = ratio_rolling_stats['std']
    ratio_rolling_pos   = ratio_rolling_stats['pos']
    ratio_rolling_mean  = ratio_rolling_stats['mean']
    # Rolling Channel Difference
    ratio_roll_chann_diff_mean =  ratio_rolling_mean - ratio_channel_mean
    ratio_roll_chann_diff_pos  =  ratio_rolling_pos - ratio_channel_pos
    ratio_roll_chann_diff_std  =  ratio_rolling_std - ratio_channel_std    


    # =================== Currency Combined =========================
    rolling_correlation = get_rolling_currency_correlation(eur.values, 
                                                           usd.values, 
                                                           windows)
    shift = 15
    between = pd.DataFrame((eur - eur.shift(shift)) / usd.shift(shift))
    
# ===========================================================================
# Strategy - Binned Probability Score by Indicator
# ===========================================================================
if 0:       
    
    export_path = '/Users/user/Desktop/to_plot/'
    
    # =================== EUR =======================================
    # Score Slopes
    ind_score = get_indicator_bins_frequency(eur_slopes, eur_long, 
                                             'long', bins)
    eur_slope_scored = ind_score['results']
    eur_slope_indexes = ind_score['indexes']    
    # HML Slopes
    ind_score = get_indicator_bins_frequency(eur_hml, eur_long, 
                                             'long', bins, True)
    eur_hml_scored = ind_score['results']
    eur_hml_indexes = ind_score['indexes']    
    eur_hml_scored.to_csv(export_path + 'eur_hml_scored.csv')
    # Score pos difference    
    ind_score = get_indicator_bins_frequency(eur_roll_chann_diff_pos, 
                                             eur_long, 'long', bins)
    eur_pos_diff_scored = ind_score['results']
    eur_pos_diff_indexes = ind_score['indexes']
    # Score pos difference    
    ind_score = get_indicator_bins_frequency(eur_roll_chann_diff_pos, 
                                             eur_long, 'long', bins)
    eur_chan_var_scored = ind_score['results']
    eur_chan_var_indexes = ind_score['indexes']
    # Score pos difference    
    ind_score = get_indicator_bins_frequency(eur_roll_chann_diff_pos, 
                                             eur_long, 'long', bins)
    eur_roll_chann_diff_pos_scored = ind_score['results']
    eur_roll_chann_diff_pos_indexes = ind_score['indexes']
    # Scale and Combine indicators into one dataframe and score
    eur_combined =  StandardScaler().fit_transform(eur_slopes)
    eur_combined += StandardScaler().fit_transform(eur_roll_chann_diff_mean)
    eur_combined = pd.DataFrame(eur_combined, 
                                columns = eur_slopes.columns,
                                index = eur_slopes.index)
    ind_score = get_indicator_bins_frequency(eur_combined, eur_long, 
                                             'long', bins)
    eur_combined_scored = ind_score['results']
    eur_combined_indexes = ind_score['indexes']    
    
    

    # Verification - Plot placement Distribution and stats on one index
    if False:
        # Slopes
        location = 3517
        plt.plot(eur_slope_indexes[location], 'o')
        print(eur_slope_indexes[location].sum())
        print(eur_long.loc[eur_slope_indexes[location]].mean())
        # Position Difference
        location = 2613
        plt.plot(eur_slope_indexes[location], 'o')
        print(eur_pos_diff_indexes[location].sum())
        print(eur_long.loc[eur_pos_diff_indexes[location]].mean())        
        # Combined
        location = 2613
        plt.plot(eur_combined_indexes[location], 'o')
        print(eur_combined_indexes[location].sum())
        print(eur_long.loc[eur_combined_indexes[location]].mean()) 
        # Channel Std
        location = 2613
        plt.plot(eur_chan_var_indexes[location], 'o')
        print(eur_chan_var_indexes[location].sum())
        print(eur_long.loc[eur_chan_var_indexes[location]].mean())     
        # diff in mean
        location = 3159
        plt.plot(eur_roll_chann_diff_mean_indexes[location], 'o')
        print(eur_roll_chann_diff_mean_indexes[location].sum())
        print(eur_long.loc[eur_roll_chann_diff_mean_indexes[location]].mean())     


        # Export Data to Tableau
        eur_slope_scored.to_csv(export_path + 'eur_slope_scored.csv')
        eur_pos_diff_scored.to_csv(export_path + 'eur_pos_diff_scored.csv')
        eur_combined_scored.to_csv(export_path + 'eur_combined_scored.csv')
        eur_chan_var_scored.to_csv(export_path + 'eur_chan_var_scored.csv')
        eur_roll_chann_diff_mean_scored.to_csv(export_path + \
                                     'eur_roll_chann_diff_mean_scored.csv')

        # Intersect good indexes from both
        intersection = eur_slope_indexes[3517] & eur_slope_indexes[2613]
        # Plot dist and outcomes on intersection 
        plt.plot(intersection)
        print(intersection.sum())
        print(eur_long.loc[intersection].mean())  


    # =================== USD =======================================    
    # Score Slopes
    ind_score = get_indicator_bins_frequency(usd_slopes, usd_short, 
                                             'short', bins)
    usd_slope_scored = ind_score['results']
    usd_slope_indexes = ind_score['indexes']    
    # Score pos difference    
    ind_score = get_indicator_bins_frequency(usd_roll_chann_diff_pos, 
                                             usd_short, 'short', bins)
    usd_pos_diff_scored = ind_score['results']
    usd_pos_diff_indexes = ind_score['indexes']
    # Scale and Combine indicators into one dataframe and score
    usd_combined =  StandardScaler().fit_transform(usd_slopes)
    usd_combined += StandardScaler().fit_transform(usd_channel_pos)
    usd_combined += StandardScaler().fit_transform(usd_roll_chann_diff_mean)    
    usd_combined = pd.DataFrame(usd_combined, 
                                columns = usd_slopes.columns,
                                index = usd_slopes.index)
    ind_score = get_indicator_bins_frequency(usd_combined, usd_short, 
                                             'short', bins)
    usd_combined_scored = ind_score['results']
    usd_combined_indexes = ind_score['indexes']    

    # Verification - Plot placement Distribution and stats on one index
    if False:
        # Slopes
        location = 3517
        plt.plot(usd_slope_indexes[location], 'o')
        print(usd_slope_indexes[location].sum())
        print(usd_short.loc[usd_slope_indexes[location]].mean())
        # Position Difference
        location = 713
        plt.plot(usd_pos_diff_indexes[location], 'o')
        print(usd_pos_diff_indexes[location].sum())
        print(usd_short.loc[usd_pos_diff_indexes[location]].mean())        
        # Combined
        location = 2471
        plt.plot(usd_combined_indexes[location], 'o')
        print(usd_combined_indexes[location].sum())
        print(usd_short.loc[usd_combined_indexes[location]].mean())     

        # Export Data to Tableau
        usd_slope_scored.to_csv(export_path + 'usd_slope_scored.csv')
        usd_pos_diff_scored.to_csv(export_path + 'usd_pos_diff_scored.csv')
        usd_combined_scored.to_csv(export_path + 'usd_combined_scored.csv')

        # Intersect good indexes from both
        intersection = usd_slope_indexes[3517] & usd_slope_indexes[2613]
        # Plot dist and outcomes on intersection 
        plt.plot(intersection)
        print(intersection.sum())
        print(usd_short.loc[intersection].mean())  


    # =================== Combined ==================================
    # Rolling Correlation with Ratio Long
    ind_score = get_indicator_bins_frequency(rolling_correlation, 
                                             ratio_long, 'long', bins)
    rolling_correlation_scored = ind_score['results']
    rolling_correlation_indexes = ind_score['indexes']    
    file = '/Users/user/Desktop/to_plot/rolling_correlation_scored.csv'
    rolling_correlation_scored.to_csv(file)
    # Verification - Correlation
    location = 3517
    plt.plot(rolling_correlation_indexes[location], 'o')
    print(rolling_correlation_indexes[location].sum())
    print(ratio_long.loc[rolling_correlation_indexes[location]].mean())



# ===========================================================================
# Day and Time Filters
# ===========================================================================
if True:
    time_index = timestamp.loc[timestamp.weekday != 6]\
                .loc[timestamp.weekday != 6]\
                .loc[timestamp.hour > 5]\
                .loc[timestamp.hour < 12]\
                .index.values
    
    
    
# ===========================================================================
# Strategy - Filtered index intersections
# ===========================================================================
if 0:
    '''
    eur_roll_chann_diff_pos.iloc[:, 7] > 1.5 for long
    '''
    
    use_time = False
    
    
    # =========== EUR ===============================================
    # Set Filters
    cond2 = eur_so.iloc[:, 0] < 10
    cond3 = True #eur_roll_chann_diff_pos.iloc[:, 0] > 2
    cond4 = volume.eur > .8
    cond5 = True # (eur_rolling_pos.fillna(0).mean(axis=1) - eur_rolling_pos.fillna(0).iloc[:, 7]) < -1.5
    cond6 = True
    
    # Apply Filters Get total
    cond1 = np.ones(eur_slopes.shape[0]).astype(bool)
    eur_index = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    # Apply Time Filter
    if use_time:
        time_index_bool = np.zeros(eur_index.shape[0]).astype(bool)
        time_index_bool[time_index] = True
        eur_index = eur_index & time_index_bool
        
    # Print and Plot Eur stats    
    print('\nEur Avg Win % \n -----------------------')
    print(eur_long.loc[eur_index].mean())
    plt.figure(figsize = (12,1))
    plt.plot(eur_index) 
    plt.title('Eur')
    print('\nPlacement count and %: {}\t{}'.format(eur_index.sum(), 
                                                   eur_index.sum() \
                                                   / eur.shape[0]))

    
    # =========== USD ===============================================
    # Set Filters
    cond2 = usd_so.iloc[:, 0] > 90
    cond3 = True #usd_roll_chann_diff_pos.iloc[:, 0] < -5
    cond4 = True #usd_roll_chann_diff_pos.iloc[:, 7] > 1.5
    cond5 = True# (usd_rolling_pos.fillna(0).mean(axis=1) - usd_rolling_pos.fillna(0).iloc[:, 7]) > 1.5
    cond6 = True # usd_rolling_std_std.iloc[:, 5] < .05
    
    # Apply Filters Get total
    cond1 = np.ones(usd_slopes.shape[0]).astype(bool)
    usd_index = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    # Apply Time Filter
    if use_time:
        time_index_bool = np.zeros(usd_index.shape[0]).astype(bool)
        time_index_bool[time_index] = True
        usd_index = usd_index & time_index_bool
    # Print and Plot usd stats    
    print('\nusd Avg Win % \n -----------------------')
    print(usd_short.loc[usd_index].mean())
    plt.figure(figsize = (12,1))
    plt.plot(usd_index)
    plt.title('usd')
    print('\nPlacement count and %: {}\t{}'.format(usd_index.sum(), 
                                                   usd_index.sum() \
                                                   / usd.shape[0]))


    # =========== Ratio =============================================
    # Get final results.  Print stats and plot Distribution
    final_index = eur_index & usd_index
    final_long = ratio_long.loc[final_index]
    print('\navg Win % on final Index\n -----------------------')
    print(final_long.mean())
    plt.figure(figsize = (12,1))
    plt.plot(final_index)
    plt.title('Ratio')
#    print('\navg minimum bars on final index\n -----------------------')
#    print(ratio_minimums.loc[final_index].mean())
    print('\nPlacement count and %: {}\t{}'.format(final_index.sum(), 
                                                   final_index.sum() \
                                                   / ratio.shape[0]))
    plt.figure(figsize = (12,4)); ratio.plot()


    # ============ Plot distributions selected  =====================
    if False:
        
        # Slopes Distribution
        plt.figure(figsize=(12,4))
        sns.distplot(eur_slopes.iloc[1000:, 4], label='eur')
        sns.distplot(usd_slopes.iloc[1000:, 4], label='eur')
        plt.title('Slopes')
        plt.legend()
        plt.show()
        # Std Difference Distribution
        plt.figure(figsize=(12,4))
        sns.distplot(eur_roll_chann_diff_std.iloc[1000:, 4], label='eur')
        sns.distplot(usd_roll_chann_diff_std.iloc[1000:, 4], label='eur')
        plt.title('roll_chann_diff_std')
        plt.legend()
        plt.show()
        # Position Distribution
        plt.figure(figsize=(12,4))
        sns.distplot(usd_channel_pos.iloc[1000:, 4], label='eur')
        sns.distplot(eur_channel_pos.iloc[1000:, 4], label='eur')
        plt.title('Slopes')
        plt.legend()
        plt.show()
        # Std Difference Distribution
        plt.figure(figsize=(12,4))
        sns.distplot(eur_roll_chann_diff_std.iloc[1000:, 4], label='eur')
        sns.distplot(usd_roll_chann_diff_std.iloc[1000:, 4], label='eur')
        plt.title('roll_chann_diff_std')
        plt.legend()
        plt.show()
        # Std  Distribution
        plt.figure(figsize=(12,4))
        sns.distplot(usd_rolling_std.iloc[1000:, 4], label='eur')
        sns.distplot(usd_rolling_std.iloc[1000:, 4], label='eur')
        plt.title('roll_chann_diff_std')
        plt.legend()
        plt.show()



# ===========================================================================
# Strategy - Machine Learning. 
# ===========================================================================
if 1:

    # What instrument / indicator combinations to us
    use_eur          = 0
    use_usd          = 1
    use_ratio        = 0
    use_cor          = 0
    use_custom       = 0
    use_hour         = 0
    
    # How to combine indicators
    use_normal       = 0
    use_subtraction  = 0
    use_column       = 1
    use_mean         = 0
    
    # Apply addition filters
    time_filter = False
    
    # Choose Outcomes df and target and window columns (as applicable)
    ml_outcomes = a.shift(60).fillna(0)# e_l.shift(1).fillna(method='backfill').astype(bool).astype(bool) #  # 
    target_column = 0
    window_columns = [0, 1, 2, 3]

    # Split
    tt_split_perc = .7
    use_validation = False   # If validate - split testing set up into halves
    
    # Create Df for machine learning based on choices above
    ml = pd.DataFrame(eur_slopes.iloc[:, 0])
    columns = ['drop']
    dfs = []
    # Choose which instrument to use
    if use_eur:
        dfs += [#(eur_so, 'eur_so'),
                #(pd.DataFrame(volume.eur), 'eur_volume'),
                #(eur_channel_std, 'eur_channel_std'),
                #(eur_hml, 'eur_hml'),
                #(eur_channel_pos, 'eur_channel_pos'),
                (eur_slopes, 'eur_slopes'),
                #(eur_rolling_std, 'eur_rolling_std'),
                (eur_rolling_pos, 'eur_rolling_pos'),
                #(eur_roll_chann_diff_pos, 'eur_roll_chann_diff_pos'),
                #(eur_roll_chann_diff_std, 'eur_roll_chann_diff_std'),
                #(eur_slopes * eur_roll_chann_diff_pos, 'times')
                ]
    if use_usd:
        dfs +=  [#(usd_so, 'usd_so'),
                #(pd.DataFrame(volume.eur), 'usd_volume'),
                #(usd_channel_std, 'usd_channel_std'),
                #(usd_hml, 'usd_hml'),
                #(usd_channel_pos, 'usd_channel_pos'),
                (usd_slopes, 'usd_slopes'),
                #(usd_rolling_std, 'usd_rolling_std'),
                (usd_rolling_pos, 'usd_rolling_pos'),
                #(usd_roll_chann_diff_pos, 'usd_roll_chann_diff_pos'),
                #(usd_roll_chann_diff_std, 'usd_roll_chann_diff_std'),
                #(usd_slopes * usd_roll_chann_diff_pos, 'times')
                ]
    if use_custom:
        dfs +=  [(a_long, 'a_long')]
    if use_ratio:
        dfs +=  [(ratio_channel_std, 'ratio_channel_std'),
                (ratio_channel_pos, 'ratio_channel_pos'),
                (ratio_slopes, 'ratio_slopes'),
                (ratio_rolling_std, 'ratio_rolling_std'),
                (ratio_rolling_pos, 'ratio_rolling_pos'),
                (ratio_roll_chann_diff_pos, 'ratio_roll_chann_diff_pos'),
                (ratio_roll_chann_diff_std, 'ratio_roll_chann_diff_std'),
                ]
    if use_cor:
        dfs +=  [(rolling_correlation.iloc[:, 3:], 'rolling_correlation'),
                (between, 'between'),
                ]
    # Choose how to use them
    for df in dfs:
        if use_normal:
            ml = ml.join(df[0], lsuffix='.')
            for col in df[0].columns:
                columns.append(df[1] + '_' + str(col))
        if use_column:
            ml = ml.join(df[0].iloc[:, window_columns], lsuffix='.')
            for i in range(len(window_columns)):
                columns.append(df[1] + '_' + str(window_columns[i]))   
        if use_subtraction:
                sub = df[0].iloc[:, 7] - df[0].iloc[:, 3]
                sub.name = df[1]
                ml = ml.join(sub, lsuffix='.')
                columns.append(df[1] + '_mean_sub')
        if use_mean:
             to_app = df[0].mean(axis=1)
             to_app.name = df[1]
             ml = ml.join(to_app, lsuffix='.')
             columns.append(df[1])
        if use_custom:
            ml = ml.join(df[0], lsuffix='.')
            for col in df[0].columns:
                columns.append(df[1] + '_' + str(col))
    if use_hour:
        ml = ml.join(timestamp['hour'], lsuffix='.')
        columns.append('hour')
        
            
    # Finalize ML Creation
    ml.columns = columns
    ml.drop('drop', inplace=True, axis = 1)
    
    # Drop nans from ml df.  Use index to drop related values from outcomes.
    keep_index = ml.loc[~(ml.isna().any(axis=1)).values].index.values
    ml = ml.loc[keep_index]
    ml_long = ml_outcomes.loc[keep_index, ml_outcomes.columns[target_column]]

    # Split  
    train_index, test_index = train_test_split(ml.index.values, 
                                               train_size=tt_split_perc, 
                                               shuffle=False)
    # Apply additional filters
    if time_filter:
        train_index = np.intersect1d(train_index, time_index)
        test_index = np.intersect1d(test_index, time_index)    
    
    x_train    = ml.loc[train_index]
    y_train    = ml_long.loc[train_index]
    x_test     = ml.loc[test_index]
    y_test     = ml_long.loc[test_index]

    
    # To split the testing set up into two equal sections
    if use_validation:
        test_index, validate_index = train_test_split(test_index,
                                                      train_size=.5,
                                                      shuffle=False)
        x_test     = ml.loc[test_index]
        y_test     = ml_long.loc[test_index]
        x_validate = ml.loc[validate_index]
        y_validate = ml_long.loc[validate_index]
 
    # Scale 
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    if use_validation:
        x_validate = scaler.transform(x_validate)
    
    # Implemnent Model 
    clf = GaussianNB()
    logreg = MLPClassifier(solver='lbfgs', alpha=1e-5,
                           hidden_layer_sizes=(5, 5), random_state=1)
    logreg = LogisticRegression(C = 1, solver = 'sag', n_jobs = 4,
                                multi_class= 'multinomial')
    logreg.fit(x_train, y_train)
    predictions = logreg.predict(x_test)
    print(cr(y_test, predictions))
    
    # Print Probability slices and distribution
    print('\n Slice, Shape, groups, win %\n-----------------------------')
    pred_probs = logreg.predict_proba(x_test)[:, 1]
    step = .01
    ml_indexes = []
    # plt.figure(figsize=(10,2))
    for _slice in np.arange(.5, 1, step).round(3):
        index = pred_probs > _slice
        ml_indexes.append(index)
        try:
            groups = get_groups(test_index[index], 300).shape[0]
            # plt.plot(['nan' if float(x) == 0 else x for x in\
            #          (pred_probs > _slice).astype(int) * _slice], 'o') 
            print('{}\t{}\t{}\t{:2f}'.format(_slice, index.sum(), groups,
                                          y_test[index].mean()))
        except:
            pass
    # plt.title('test')
    
    if use_validation:
        # CROSS VALIDATION SCORE
        print(' \n\n====== CROSS VALIDATION ==========')
        validation_predictions = logreg.predict(x_validate)
        print(cr(y_validate, validation_predictions))
        # Print Probability slices and distribution
        print('Slice, Shape, groups, win %\n-----------------------------')
        pred_probs = logreg.predict_proba(x_validate)[:, 1]
        step = .02
        ml_indexes = []
        # plt.figure(figsize=(10,2))
        for _slice in np.arange(.5, 1, step).round(3):
            index = pred_probs > _slice
            ml_indexes.append(index)
            try:
                groups = get_groups(test_index[index][index], 
                                    300).shape[0]
                # plt.plot(['nan' if float(x) == 0 else x for x in\
                #          (pred_probs > _slice).astype(int) * _slice], 'o') 
                print('{}\t{}\t{}\t{:2f}'.format(_slice, index.sum(), groups,
                                              y_validate[index].mean()))
            except:
                pass
        # plt.title('cross validation')
            
    # Get Scores for each variable from from machine learning.
    try:
        plt.figure(figsize=(8, 12))
        sns.barplot(y = ml.columns.tolist(), x = logreg.coef_[0])
        plt.title('Variable Scores from ml')
    except:
        pass
    plt.show()
    
    # Get and plot Correlations for each variable in ml
    plt.figure(figsize=(4, 4))    
    ml_corrs = ml.corr()
    sns.heatmap(ml_corrs)
    plt.show()
    
    # Print prediction results for all columns
    print('\nPredictions on all Target Columns in Test Set\n-----------------')
    print(ml_outcomes.loc[test_index].loc[predictions].mean())
    
    # Print Bars
    print('\nBars on Predictions\n--------------------')
    print(eur_minimums.loc[test_index].loc[predictions].mean())
    
    
    # Save pred prods for individual indicator
    if False:
        # Set Indexes
        hml_preds = pred_probs          # target column = 6, columns = all - descent resutls
        diff_pos_preds = pred_probs     
        roll_pos_preds = pred_probs     # target column = 7, columns = [4, 5, 6, 7] - excellent results
        # Intersect Indexes and call outcomes
        if False:
            combined = (hml_preds[:-1] > .52) & (diff_std_preds > .52)
            # Print combined index on all target columns
            print(ml_outcomes.loc[test_index].loc[combined].mean())
        
    
    
    # Get Training Scores - don't need quite now - good thoug 
    if False:
        train_scores, valid_scores = validation_curve(LogisticRegression(), 
                                                      x_train, y_train)
        train_sizes, train_scores, valid_scores = learning_curve(
                LogisticRegression(), x_train, y_train)    
        plt.figure(figsize=(4, 4))
        plt.scatter(train_scores, valid_scores)
        plt.title('Learning Curve - train score vs validiate score')
    
    
    # ======= Filter Under-performing Variables from model  =========  
    if False:
        # Filter out by coefficient on logistice regression
        a_cols = ml.columns.values[(abs(logreg.coef_) > .1)[0]]
        ml = ml.loc[:, a_cols]
        # Filter out by columns example
        ml.drop('eur_roll_chann_diff_pos_2', axis=1, inplace=True)
        
        
        
    
    


# ===========================================================================
# Further ML Testing - Ratios
# ===========================================================================
if 0:
    
    # ===================================================================
    # Setup splits and df's for further processing
    # ===================================================================
    if False:
        
        # Set prediction preds for each currency
        eur_preds = hml_preds #pred_probs
        usd_preds = roll_pos_preds # pred_probs 
        ratio_preds = diff_pos_preds #pred_probs
        
        eur_preds = pred_probs
        eur_t_index = test_index
        
        usd_preds = pred_probs 
        usd_t_index = test_index
        
    if True:
        
        # Parameters
        ratio_split = .7
        ratio_outcomes = ratio_long
        final_target_column = 3
    
        ratio_ml = pd.DataFrame({'eur': eur_preds, 'usd': usd_preds})
        ratio_ml.index = test_index

        # Build new df from prediction probabilities        
#        eur_preds_df = pd.DataFrame(np.array(eur_preds), columns=['eur'], index = eur_t_index)
#        usd_preds_df = pd.DataFrame(usd_preds, columns=['usd'], index = usd_t_index)
#        ratio_ml = eur_preds_df.join(usd_preds_df, lsuffix='v')
#        ratio_ml.dropna(inplace=True)
        
        # Get Final Outcomes (using ratio_ml index
        ratio_ml_outcomes = ratio_outcomes.loc[ratio_ml.index, 
                              ratio_outcomes.columns[final_target_column]]
        
        # Split Data from above's test set.
        ratio_train_index, ratio_test_index = train_test_split(ratio_ml.index,
                                              train_size=ratio_split, 
                                              shuffle=False) 
        
    

    # ===================================================================
    # Cluster / Novelty Detection on indicator / currency prediction probs
    # ===================================================================
    if True:
        
        ''' Uses ratio_ml formed in above segment '''

        # Supervised Outlier Detection    
        if False:
            # fit the model Using supervised training class
            clf = EllipticEnvelope()
            clf.fit(ratio_ml.loc[ratio_train_index].values.astype(np.float32),
                    ratio_ml_outcomes.loc[ratio_train_index].values.astype(np.float32)) 
            
            # Did i forget the suprevised part?
            cluster_predictions = clf.predict(ratio_ml.loc[ratio_test_index]\
                                              .values.astype(np.float32))    
            decision = clf.decision_function(ratio_ml.loc[ratio_test_index]\
                                             .values.astype(np.float32))
            envelope_isolation_prediction = cluster_predictions.copy()
            env_decision = decision.copy()
            
        # Unsupervised Outlier Detection
        if False:
            rng = np.random.RandomState(42)
            clf = IsolationForest(max_samples=100, random_state=rng, n_jobs=-1) 
            clf.fit(ratio_ml.loc[ratio_train_index].values.astype(np.float32))
            cluster_predictions = clf.predict(ratio_ml.loc[ratio_test_index]\
                                              .values.astype(np.float32))
            decision = clf.decision_function(ratio_ml.loc[ratio_test_index]\
                                             .values.astype(np.float32))
            forest_isolation_prediction = cluster_predictions.copy()
            forest_decision = decision.copy()

            
        # One Class Supervised  Novelty Detection
        if False:
            false_index = ratio_ml_outcomes.loc[ratio_ml_outcomes == False]
            false_index = false_index.index.values
            cluster_train_index = np.intersect1d(false_index, 
                                                 ratio_train_index)
            # Set test and train values 
            cluster_training_values = ratio_ml.loc[cluster_train_index].values
            cluster_testing_values = ratio_ml.loc[ratio_test_index].values
            # As contigous arrays
            cluster_training = np.ascontiguousarray(cluster_training_values)
            cluster_testing = np.ascontiguousarray(cluster_testing_values)
            # Scaled
            scaler = StandardScaler()
            scaled_training_values = scaler.fit_transform(cluster_training)
            scaled_testing_values = scaler.transform(cluster_testing)

            # fit the model
            clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1, cache_size=5000)
            clf.fit(scaled_training_values)
            cluster_predictions = clf.predict(scaled_testing_values)
            
            
        # Print Cluster Prediction Analysis
        if False:
            location_outside_cluster = ratio_test_index\
                                        [cluster_predictions == -1]
            cluster_win_perc = ratio_ml_outcomes.loc[location_outside_cluster]
            cluster_win_perc = cluster_win_perc.mean()
            groups = get_groups(np.arange(cluster_predictions.shape[0])\
                                [cluster_predictions == -1], 100)
            print('Percentage Outside of cluster: {}'.format(\
                                          (cluster_predictions == -1).mean()))
            print('Number Outside of cluster: {}'.format(\
                                          (cluster_predictions == -1).sum()))
            print('Placement Groups: {}'.format(groups.shape[0]))
            print(ratio_outcomes.loc[ratio_test_index]\
                  .loc[location_outside_cluster].mean())
            # Plot distribution
            sns.distplot(decision)
        
        
        # Combine cluster techniques - any new information ? 
        if False:
            combined_clusters = (forest_isolation_prediction == -1) \
                              & (envelope_isolation_prediction == -1)
                              
            print(ratio_outcomes.loc[ratio_test_index]\
                  .loc[combined_clusters].mean())


        # Export Cluster data fro Analysis
        if False:
            ml_export = ratio_ml.loc[ratio_test_index].copy()
            ml_export['env_measure'] = env_decision
            ml_export['forest_measure'] = forest_decision
            ml_export['forest_clusters'] = forest_isolation_prediction
            ml_export['env_clusters'] = envelope_isolation_prediction
            ml_export['outcomes'] = ratio_ml_outcomes.loc[ratio_test_index]
            ml_export.to_csv('/Users/user/Desktop/to_plot/cluster_analysis.csv')
        
    
    # =======================================================================
    # Ratio outcomes model based on currency prediction probabilities
    # =======================================================================
    if True:
    
        # ==== Complete Build of DF with Correlation & ratio indicators ===
        
        # Final ratio_ml parameters
        use_ratio = 0
        use_cor   = 1
        rc_columns = [0, 1, 2, 3]
    
        # Add data to df
        dfs = []
        if use_ratio:
            ratio_ml['ratio'] = ratio_preds
        if use_cor:
            rc = rolling_correlation.iloc[test_index, rc_columns]
            ratio_ml = ratio_ml.join(rc, lsuffix='.')
        
        
        # ==== Design and implement model ===
        
        # Split Data         
        x_train    = ratio_ml.loc[ratio_train_index]
        y_train    = ratio_ml_outcomes.loc[ratio_train_index]
        x_test     = ratio_ml.loc[ratio_test_index]
        y_test     = ratio_ml_outcomes.loc[ratio_test_index]
        
        
        # Scale 
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        
        
        # Implemnent Model 
        logreg = GaussianNB()
        logreg = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(25, 25), random_state=1)
        logreg = LogisticRegression(C = 1, solver = 'sag', n_jobs = 4)
        logreg.fit(x_train, y_train)
        ratio_predictions = logreg.predict(x_test)
    
    
        # ==== Show Analysis ===
    
        # Print analysis
        months = x_test.shape[0] / 1440 / 20
        print('\nValidation TimeFrame in Months: {:.2f}\n'.format(months))
        print(cr(y_test, ratio_predictions))
        
        # Print Probability slices and distribution
        print('\n Slice, Shape, groups, win %\n-----------------------------')
        ratio_pred_probs = logreg.predict_proba(x_test)[:, 1]
        step = .02
        ratio_ml_indexes = []
        plt.figure(figsize=(10,5))
        for _slice in np.arange(.5, 1, step).round(3):
            index = ratio_pred_probs > _slice
            ratio_ml_indexes.append(index)
            try:
                groups = get_groups(np.arange(index.shape[0])[index], 
                                    300).shape[0]
                plt.plot(['nan' if float(x) == 0 else x for x in\
                          (pred_probs > _slice).astype(int) * _slice], 'o') 
                print('{}\t{}\t{}\t{:2f}'.format(_slice, 
                                                 index.sum(), 
                                                 groups,
                                                 y_test[index].mean()))
            except:
                pass
        plt.title('test')
        plt.show()
        
        
        # Print on all
        print(ratio_outcomes.loc[ratio_test_index].loc[predictions].mean())
        
        # Print on threshold
        print(ratio_outcomes.loc[ratio_test_index].loc[ratio_pred_probs > .52].mean())
        
        
        
        

        
        
        # =======  Combined cluster and prediciton analysis ===============   
        if True:
                
            # Gather results where outcome prediciton is 1 and out of cluster
            cond1 = cluster_predictions == -1
            cond2 = ratio_pred_probs > .5
            combined_bool_index = cond1 & cond2
            
            # Get outcome results for combined
            print('Placements:  {}'.format(combined.sum()))
            print('Placement %: {:.2f}'.format(combined.sum() / cluster_predictions.shape[0]))
            combined_outcomes = ratio_ml_outcomes.loc[ratio_test_index].copy()
            combined_outcomes = combined_outcomes.loc[combined_bool_index]
            print(combined_outcomes.mean())
        
    
    # ===================================================================
    # Export combined analysis to Tableau
    # ===================================================================
    if False:
        
        # What instrument to us
        use_eur          = 1
        use_usd          = 1
        use_ratio        = 1
        use_cor          = 1     
        # How to combine them
        use_normal       = 0
        use_subtraction  = 0
        use_column       = 1
        use_mean         = 0
    
        # Create Df for machine learning based on choices above
        ml_export = pd.DataFrame(predictions)
        ml_export.index = ratio_test_index
        columns = ['drop']
        dfs = []    
        if use_eur:
            dfs += [(eur_channel_std, 'eur_channel_std'),
                    (eur_hml, 'eur_hml'),
                    (eur_channel_pos, 'eur_channel_pos'),
                    (eur_slopes, 'eur_slopes'),
                    (eur_rolling_std, 'eur_rolling_std'),
                    (eur_rolling_pos, 'eur_rolling_pos'),
                    (eur_roll_chann_diff_pos, 'eur_roll_chann_diff_pos'),
                    (eur_roll_chann_diff_std, 'eur_roll_chann_diff_std')
                    ]
        if use_usd:
            dfs +=  [(usd_channel_std, 'usd_channel_std'),
                    (usd_hml, 'usd_hml'),
                    (usd_channel_pos, 'usd_channel_pos'),
                    (usd_slopes, 'usd_slopes'),
                    (usd_rolling_std, 'usd_rolling_std'),
                    (usd_rolling_pos, 'usd_rolling_pos'),
                    (usd_roll_chann_diff_pos, 'usd_roll_chann_diff_pos'),
                    (usd_roll_chann_diff_std, 'usd_roll_chann_diff_std')
                    ]
        if use_ratio:
            dfs +=  [(ratio_channel_std, 'ratio_channel_std'),
                    (ratio_channel_pos, 'ratio_channel_pos'),
                    (ratio_slopes, 'ratio_slopes'),
                    (ratio_rolling_std, 'ratio_rolling_std'),
                    (ratio_rolling_pos, 'ratio_rolling_pos'),
                    (ratio_roll_chann_diff_pos, 'ratio_roll_chann_diff_pos'),
                    (ratio_roll_chann_diff_std, 'ratio_roll_chann_diff_std'),
                    ]
        if use_cor:
            dfs += [(rolling_correlation, 'rolling_correlation')]
        # Choose how to use them
        for df in dfs:
            dataframe = df[0] # Removed scaling however - kept name of variable
            if use_normal:
                ml_export = ml_export.join(dataframe.loc[ratio_test_index], lsuffix='.')
                for col in df[0].columns:
                    columns.append(df[1] + '_' + str(col))
            if use_column:
                ml_export = ml_export.join(dataframe.iloc[ratio_test_index, window_columns], lsuffix='.')
                for i in range(len(window_columns)):
                    columns.append(df[1] + '_' + str(window_columns[i]))   
            if use_subtraction:
                    sub = dataframe.iloc[ratio_test_index, 7] - dataframe.iloc[ratio_test_index, 3]
                    sub.name = df[1]
                    ml_export = ml_export.join(sub, lsuffix='.')
                    columns.append(df[1] + '_mean_sub')
            if use_mean:
                 to_app = dataframe.loc[ratio_test_index].mean(axis=1)
                 to_app.name = df[1]
                 ml_export = ml_export.join(to_app, lsuffix='.')
                 columns.append(df[1])
        # Finalize ml_export Creation
        ml_export.columns = columns
        ml_export.drop('drop', inplace=True, axis = 1)
        
        # Combine all data from previous to one ml.
        ml_export['eur'] = ratio_ml.eur
        ml_export['usd'] = ratio_ml.usd
        ml_export['outcomes'] = ratio_ml_outcomes
        ml_export['predictions'] = ratio_pred_probs
        ml_export['cluster_predictions'] = cluster_predictions
        ml_export.to_csv('/Users/user/Desktop/to_plot/ratio_analysis.csv')
        
        
        


        


    # = Final Predictions just based on currency prediction Thresholds ====
    if True:
        
        # Currency Probability Sums greater than 
        threshold = .55
        print('\n-------------------------------------------------------')
        print('Currency Probability Sums greater than ' + str(threshold))
        cond1 = ratio_ml.loc[ratio_test_index, 'eur'] > threshold
        cond2 = ratio_ml.loc[ratio_test_index, 'usd'] > threshold
        cur_index = ratio_ml.loc[ratio_test_index].loc[cond1 & cond2].index.values
        groups = get_groups(cur_index, 60)
        print('Placements and Groups:\t{}\t{}'.format(cur_index.shape[0],
                                                      groups.shape[0]))
        print(ratio_ml_outcomes.loc[cur_index].mean())


        # All ratio_ml Probability Sums greater than
        threshold = 4.5        
        print('\n-------------------------------------------------------')
        print('All ratio_ml Probability Sums greater tha' + str(threshold))
        ratio_ml_long = ratio_test
        ratio_ml_long.index = ratio_ml.index
        all_index = ratio_ml[ratio_ml.sum(axis=1) > threshold].index.values
        print('Placements and Groups:{}\t\t{}'.format(all_index.shape[0],
                                                      groups.shape[0]))
        print(ratio_ml_long.loc[all_index].mean())   
        
        
        # Plot two previous distributions
        plt.figure(figsize=(10, 1))
        cur_dist = np.ones(ratio_ml.eur.shape[0]) * np.nan
        all_dist = np.ones(ratio_ml.eur.shape[0]) * np.nan
        cur_dist[cur_index] = .5
        all_dist[all_index] = 1
        plt.plot(cur_dist, 'o', label='currencies')
        plt.plot(all_dist, 'o', label = 'all ratio_ml')
        plt.title('Distribution on thresholds')
        plt.legend()
        plt.show()
        
 
        


# =============================================================================
# Export
# =============================================================================
if False:
    
    eur_channel_std.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_channel_std.pkl')
    eur_channel_pos.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_channel_pos.pkl')
    eur_channel_mean.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_channel_mean.pkl')
    eur_slopes.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_slopes.pkl')
    eur_rolling_std.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_rolling_std.pkl')
    eur_rolling_pos.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_rolling_pos.pkl')
    eur_rolling_mean.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_rolling_mean.pkl')

    ratio_channel_std.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_channel_std.pkl')
    ratio_channel_pos.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_channel_pos.pkl')
    ratio_channel_mean.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_channel_mean.pkl')
    ratio_slopes.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_slopes.pkl')
    ratio_rolling_std.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_rolling_std.pkl')
    ratio_rolling_pos.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_rolling_pos.pkl')
    ratio_rolling_mean.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_rolling_mean.pkl')

    usd_channel_std.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_channel_std.pkl')
    usd_channel_pos.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_channel_pos.pkl')
    usd_channel_mean.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_channel_mean.pkl')
    usd_slopes.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_slopes.pkl')
    usd_rolling_std.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_rolling_std.pkl')
    usd_rolling_pos.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_rolling_pos.pkl')
    usd_rolling_mean.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_rolling_mean.pkl')

    eur_long.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_long.pkl')
    eur_short.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_short.pkl')
    eur_down.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_down.pkl')
    eur_up.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_up.pkl')
    eur_minimums.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_minimums.pkl')

    usd_long.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_long.pkl')
    usd_short.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_short.pkl')
    usd_down.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_down.pkl')
    usd_up.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_up.pkl')
    usd_minimums.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_minimums.pkl')

    ratio_long.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_long.pkl')
    ratio_short.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_short.pkl')
    ratio_down.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_down.pkl')
    ratio_up.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_up.pkl')
    ratio_minimums.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratio_minimums.pkl')

    cur.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/cur.pkl')
    ratios.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/ratios.pkl')
    high.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/high.pkl')
    low.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/low.pkl')
    volume.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/volume.pkl')
    timestamp.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/timestamp.pkl')

    rolling_correlation.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/rolling_correlation.pkl')
    eur_hml.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/eur_hml.pkl')
    usd_hml.to_pickle('/Users/user/Desktop/2_year_eur_usd_currencies_and_ratio/usd_hml.pkl')



# =============================================================================
# Import
# =============================================================================
if False:        

    eur_channel_std = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_channel_std.pkl')
    eur_channel_pos = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_channel_pos.pkl')
    eur_channel_mean = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_channel_mean.pkl')
    eur_slopes = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_slopes.pkl')
    eur_rolling_std = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_rolling_std.pkl')
    eur_rolling_pos = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_rolling_pos.pkl')
    eur_rolling_mean = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_rolling_mean.pkl')

    usd_channel_std = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_channel_std.pkl')
    usd_channel_pos = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_channel_pos.pkl')
    usd_channel_mean = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_channel_mean.pkl')
    usd_slopes = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_slopes.pkl')
    usd_rolling_std = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_rolling_std.pkl')
    usd_rolling_pos = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_rolling_pos.pkl')
    usd_rolling_mean = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_rolling_mean.pkl')

    ratio_channel_std = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_channel_std.pkl')
    ratio_channel_pos = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_channel_pos.pkl')
    ratio_channel_mean = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_channel_mean.pkl')
    ratio_slopes = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_slopes.pkl')
    ratio_rolling_std = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_rolling_std.pkl')
    ratio_rolling_pos = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_rolling_pos.pkl')
    ratio_rolling_mean = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_rolling_mean.pkl')

    eur_long = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_long.pkl')
    eur_short = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_short.pkl')
    eur_down = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_down.pkl')
    eur_up = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_up.pkl')
    eur_minimums = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_minimums.pkl')

    usd_long = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_long.pkl')
    usd_short = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_short.pkl')
    usd_down = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_down.pkl')
    usd_up = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_up.pkl')
    usd_minimums = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_minimums.pkl')

    ratio_long = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_long.pkl')
    ratio_short = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_short.pkl')
    ratio_down = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_down.pkl')
    ratio_up = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_up.pkl')
    ratio_minimums = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratio_minimums.pkl')

    cur = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/cur.pkl')
    ratios = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/ratios.pkl')
    high = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/high.pkl')
    low = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/low.pkl')
    volume = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/volume.pkl')
    timestamp = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/timestamp.pkl')

    rolling_correlation = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/rolling_correlation.pkl')
    eur_hml = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/eur_hml.pkl')
    usd_hml = pd.read_pickle('/Users/user/Desktop/10_year_currency_ratio/usd_hml.pkl')
    
    # Rolling Channel Difference
    usd_roll_chann_diff_mean =  usd_rolling_mean - usd_channel_mean
    usd_roll_chann_diff_pos  =  usd_rolling_pos - usd_channel_pos
    usd_roll_chann_diff_std  =  usd_rolling_std - usd_channel_std    

    # Rolling Channel Difference
    eur_roll_chann_diff_mean =  eur_rolling_mean - eur_channel_mean
    eur_roll_chann_diff_pos  =  eur_rolling_pos - eur_channel_pos
    eur_roll_chann_diff_std  =  eur_rolling_std - eur_channel_std

    # Rolling Channel Difference
    ratio_roll_chann_diff_mean =  ratio_rolling_mean - ratio_channel_mean
    ratio_roll_chann_diff_pos  =  ratio_rolling_pos - ratio_channel_pos
    ratio_roll_chann_diff_std  =  ratio_rolling_std - ratio_channel_std   
    
    
    # Currencies and instrument
    eur = cur.loc[:, 'eur']
    usd = cur.loc[:, 'usd']
    ratio = ratios.loc[:, 'eur_usd']
    
    # Indicators and Outcomes
    currency_targets = eur_long.columns
    ratio_targets = ratio_long.columns
    windows = eur_slopes.columns

    
    
    # Standardize slopes
    slope_vals = StandardScaler().fit_transform(usd_slopes)
    usd_slope_std     = pd.DataFrame(slope_vals,
                                     columns = usd_slopes.columns,
                                     index = usd_slopes.index)
    # Standardize std
    slope_vals = MinMaxScaler().fit_transform(usd_rolling_std)
    usd_rolling_std_std     = pd.DataFrame(slope_vals,
                                     columns = usd_rolling_std.columns,
                                     index = usd_rolling_std.index)
    
    
    # Standardize slopes
    slope_vals = StandardScaler().fit_transform(eur_slopes)
    eur_slope_std     = pd.DataFrame(slope_vals,
                                     columns = eur_slopes.columns,
                                     index = eur_slopes.index)
    # Standardize std
    slope_vals = MinMaxScaler().fit_transform(eur_rolling_std)
    eur_rolling_std_std     = pd.DataFrame(slope_vals,
                                     columns = eur_rolling_std.columns,
                                     index = eur_rolling_std.index)
    
    

    
    
   