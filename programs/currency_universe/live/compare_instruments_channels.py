###############################################################################
# Notes
###############################################################################
if True:

    '''
    
    
    FIND THE CHANNEL BREAKOUT
    
    
    Find channel breakouts using multiple channels
    
    Do this across instruments.
    
    Is a currency about to shift?
    
    Can we match two currencies? 
    
    Would also like to use order books
    
        
    
    '''


###############################################################################
# Imports 
###############################################################################
if 0:

    # Imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from time import sleep
    import seaborn as sns
    import os; os.chdir('/northbend')
    from libraries.currency_universe import get_universe_singular
    from libraries.currency_universe import backfill_with_singular
    from libraries.indicators import get_rolling_currency_correlation
    from libraries.indicators import get_rolling_mean_pos_std
    from libraries.indicators import get_channel_mean_pos_std
    from libraries.oanda import get_time
    from libraries.oanda import market
    from classes.channel import Channel
    from libraries.stats import get_distribution_boundary

    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15)
    np.warnings.filterwarnings('ignore')   
    plt.rcParams['figure.figsize'] = [14, 6]
    
    

###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 0:   

    # General Parameters
    _from = '2018-11-01T00:00:00Z'
    _to   = '2019-01-01T00:00:00Z'
    granularity = 'M5'
    # Currencies to use
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 
                  'hkd', 'jpy', 'nzd', 'sgd', 'usd']   
    
    # Get instrument List
    instrument_list = []
    for mark in market:
        if mark.split('_')[0] in [x.upper() for x in currencies]: 
            if mark.split('_')[1] in [x.upper() for x in currencies]:
                instrument_list.append(mark)    
    
    # Start with Data and Ratios Backfilled
    cu = pd.DataFrame(columns = currencies)
    cu, ratios = backfill_with_singular(currencies, granularity, _from, _to)
    cu.index.names = ['timestamp']
    ratios.index.names = ['timestamp']
    
    

###############################################################################    
# Complete all indicators over backfilled cu
###############################################################################    
if 0:
    
    # Params
    windows = np.array([15, 30, 60, 90, 120, 240, 360, 480, 960])   
    
    # Get Indicators
    rolling_pos, rolling_dev = get_rolling(cu, currencies, windows)
    correlation = get_correlation(cu, currencies, windows)
    channel_pos, channel_dev = get_channels(cu, currencies, windows)
    
    cu = cu.reset_index()
    ratios = ratios.reset_index()
    eur = cu.eur.copy()
    
###############################################################################    
# Complete all indicators over backfilled cu
###############################################################################    
if 0:
    
    make_channel_values = eur.loc[:1700].values
    channel = Channel(make_channel_values)
    
    channel_upper = get_distribution_boundary(channel.flattened, .02)['upper_bound']
    channel_lower = get_distribution_boundary(channel.flattened, .02)['lower_bound']

    channel_window = 60
    channel_position = channel_pos.loc[channel_pos.currency == 'eur']\
                       .loc[channel_pos.windows == channel_window, 'channel_pos']
    channel_position = channel_position.reset_index(drop=True)
    channel_position = pd.DataFrame(np.insert(channel_position.values, 0, [np.nan] * channel_window))
    channel_deviation = channel_dev.loc[channel_dev.currency == 'eur']\
                       .loc[channel_dev.windows == channel_window, 'channel_deviation']
    channel_deviation = channel_deviation.reset_index(drop=True)
    channel_deviation = pd.DataFrame(np.insert(channel_deviation.values, 0, [np.nan] * channel_window))


    sns.distplot(channel.flattened)
    
    plt.figure()
    plt.plot(channel.flattened)
    plt.plot(np.ones(make_channel_values.shape[0]) * channel_upper, label='upper')
    plt.plot(np.ones(make_channel_values.shape[0]) * channel_lower, label='lower')
    plt.plot(np.ones(make_channel_values.shape[0]) * 2 * channel.channel_deviation, label='(x2) deviation')
    plt.plot(np.ones(make_channel_values.shape[0]) * -2 * channel.channel_deviation, label='(x2) deviation')
    plt.plot(channel_deviation)
    plt.legend()
        
    
    plt.plot()
    
#    plt.plot(channel_position.values)

    
###############################################################################    
# Complete all indicators over backfilled cu
###############################################################################    
if 1:
    
    
    start = 1551320
    end = 1558620 
    column = 5
    mean_bound = .01
    
    e = eur.loc[start: end]
    
    dev = eur_rolling_std.iloc[start: end, column]
    mean = eur_rolling_mean.iloc[start: end, column]
    
    c_dev = eur_channel_std.iloc[start: end, column]
    c_mean = eur_channel_mean.iloc[start: end, column]
    c_pos = eur_channel_pos.iloc[start: end, column]
        
    
    avg_pos = eur_channel_pos.iloc[start: end].mean(axis=1)
    boundaries = get_distribution_boundary(avg_pos.values, mean_bound)    
    upper_index = boundaries['upper_index']
    upper_threshold = boundaries['upper_bound']
    upper_index = e.first_valid_index() + upper_index
    upper_index = upper_index[upper_index < e.last_valid_index()]
    
    
    plt.figure(); sns.distplot(avg_pos.values)
    
    plt.figure()
    e.plot()
    (mean + (2 * dev)).plot(label='roll')    
    (mean + (-2 * dev)).plot(label='roll')
    (mean + (2 * c_dev)).plot(label='channel')    
    (mean + (-2 * c_dev)).plot(label='channel')
    plt.plot(upper_index, e.loc[upper_index] , 'o')
    
    plt.legend()


    t_index = eur_channel_pos.dropna().mean(axis=1)
    t_index = t_index.loc[t_index > 4].index.values

    plt.figure()
    eur.plot()
    eur.loc[t_index].plot(style='o')
    



    
    
    

