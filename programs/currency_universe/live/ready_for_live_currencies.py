###############################################################################
# Notes
###############################################################################
if True:

    
    '''
    
    Purpose:
        
        Manually make placements in real time (during morning hours)
        Have Tableau workbook to help graph
        
        
    Goal:
        
        Do all of this again so I can just change out the values
        x    Complete functions to update in real time and export
        x    Setup subaccount for purpose
        Create (save) tableau workbook with all graphs desired
        update targets as they are completed
            maybe can even do with real placements - real data
        Have notes on currnecy movements
            what currencies are a good match ( values are close)
            how much each need to move for half share of instru. move (@)
                    
        
        
    Placement Style:
        
        Really just want to put an eye on cur and ration behavior
        Do not worry about betting specific amounts - just try to win
        maybe go for 1:1 ratios right now - just need general confirmations



    Misc:
        check flattened deviation - this is better.
        Can add rolling pos difference (lots of vals to compute)
        Perhaps can add rolling bound(threshold (rolling quantile))
        calculated ratios
        
    Analysis Notes:
        roll eur > 3 & roll_usd < -3 gives a few point bump
            combining windows helps
            Works best on short time frames
        
    '''



###############################################################################
# Imports 
###############################################################################
if 1:

    # Imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from time import sleep
    import os; os.chdir('/northbend')
    from libraries.currency_universe import get_universe_singular
    from libraries.currency_universe import backfill_with_singular
    from libraries.indicators import get_rolling_currency_correlation
    from libraries.indicators import get_rolling_mean_pos_std
    from libraries.indicators import get_channel_mean_pos_std
    from libraries.oanda import get_time
    from libraries.oanda import market
    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15)
    np.warnings.filterwarnings('ignore')   
    plt.rcParams['figure.figsize'] = [14, 6]
    
    
    
###############################################################################
# Parameters
###############################################################################
if 1:
    
    path = '/Users/user/Desktop/currencies_live/'   
    
 
    
###############################################################################
# Update Singular rows - called when new candle found
###############################################################################    
if 1:   
    
    
    def get_rolling(curreny_df, currencies, windows):
        rolling = curreny_df.copy()
        columns = currencies.copy()
        for cur in currencies:
            roll = get_rolling_mean_pos_std(curreny_df[cur], windows)
            pos = roll['pos']
            std = roll['std']
            pos.index = rolling.index
            std.index = rolling.index
            rolling = rolling.join(pos, lsuffix = '.')
            rolling = rolling.join(std, lsuffix = '.')
            [columns.append(str(cur) + '_pos_' + str(x)) \
                 for x in roll['pos'].columns]
            [columns.append(str(cur) + '_std_' + str(x)) \
                 for x in roll['pos'].columns]
        rolling.columns = columns
        rolling.drop(currencies, axis=1, inplace=True)
        return rolling
        
        
    def get_correlation(curreny_df, currencies, windows):
        # Get Correlation df with rolling posiiton and std
        correlation = curreny_df.copy()
        columns = currencies.copy()
        for pair in instrument_list:
            split = pair.split('_')
            rolling_correlation = get_rolling_currency_correlation(\
                                       cu[split[0].lower()].values, 
                                       cu[split[1].lower()].values, 
                                       windows)
            rolling_correlation.index = correlation.index
            correlation = correlation.join(rolling_correlation, lsuffix = '.')
            [columns.append('correlation_' + str(pair).lower() + '_' + str(x))\
                 for x in rolling_correlation.columns]
        correlation.columns = np.array(columns)
        correlation.drop(currencies, axis=1, inplace=True)    
        correlation.index.names = ['timestamp']
        return correlation
    
    
    def get_channels(curreny_df, currencies, windows):
        channels = curreny_df.copy()
        columns = currencies.copy()
        for cur in currencies:
            roll = get_channel_mean_pos_std(curreny_df[cur], windows)
            pos = roll['pos']
            std = roll['std']
            pos.index = channels.index
            std.index = channels.index
            channels = channels.join(pos, lsuffix = '.')
            channels = channels.join(std, lsuffix = '.')
            [columns.append(str(cur) + '_c_pos_' + str(x)) \
                 for x in roll['pos'].columns]
            [columns.append(str(cur) + '_c_std_' + str(x)) \
                 for x in roll['pos'].columns]
        channels.columns = columns
        channels.drop(currencies, axis=1, inplace=True)
        return channels
    


    
    def update(timestamp):
        # Update currency and ratios
        a, b = get_universe_singular(currencies, granularity)
        cu.loc[timestamp] = a
        ratios.loc[timestamp] = b   
        
        # Update rolling based on new currencies
        r = get_rolling(cu.tail(windows[-1] + 10), currencies, windows)
        rolling.loc[timestamp] = r.loc[r.last_valid_index()]
        
        # Update Correlation ased on new currencies
        c = get_correlation(cu.tail(windows[-1] + 10), currencies, windows)
        correlation.loc[timestamp] = c.loc[c.last_valid_index()]
    
        # Update channels based on new currencies
        ch = get_channels(cu.tail(windows[-1] + 10), currencies, windows)
        channels.loc[timestamp] = ch.loc[ch.last_valid_index()]
    
        # Export
        cu.to_csv(path + 'currencies.csv')
        ratios.to_csv(path + 'ratio.csv')
        correlation.to_csv(path + 'correlation.csv')
        rolling.to_csv(path + 'rolling.csv')
        channels.to_csv(path + 'channels.csv')
        
    

###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   

    # General Parameters
    _from = '2018-12-07T00:00:00Z'
    _to   = '2019-01-01T00:00:00Z'
    granularity = 'M1'
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
    
    # Export
    cu.index.names = ['timestamp']
    cu.to_csv(path + 'currencies.csv')
    ratios.index.names = ['timestamp']
    ratios.to_csv(path + 'ratio.csv')
    
    ''' Testing only '''
    if False:
        cu = cu.loc[cu.index[:100]]
        ratios = ratios.loc[ratios.index[:100]]
    

###############################################################################    
# Complete all indicators over backfilled cu
###############################################################################    
if 1:
    
    # Params
    windows = np.array([15, 30, 60, 90, 120, 240, 360, 480])   
    
    # Backfill rolling Indicators
    rolling = get_rolling(cu, currencies, windows)
    
    # Bavckfill Correlation indicator
    correlation = get_correlation(cu, currencies, windows)

    # Bavckfill Correlation indicator
    channels = get_channels(cu, currencies, windows)
    
    # Export
    rolling.to_csv(path + 'rolling.csv')
    correlation.to_csv(path + 'correlation.csv')
    channels.to_csv(path + 'channels.csv')

    

###############################################################################
# Update All Info in real time
###############################################################################
if 1:
    
    # Parameters
    pause_time = 10
    timestamp =  cu.last_valid_index()

    # Look for new candle every _x_ seconds.  Update all info when found
    while True:
        new_time = get_time(granularity)
        if new_time > timestamp:
            timestamp = new_time
            print('Candle Found at:\t' + str(timestamp))
            update(timestamp)
        else:
            print('waiting on new candle')
        sleep(pause_time)
                        






























###############################################################################
# some light analysis
###############################################################################
if 0:

    
    windows = np.array([15, 30, 60, 90, 120, 240, 360, 480])
    
    
    a = roll_eur.loc[:, 90]
    b = roll_usd.loc[:, 90]
    diff = a-b
    df = pd.DataFrame()
    df['pos'] = a
    df['diff'] = diff
    df.to_csv('/Users/user/Desktop/to_plot/diff_candles.csv')
    
    
    
    windows = np.array([15, 30, 60, 90, 120, 240, 360, 480])
    windows = np.array([5, 10, 15, 30, 45, 60, 90, 120, 240, 360, 480])
    roll_usd = get_rolling_mean_pos_std(cur['usd'], windows)['pos']
    roll_eur = get_rolling_mean_pos_std(cur['eur'], windows)['pos']


    for win in windows:
        cond1 = roll_eur.loc[:, win] > 3
        cond2 = roll_usd.loc[:, win] < -3
        cond1 = roll_eur.loc[:, win] < 3.5
        cond2 = roll_usd.loc[:, win] > -3
        conditions = cond1 & cond2 & cond3 & cond4
        print('--------- ' + str(win) + ' ----------')
        print(conditions.sum(), conditions.mean())
        print()
        print(ratio_short.loc[conditions].mean())
    
    
    cond1 = roll_eur.loc[:, 15] > 2
    cond2 = roll_usd.loc[:, 15] < -2
    cond3 = roll_eur.loc[:, 120] > 3
    cond4 = roll_usd.loc[:, 120] < -3
    conditions = cond1 & cond2 & cond3 & cond4
    index = cond1.loc[conditions].index.values
    groups = get_groups(index, 100)
    print(conditions.sum(), conditions.mean())
    print()
    print(ratio_short.loc[index].mean())
    print()
    print(groups.shape[0], groups.shape[0] / cur.shape[0])
    print()    
    print(ratio_short.loc[groups].mean())
    plot_index_distribution(groups)
    


    cond1 = roll_eur.loc[:, 15] > 2
    cond2 = roll_usd.loc[:, 15] < -2
    cond3 = eur_slopes.loc[:, 120] > 0
    cond4 = usd_slopes.loc[:, 120] < 0
    cond5 = eur_rolling_std > .0001
    cond6 = usd_rolling_std > .0001
    conditions = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    index = cond1.loc[conditions].index.values
    groups = get_groups(index, 100)
    print(conditions.sum(), conditions.mean())
    print()
    print(ratio_short.loc[index].mean())
    print()
    print(groups.shape[0], groups.shape[0] / cur.shape[0])
    print()    
    print(ratio_short.loc[groups].mean())
    plot_index_distribution(groups)
    




















