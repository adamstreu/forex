###############################################################################
# Notes
###############################################################################
if True:

    '''
    
    Try:
        
        Channel breakout strategy - 
            find good channel fit then find where it leaves channel
            
            
    
    Todays watch:
    
       analysis: find when sum of difference is high, but diff between them are low 
        
        
    '''    
    
    
    
    '''
    
    Purpose:
        
        Manually make placements in real time (during morning hours)
        Have Tableau workbook to help graph
        
        
    Goal ( in order):
        
        Calculate Difference
        Create tableau workbook with all graphs desired
        Have notes on currnecy movements
            what currencies are a good match ( values are close)
            how much each need to move for half share of instru. move (@)
        update targets as they are completed
            maybe can even do with real placements - real data
        any additional indicators?
        
        
    Placement Style:
        
        Really just want to put an eye on cur and ration behavior
        Do not worry about betting specific amounts - just try to win
        maybe go for 1:1 ratios right now - just need general confirmations



    Misc:
        check flattened deviation - this is better.
        Can add rolling pos difference (lots of vals to compute)
        Perhaps can add rolling bound(threshold (rolling quantile))
        calculated ratios
        
    Anal
    ysis Notes:
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
    
    path = '/Users/user/Desktop/currencies_live_filtered/'   
    
 
    
###############################################################################
# Update Singular rows - called when new candle found
###############################################################################    
if 1:   
    
    
    def get_rolling(curreny_df, currencies, windows):
        df = curreny_df.copy()
        positions = pd.DataFrame()
        deviations = pd.DataFrame()
        for cur in currencies:
            roll = get_rolling_mean_pos_std(df[cur], windows)
            pos = roll['pos']
            std = roll['std']
            pos.index = df.index
            std.index = df.index
            # Get Positions stacked
            pos_stack = pd.DataFrame(pos.stack())
            pos_stack.index.names = ['timestamp', 'windows']
            pos_stack.columns = ['rolling_position']
            pos_stack['currency'] = cur
            # Get Deviations stacked        
            dev_stack = pd.DataFrame(std.stack())
            dev_stack.index.names = ['timestamp', 'windows']
            dev_stack.columns = ['rolling_deviation']
            dev_stack['currency'] = cur
            # Concat to collections
            positions = pd.concat((positions, pos_stack))
            deviations = pd.concat((deviations, dev_stack))
        positions.reset_index(inplace=True)
        deviations.reset_index(inplace=True)
        return positions, deviations

        
    def get_channels(curreny_df, currencies, windows):
        df = curreny_df.copy()
        positions = pd.DataFrame()
        deviations = pd.DataFrame()
        for cur in currencies:
            roll = get_channel_mean_pos_std(df[cur], windows)
            pos = roll['pos']
            std = roll['std']
            pos.index = df.index
            std.index = df.index
            # Get Positions stacked
            pos_stack = pd.DataFrame(pos.stack())
            pos_stack.index.names = ['timestamp', 'windows']
            pos_stack.columns = ['channel_pos']
            pos_stack['currency'] = cur
            # Get Deviations stacked        
            dev_stack = pd.DataFrame(std.stack())
            dev_stack.index.names = ['timestamp', 'windows']
            dev_stack.columns = ['channel_deviation']
            dev_stack['currency'] = cur
            # Concat to collections
            positions = pd.concat((positions, pos_stack))
            deviations = pd.concat((deviations, dev_stack))
        positions.reset_index(inplace=True)
        deviations.reset_index(inplace=True)
        return positions, deviations
    
    
    def get_correlation(curreny_df, currencies, windows):
        correlation = pd.DataFrame()
        for pair in instrument_list:
            split = pair.split('_')
            rolling_correlation = get_rolling_currency_correlation(\
                                       curreny_df[split[0].lower()].values, 
                                       curreny_df[split[1].lower()].values, 
                                       windows)
            rolling_correlation.index = curreny_df.index
            # Get Positions stacked
            corr_stack = pd.DataFrame(rolling_correlation.stack())
            corr_stack.index.names = ['timestamp', 'windows']
            corr_stack.columns = ['correlation']
            corr_stack['instrument'] = pair
            correlation = pd.concat((correlation, corr_stack))
        correlation.reset_index(inplace=True)
        return correlation
    

    def get_diff(df, windows, instrument_list):
        # Get Column name to work with
        column = df.loc[df.currency == 'eur']\
                 .unstack().columns.levels[0].values
        column = column[column != 'currency'][0]
        pairs = pd.DataFrame()
        for instrument in instrument_list:
            cur1, cur2 = list(map(lambda x: x.lower(), instrument.split('_')))        

            a = df.loc[df.currency == cur1]
            a.reset_index(inplace=True)
            a = a.pivot(index='timestamp', columns = 'windows', values=column)
            
            b = df.loc[df.currency == cur2]
            b.reset_index(inplace=True)
            b = b.pivot(index='timestamp', columns = 'windows', values=column)
    
            diff = a - b
            diff_stack = pd.DataFrame(diff.stack())
            diff_stack['instrument'] = instrument
            
            pairs = pd.concat((pairs, diff_stack))
        pairs.reset_index(inplace=True)
        return pairs
        
    
    
    def update(timestamp, cu, ratios, rolling_pos, rolling_dev, channel_pos, 
               channel_dev, correlation):

        # Update currency and ratios
        a, b = get_universe_singular(currencies, granularity)
        cu.loc[timestamp] = a
        ratios.loc[timestamp] = b   
        
        # Update rolling based on new currencies
        r_p, r_d = get_rolling(cu.tail(windows[-1] + 10), currencies, windows)
        rolling_pos = rolling_pos.append(r_p.loc[r_p.timestamp == timestamp], 
                                         ignore_index=True)
        rolling_dev = rolling_dev.append(r_d.loc[r_d.timestamp == timestamp], 
                                         ignore_index=True)

        # Update Correlation ased on new currencies
        c = get_correlation(cu.tail(windows[-1] + 10), currencies, windows)
        correlation = correlation.append(c.loc[c.timestamp == timestamp], 
                                         ignore_index=True)
    
        # Update channels based on new currencies
        ch_p, ch_d = get_channels(cu.tail(windows[-1] + 10), currencies, windows)
        channel_pos = channel_pos.append(ch_p.loc[ch_p.timestamp == timestamp], 
                                     ignore_index=True)
        channel_dev = channel_dev.append(ch_d.loc[ch_d.timestamp == timestamp], 
                                         ignore_index=True)
        
        
        # Append csv Files for update
        pd.DataFrame(cu.loc[pd.to_datetime(timestamp)]).T\
                    .to_csv('/Users/user/Desktop/test.csv',
                    mode='a', header=False, index=True)
        pd.DataFrame(ratios.loc[pd.to_datetime(timestamp)]).T\
                    .to_csv('/Users/user/Desktop/test.csv',
                    mode='a', header=False, index=True)
        rolling_pos.loc[rolling_pos.timestamp == timestamp]\
                    .to_csv(path + 'rolling_pos.csv', index=False,
                            header=False, mode='a')
        rolling_dev.loc[rolling_dev.timestamp == timestamp]\
                    .to_csv(path + 'rolling_dev.csv', index=False,
                            header=False, mode='a')                    
        channel_pos.loc[channel_pos.timestamp == timestamp]\
                    .to_csv(path + 'channel_pos.csv', index=False,
                            header=False, mode='a')        
        channel_dev.loc[channel_dev.timestamp == timestamp]\
                    .to_csv(path + 'channel_dev.csv', index=False,
                            header=False, mode='a')        
        correlation.loc[correlation.timestamp == timestamp]\
                    .to_csv(path + 'correlation.csv', index=False,
                            header=False, mode='a') 

        
        return {   
                'cu': cu,
                'ratios' : ratios,
                'rolling_pos': rolling_pos,
                'rolling_dev': rolling_dev,
                'correlation': correlation,
                'channel_pos': channel_pos,
                'channel_dev': channel_dev
                }
        
        

###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   

    # General Parameters
    _from = '2018-12-05T00:00:00Z'
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
    cu.reset_index(inplace=True)
    ratios.reset_index(inplace=True)
    ratios.iloc[:, 1] = ratios.iloc[:, 1:].astype(float)
    cu.iloc[:, 1] = cu.iloc[:, 1:].astype(float)

    
###############################################################################    
# Complete all indicators over backfilled cu
###############################################################################    
if 1:
    
    # Params
    windows = np.array([15, 30, 60, 90, 120, 240, 360, 480])   
    
    # Get Indicators
    rolling_pos, rolling_dev = get_rolling(cu, currencies, windows)
    correlation = get_correlation(cu, currencies, windows)
    channel_pos, channel_dev = get_channels(cu, currencies, windows)
    
    
    
###############################################################################    
# Export all df's for first time
###############################################################################    
if 1:
    
    # Export cu and ratios
    c = cu.stack().reset_index()
    r = ratios.stack().reset_index()
    c.columns = ['timestamp', 'currency', 'currency_values']
    r.columns = ['timestamp', 'instrument', 'instrument_values']
    c.to_csv(path + 'currencies.csv', index=False)  
    r.to_csv(path + 'ratios.csv', index=False)
    # Export indicators
    rolling_pos.to_csv(path + 'rolling_pos.csv', index=False)
    rolling_dev.to_csv(path + 'rolling_dev.csv', index=False)
    correlation.to_csv(path + 'correlation.csv', index=False)
    channel_pos.to_csv(path + 'channel_pos.csv', index=False)
    channel_dev.to_csv(path + 'channel_dev.csv', index=False)

    


###############################################################################
# Update All Info in real time
###############################################################################
if 1:
    
    # Parameters
    pause_time = 25
    timestamp =  cu.last_valid_index()

    # Look for new candle every _x_ seconds.  Update all info when found
    while True:
        # Do we have new information ?
        new_time = get_time(granularity)
        if new_time > timestamp:
            timestamp = new_time
            print('Candle Found at:\t' + str(timestamp))
            # Update all df's with new oanda data
            up = update(timestamp,
                        cu, 
                        ratios,
                        rolling_pos,
                        rolling_dev, 
                        channel_pos, 
                        channel_dev, 
                        correlation)
            cu = up['cu']
            ratios = up['ratios']
            rolling_pos = up['rolling_pos']
            rolling_dev = up['rolling_dev']
            correlation = up['correlation']
            channel_pos = up['channel_pos']
            channel_dev = up['channel_dev'] 
        else:
            print('waiting on new candle: ' + str(timestamp))
        sleep(pause_time)
                        
















































