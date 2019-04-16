###############################################################################
# Notes
###############################################################################
if 0:

    '''
    order book
    
    Try to find what a currency might do by 
    totalling up long open positions from order book
    
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
    from libraries.oanda import get_orderbook
    
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
    _to   = '2018-12-01T00:00:00Z'
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
# Find long orders for all with Eur at one time
###############################################################################    
if 1:   

    currency = 'usd'
    #start = cu.first_valid_index()
    start = pd.to_datetime('2018-12-12T14:00:00Z')


    pairs = ['AUD_USD',
             'EUR_USD',
             
             'GBP_USD',
             'NZD_USD',
             'USD_CAD',
             
             'USD_CHF']
    currency = 'usd'
    helper = [-1, -1, -1, -1, 1, 1]



    pairs = ['EUR_AUD',
                #     'EUR_CAD',
                     'EUR_CHF',
                     'EUR_GBP',
                #     'EUR_HKD',
                #     'EUR_JPY',
                #     'EUR_NZD',
                #     'EUR_SGD',
                     'EUR_USD']
    currency = 'eur'
    helper = [1, 1, 1, 1]
          
    
    currency_book = pd.DataFrame()
    for i in range(len(pairs)):
         
        instrument=pairs[i]
        timestamp = start 
        timestamp = str(timestamp).replace(' ', 'T') + 'Z'
        orderbook = get_orderbook(instrument, timestamp)['orderBook']
        
        price_loc = []
        longs = []
        shorts = []
        for bucket in orderbook['buckets']:
            price_loc.append(bucket['price'])
            longs.append(bucket['longCountPercent'])
            shorts.append(bucket['shortCountPercent'])
        
        book = np.c_[price_loc, longs, shorts].astype(float)
        book = pd.DataFrame(book, columns = ['location', 'long', 'short'])
        
        interval = 100
        price = float(orderbook['price'])
        location = np.argmax(book.location.values > price)
        if helper[i] == 1:
            vals = book.loc[location - interval: location + interval, 'long']\
                 - book.loc[location - interval: location + interval, 'short']
        else:
            vals = book.loc[location - interval: location + interval, 'short']\
                 - book.loc[location - interval: location + interval, 'long']  
        currency_book[instrument] = vals.values
    
        print(instrument)
        print(orderbook['bucketWidth']) 
        print(timestamp)
        print(location)
        print()





###############################################################################
# Plot above data SUm and Currency
###############################################################################   
if 1:
    
    currency_book.mean(axis=1).plot()
    currency_book.mean(axis=1).plot(style='o')
    plt.plot(np.zeros(currency_book.shape[0]), color='grey')
    plt.plot(100, 0, 'o', color='black')
    
    plt.figure()
    plt_hours = 24
    end = pd.to_datetime(start) + pd.Timedelta(hours=plt_hours)
    cu.loc[start: end, currency].plot()
    
    print(currency_book.mean().mean())








































###############################################################################
# Call and collect the order book
###############################################################################    
if 0:   

    instrument = 'EUR_USD' 
    
    start = pd.to_datetime('2018-11-05T00:00:00Z')
    start = cu.first_valid_index()
    net_coll = []
    timestamp_coll = []
    for i in range(100):# in cu,.index:
        
        try:
            print(i)
            timestamp = start + pd.Timedelta(hours=i)
            timestamp = str(timestamp).replace(' ', 'T') + 'Z'
            timestamp_coll.append(timestamp)
            orderbook = get_orderbook(instrument, timestamp)['orderBook']
            print(orderbook['bucketWidth']) # Verification
            print(timestamp)
            
            price = float(orderbook['price'])
            price_loc = []
            longs = []
            shorts = []
            net = []
            for bucket in orderbook['buckets']:
                price_loc.append(bucket['price'])
                longs.append(bucket['longCountPercent'])
                shorts.append(bucket['shortCountPercent'])
            book = np.c_[price_loc, longs, shorts].astype(float)
        
            
            # Testing only
            book = pd.DataFrame(book, columns = ['location', 'long', 'short'])
            
            
            interval = 200
            location = np.argmax(book.location.values > price)
    
            vals = (book.loc[location - interval: location + interval, 'long']\
                 - book.loc[location - interval: location + interval, 'short'])\
                   .values.tolist()
        except:
            vals = np.zeros(interval + 1).tolist()
        net_coll.append(vals)

    volume_ = pd.DataFrame(net_coll).T
    
    
    
    
    
    
    
    
    


###############################################################################
# Some plotting experimentation
###############################################################################    
if 0:  
    
    '''
    So, for instance, we want to find buy orders below price
    '''
    
    
    plt_hours = 24

    plt_interval = 50   # Print books in time sequence
    for i in range(volume.columns.shape[0]):
        plt.figure()
        volume.iloc[interval - plt_interval:interval + plt_interval, i].plot(ylim = ((-3, 3)))
        plt.plot([interval, interval], [-1, 1], color='black')
        plt.show()
        
        
        maximum = abs(volume.iloc[:, i].idxmax() - interval)
        maximum *= - .0005
        minimum = abs(volume.iloc[:, i].idxmin() - interval)
        minimum *= .0005
        
        try:
            
            if ratios.loc[pd.to_datetime(timestamp_coll[i])].shape[0] > 0:
                
                begin = pd.to_datetime(timestamp_coll[i]) - pd.Timedelta(hours=plt_hours)
                end = pd.to_datetime(timestamp_coll[i])
                ratios.loc[(cu.index > begin) & (cu.index < end), 'EUR_USD'].plot()
#               plt.plot(pd.to_datetime(timestamp_coll[i]),
#                         ratios.loc[pd.to_datetime(timestamp_coll[i]), 'EUR_USD'], 'o')
                
                
                
                '''
                plt.plot(ratios.loc[(cu.index > begin) & (cu.index < end), 'EUR_USD'].index,
                        np.ones(ratios.loc[(cu.index > begin) & (cu.index < end), 'EUR_USD'].shape[0])\
                        * (maximum + ratios.loc[pd.to_datetime(timestamp_coll[i]), 'EUR_USD']))
                plt.plot(ratios.loc[(cu.index > begin) & (cu.index < end), 'EUR_USD'].index,
                        np.ones(ratios.loc[(cu.index > begin) & (cu.index < end), 'EUR_USD'].shape[0])\
                        * (minimum + ratios.loc[pd.to_datetime(timestamp_coll[i]), 'EUR_USD']))
                '''
                
                
                
#                maximum = abs((volume.loc[ volume.iloc[:, i] > 0, i].index.values  * volume.loc[ volume.iloc[:, i] > 0, i].values).mean())
#                minimum = abs((volume.loc[ volume.iloc[:, i] < 0, i].index.values  * volume.loc[ volume.iloc[:, i] < 0, i].values).mean())
#               
                perc = ((volume.loc[ volume.iloc[:, 0] > 0, 0].index.values  * volume.loc[ volume.iloc[:, 0] > 0, 0].values) / (volume.loc[ volume.iloc[:, 0] > 0, 0].index.values  * volume.loc[ volume.iloc[:, 0] > 0, 0].values).sum())
                maximum = (volume.loc[ volume.iloc[:, 0] > 0, 0].index.values  * perc).mean()
                perc = ((volume.loc[ volume.iloc[:, 0] < 0, 0].index.values  * volume.loc[ volume.iloc[:, 0] < 0, 0].values) / (volume.loc[ volume.iloc[:, 0] < 0, 0].index.values  * volume.loc[ volume.iloc[:, 0] < 0, 0].values).sum())
                minimum = (volume.loc[ volume.iloc[:, 0] < 0, 0].index.values  * perc).mean()
                maximum *= - .0005
                minimum *= .0005


                plt.plot(ratios.loc[(cu.index > begin) & (cu.index < end), 'EUR_USD'].index,
                        np.ones(ratios.loc[(cu.index > begin) & (cu.index < end), 'EUR_USD'].shape[0])\
                        * (maximum + ratios.loc[pd.to_datetime(timestamp_coll[i]), 'EUR_USD']))
                plt.plot(ratios.loc[(cu.index > begin) & (cu.index < end), 'EUR_USD'].index,
                        np.ones(ratios.loc[(cu.index > begin) & (cu.index < end), 'EUR_USD'].shape[0])\
                        * (minimum + ratios.loc[pd.to_datetime(timestamp_coll[i]), 'EUR_USD']))
                
                
        except:
            pass
        
        print('a')
        raw_input = input('touch')
        if raw_input == 'd':
            break
        
        





















