 ###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   
    
    # Imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from time import sleep
    import os; os.chdir('/northbend')
    from libraries.currency_universe import backfill_with_singular
    from libraries.currency_universe import get_universe_singular
    from libraries.oanda import market
    from libraries.oanda import get_time
    from libraries.oanda import get_multiple_candles_spread

    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15) 
    np.warnings.filterwarnings('ignore')
    plt.rcParams['figure.figsize'] = [14, 4]
    
   
###############################################################################
# Set Parameters
###############################################################################    
if 1:     

    # Only use these currencies for placements (high leverage)   
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 
                  'hkd', 'jpy', 'nzd', 'usd']     
    
    # General Parameters
    _from = '2018-12-26T00:00:00Z'
    _to   = '2019-12-01T00:00:00Z'
    granularity = 'M1'
    
    # Get cu file from Desktop
    file = '/northbend/tmp/currencies_'
    file += granularity
    file += '.pkl'

    # Pause timer
    if granularity[1:] == '1':
        pause_time = 10
    else:
        pause_time = 20

###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   
    
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

    # Print average sum of Currencies ( must equal 1)
    print(cu.sum(axis=1).mean())

     

###############################################################################
# Fetch Currency Universe and save to disk.
###############################################################################    
if 1:     
    
    cu.to_pickle(file)
    timestamp =  cu.loc[cu.last_valid_index(), 'timestamp']
    # Look for new candle every _x_ seconds.  Update all info when found
    while True:
        # Do we have new information ?
        new_time = get_time(granularity)
        if new_time > timestamp:
            try:
                sleep(2)
                timestamp = new_time
                print('Candle Found at:\t' + str(timestamp))
                # Update ratios and cu with new data
                a, b = get_universe_singular(currencies, granularity)
                # Add line to cur
                a['timestamp'] = timestamp
                cu = cu.append(a, ignore_index=True, verify_integrity=True)
                # Export currencies
                cu.to_pickle(file)
                # Print Data
                print()
                for k, v in sorted(a.items()): print(v)
                print()
                # Print Spread
                spreads = get_multiple_candles_spread(instrument_list, granularity)
                for k, v in sorted(spreads.items()): print(v)
                print()
            except:
                plt.figure(figsize=(8,3))
                msg = 'Feeder Data has hit an error on granularity ' 
                msg += str(granularity)
                plt.text(.10, .5, msg)
                plt.pause(.01)
        else:
            print('{}: Waiting on new candle'.format(timestamp))
        sleep(pause_time)
            
            
            


    
    