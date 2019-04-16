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
    _from = '2018-06-01T00:00:00Z'
    _to   = '2020-01-01T00:00:00Z'
    granularities = ['M1', 'M5', 'M15', 'M30', 'H1']
    update_granularity = 'M1'  # Keep set here
    
    # Get cu file from Desktop
    path = '/northbend/tmp/currencies_'
    ratios_path = '/northbend/tmp/ratios_'

    # Pause timer
    pause_time = 5



###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   
    
    for granularity in granularities:
        # Get instrument List
        instrument_list = []
        for mark in market:
            if mark.split('_')[0] in [x.upper() for x in currencies]: 
                if mark.split('_')[1] in [x.upper() for x in currencies]:
                    instrument_list.append(mark)    
        
        # Start with Data and Ratios Backfilled
        cu = pd.DataFrame(columns = currencies)
        cu, ratios = backfill_with_singular(currencies, granularity, 
                                            _from, _to)
        # Export cu
        cu.index.names = ['timestamp']
        cu.reset_index(inplace=True)
        cu.iloc[:, 1] = cu.iloc[:, 1:].astype(float)
        cu.to_pickle(path + str(granularity) + '.pkl')

        # Export ratios
        ratios.index.names = ['timestamp']
        ratios.reset_index(inplace=True)
        ratios.iloc[:, 1] = ratios.iloc[:, 1:].astype(float)
        ratios.to_pickle(ratios_path + str(granularity) + '.pkl')
    
        # Print average sum of Currencies ( must equal 1)
        print(granularity)
        print(cu.sum(axis=1).mean())
        print(cu.shape)
        print()



###############################################################################
# Fetch Currency Universe and save to disk.
###############################################################################    
if 1:     
    
    timestamp =  get_time(granularities[0])
    
    # Look for new candle every _x_ seconds.  Update all info when found
    while True:
        
        # Do we have new information ?
        new_time = get_time(update_granularity)
        if new_time > timestamp:
            try:
                timestamp = new_time
                
                # Update ratios and cu with new data
                a, b = get_universe_singular(currencies, update_granularity)
                a['timestamp'] = timestamp
                b['timestamp'] = timestamp
                
                # If timestamp hits granularity, import, append and export df, rat                
                for granularity in granularities:
                    if (timestamp.minute + 1 ) % int(granularity[1:]) == 0:
                        # Currencies
                        cu = pd.read_pickle(path + str(granularity) + '.pkl')
                        cu = cu.append(a, ignore_index=True, 
                                       verify_integrity=True)
                        cu.to_pickle(path + str(granularity) + '.pkl')
                        # Ratios
                        rat = pd.read_pickle(ratios_path + str(granularity) + '.pkl')
                        rat = rat.append(b, ignore_index=True, 
                                       verify_integrity=True)
                        rat.to_pickle(ratios_path + str(granularity) + '.pkl')                        
                        # Print update
                        msg = '{}: \t Appended {}. \t Last location: {}'
                        print(msg.format(timestamp, granularity, 
                                         cu.last_valid_index()))
                        
                # Print Data prices and spread
                print()
                a.pop('timestamp')
                for k, v in sorted(a.items()): 
                    print(v)
                print()
                spreads = get_multiple_candles_spread(instrument_list, 
                                                      update_granularity)
                for k, v in sorted(spreads.items()): 
                    print(v)
                print()
            
            # Print Window Warning with connection disconnect
            except Exception as e:
                print(e)
                plt.figure(figsize=(8,3))
                msg = 'Feeder Data has hit an error on granularity ' 
                msg += str(granularity)
                plt.text(.10, .5, msg)
                plt.pause(.01)
        
        # Wait on new Timestamp
        else:
            # Add real time to M1
            a, b = get_universe_singular(currencies, 'S5')
            a['timestamp'] = timestamp
            # Add line to cur
            cu = pd.read_pickle(path + 'M1.pkl')
            for currency in currencies:
                cu.loc[cu.last_valid_index(), currency] = a[currency]
            cu.to_pickle(path + 'M1.pkl')
            msg = 'Appended M1. \t Last location: {}.  Waiting on new candle'
            print(msg.format(cu.last_valid_index()))
# Sleep in between timestamp queries
        sleep(pause_time)
            
            
            


    
    