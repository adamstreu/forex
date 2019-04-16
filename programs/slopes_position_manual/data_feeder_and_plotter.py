###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   
    
    # Imports
    import numpy as np
    import pandas as pd
    from scipy.stats import normaltest
    import matplotlib.pyplot as plt
    from matplotlib import animation    
    from time import sleep
    import os; os.chdir('/northbend')
    from classes.channel import Channel
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
    
    # Solo interval for graphing
    interval = 240
    
    # Normal Test
    normaltest_alpha = 1e-3
    normaltest_step = 10

    # General Parameters
    _from = '2018-12-26T00:00:00Z'
    _to   = '2019-12-01T00:00:00Z'
    granularities = ['M1', 'M5', 'M15']
    update_granularity = 'M1'
    
    # Get cu file from Desktop
    path = '/northbend/tmp/currencies_'

    pause_time = 10

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
    
    for granularity in granularities:
        
        # Start with Data and Ratios Backfilled
        cu = pd.DataFrame(columns = currencies)
        cu, ratios = backfill_with_singular(currencies, granularity,_from, _to)
        cu.index.names = ['timestamp']
        ratios.index.names = ['timestamp']
        cu.reset_index(inplace=True)
        ratios.reset_index(inplace=True)
        ratios.iloc[:, 1] = ratios.iloc[:, 1:].astype(float)
        cu.iloc[:, 1] = cu.iloc[:, 1:].astype(float)

        # Export data as pickle
        cu.to_pickle(path + str(granularity) + '.pkl')

        # Print average sum of Currencies ( must equal 1)
        print(str(granularity) + ': ' + str(cu.sum(axis=1).mean()))




###############################################################################
# Animation Plot Function
###############################################################################    
def run(data, granularity, path=path, interval=interval, 
        normaltest_step=normaltest_step,
        normaltest_alpha=normaltest_alpha):
    
    print(granularity)

    # Import Data
    df = pd.read_pickle(path + granularity + '.pkl')
    plot_index = np.arange(df.last_valid_index() - interval, 
                           df.last_valid_index() + 1)
    
    # Plot Data
    for i in range(len(ax)):
        ax[i].cla()
        df.iloc[plot_index, 1+i].plot(ax=ax[i])
        
    # Add labels
    for i in range(ax.shape[0]):
        ax[i].set_ylabel(df.columns[i + 1], rotation=90, size='large', color='white')
        ax[i].yaxis.tick_right()
        ax[i].tick_params(axis='y', colors='white')

    # get channels for currency
    for i in range(1, df.columns.shape[0]):
        coll = []
        currency = df.columns[i]
        for s in range(20, interval + normaltest_step, normaltest_step):
            channel = Channel(df.loc[df.last_valid_index() - s: \
                                     df.last_valid_index(), currency].values)
            k2, p = normaltest(channel.flattened / channel.channel_deviation)
            #print("p = {:g}".format(p))
            if p < normaltest_alpha:
                pass # "The null hypothesis can be rejected")
            else:
                coll.append(s) # "The null hypothesis cannot be rejected")
        coll = np.array(coll)
    
        # Get Positions for channel breaks
        keep = []
        for k in range(coll.shape[0] - 1):
            if coll[k + 1] > coll[k] + normaltest_step:
                keep.append(coll[k])
        if coll.shape[0] > 0: keep.append(coll[-1])
        keep = np.array(keep)
        
        # Plot chnnels by normal break
        for k in keep[::-1]:
            try:
                channel = Channel(df.loc[df.last_valid_index() - k:, currency].values)
                df.loc[df.last_valid_index() - (k + 1):, currency].plot(ax=ax.ravel()[i-1], marker='|', markersize=.5)
                line = channel.line
                line_x = np.arange(df.last_valid_index() - (k) , df.last_valid_index() + 1)
                ax.ravel()[i-1].plot(line_x, line, color='grey', linewidth=.5)
                ax.ravel()[i-1].plot(line_x, line + (np.ones(line.shape[0]) * 2 * channel.channel_deviation), color='black', linewidth=.5)
                ax.ravel()[i-1].plot(line_x, line + (np.ones(line.shape[0]) * -2 * channel.channel_deviation), color='black', linewidth=.5)
            except Exception as e:
                print(e)
                print(line_x)
                print(df.loc[df.last_valid_index() - k:, currency].index.values)
                print(channel.flattened.shape)
                print(line_x.shape)
    
    # Add line at next granularity for M15 and M5
    if granularity == 'M15':
        line_break = 3
    else:
        line_break = 5
    if granularity != 'M1':
        max_ticks = df.iloc[plot_index, i].max()
        min_ticks = df.iloc[plot_index, i].min()  
        vert = int(df.last_valid_index() - interval / line_break)
        ys = ax[-1].get_ylim()
        ax.ravel()[-1].plot([vert, vert], [ys[0], ys[0] + ((ys[1] - ys[0]) / 3)], color='black', linewidth=.5)
    
    
    # Add grid to subplots
    for i in range(1, df.columns.shape[0]):    
        max_ticks = df.iloc[plot_index, i].max()
        min_ticks = df.iloc[plot_index, i].min()
        # Different ticks for jpy and hkd
        if df.columns[i] == 'jpy':
            ticks = np.arange(min_ticks, max_ticks, .000005).round(6)
        elif df.columns[i] == 'hkd':
            ticks = np.arange(min_ticks, max_ticks, .00001).round(5)
        else:
            ticks = np.arange(min_ticks, max_ticks, .0005).round(4)
        # Create ticks    
        ax.ravel()[i - 1].grid(which='both', linewidth=.5, color='grey', b=True)    
        ax.ravel()[i - 1].set_yticks(ticks)
        
    # Print and Return
    print(df.last_valid_index())
    plt.pause(.01)

    
###############################################################################
# Fetch Currency Universe and save to disk.
###############################################################################    
if 1:         

    for granularity in granularities:
        # Import Data
        df = pd.read_pickle(path + granularity + '.pkl')
        print(path + granularity + '.pkl')
        print(df.shape)
        # Instantiate Figure
        fig, ax = plt.subplots(df.shape[1] - 1, 1, sharex=True, num=granularity,
                               facecolor='grey')
        plt.subplots_adjust(hspace=.1, wspace=.05, bottom=.05, top=.98, right=.9, left=.05)
        # Run Animation
        ani = animation.FuncAnimation(fig, run, fargs=(granularity,), blit=False, interval=15000)    
        plt.pause(5)
    
     

###############################################################################
# Fetch Currency Universe and save to disk.
###############################################################################    
if 1:     
    
    timestamp =  cu.loc[cu.last_valid_index(), 'timestamp']
    get_time(update_granularity)
    # Look for new candle every _x_ seconds.  Update all info when found
    while True:
        # Do we have new information ?
        new_time = get_time(update_granularity)
        if new_time > timestamp:
            try:
                timestamp = new_time
                print('Candle Found at:\t' + str(timestamp))
                # Update ratios and cu with new data
                a, b = get_universe_singular(currencies, granularity)
                a['timestamp'] = timestamp
                
                # Import, update, export granularities
                for granularity in granularities:
                    if timestamp.minute % int(granularity[1:]) == 0:
                        df = pd.read_pickle(path + granularity + '.pkl')
                        df = df.append(a, ignore_index=True, 
                                       verify_integrity=True)
                        df.to_pickle(path + granularity + '.pkl')
                        print('{}: Updated Granularity {}'.format(timestamp,
                              granularity))
                
                # Print Spreads and currency prices
                print()
                for k, v in a.items(): print(v)
                print()
                # Print Spread
                spreads = get_multiple_candles_spread(instrument_list, 
                                                      granularity)
                for k, v in spreads.items(): print(v)
                print()
            except:
                # Print error message if problem connecting
                plt.figure(figsize=(8,3))
                msg = 'Feeder Data has hit an error on granularity ' 
                msg += str(granularity)
                plt.text(.10, .5, msg)
                plt.pause(.01)
        else:
            print('{}: Waiting on new candle'.format(timestamp))
        sleep(pause_time)
            
            
            


    
    