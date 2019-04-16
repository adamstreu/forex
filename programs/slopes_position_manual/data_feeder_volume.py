###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   
    
    # Imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from time import sleep
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import os; os.chdir('/northbend')
    from libraries.oanda import get_time
    from libraries.currency_universe import get_volume_universe_singular
    from libraries.currency_universe import backfill_volume_with_singular
    from libraries.currency_universe import backfill_with_singular
    from libraries.currency_universe import get_universe_singular
    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15) 
    np.warnings.filterwarnings('ignore')
    plt.rcParams['figure.figsize'] = [14, 4]
    
    
    
    '''
    Update time is wrong - check.  Maybe in oanda
    '''
    
   
###############################################################################
# Set Parameters
###############################################################################    
if 1:     

    # Only use these currencies for placements (high leverage)   
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 
                  'hkd', 'jpy', 'nzd', 'usd']     
    
    # General Parameters
    _from = '2018-01-015T00:00:00Z'
    _to   = '2020-01-01T00:00:00Z'
    granularity = 'M1'
    
    # Get cu file from Desktop
    volume_path = '/northbend/tmp/volume.pkl'
    cu_small_path = '/northbend/tmp/cu_small.pkl'

    # Pause timer
    pause_time = 15



###############################################################################
# Backfill Volume data
###############################################################################    
if 1:   
    
    # Get historical Volume data
    volume, rat_volume = backfill_volume_with_singular(currencies, granularity, _from, _to)
    volume.reset_index(inplace=True)
    volume = volume.rename({'index':'timestamp'}, axis=1)
    volume.to_pickle(volume_path)
    
    # try standardized
    ss = StandardScaler().fit_transform(volume.iloc[:, 1:])
    ss = pd.DataFrame(ss, columns = volume.columns[1:], index=volume.index)
    ss.insert(0, 'timestamp', volume.timestamp)
    ss.drop('jpy', axis=1, inplace=True)
    ss.drop('hkd', axis=1, inplace=True)    
    
    cu, ratios = backfill_with_singular(currencies, granularity, 
                                        _from, _to)
    cu.reset_index(inplace=True)
    cu = cu.rename({'index':'timestamp'}, axis=1)
    cu.to_pickle(cu_small_path)
    
    ratios.reset_index(inplace=True)
    ratios = ratios.rename({'index':'timestamp'}, axis=1)
    
    
    
    
    
###############################################################################
# Update Volume and Diff
###############################################################################    
if 0:   
    
    timestamp = get_time(granularity)
    # Look for new candle every _x_ seconds.  Update all info when found
    while True:
        
        # Do we have new information ?
        new_time = get_time(granularity)
        if new_time > timestamp:
        
            timestamp = new_time
            
            # Update data
            a, b = get_volume_universe_singular(currencies, 
                                                granularity)
            a['timestamp'] = timestamp
            
            # Update cu with new data
            a, b = get_universe_singular(currencies, granularity)
            a['timestamp'] = timestamp
            
            # Short Currency
            cu = pd.read_pickle(cu_small_path)
            cu = cu.append(a, ignore_index=True, 
                                       verify_integrity=True)
            cu.to_pickle(cu_small_path)
            

            # Volume
            volume = pd.read_pickle(volume_path)
            volume = volume.append(a, ignore_index=True, 
                           verify_integrity=True)
            volume.to_pickle(volume_path)
            
            
            # Print update
            msg = '{}: {}'
            print(msg.format(timestamp, 
                             volume.last_valid_index()))
    
        # Wait on new Timestamp
        else:
            '''
            # Add real time to M1
            a, b = get_universe_singular(currencies, 'S5')
            a['timestamp'] = timestamp
            '''
            sleep(pause_time)
            
            
            

###############################################################################
# Update Volume and Diff
###############################################################################    
if 1:   
    
    diff = cu - cu.shift(1)
    diff = diff.dropna()
    values = diff.eur.values
    
    step = 20
    alpha = 1e-6
    for i in range(step, values.shape[0], step):
        vals = values[values.shape[0] - step:]
        k2, p = stats.normaltest(vals)
        if p < alpha:  # null hypothesis: x comes from a normal distribution
            print('{}: break'.format(i))
            #print("The null hypothesis can be rejected")
        else:
            #print("The null hypothesis cannot be rejected")
            if i % 3000 == 0:
                print(i)
    os.system('say "Completed"')
        
    
    
    
###############################################################################
# Plot Everything.  Ratios first
###############################################################################    
if 1:   
    
    cu.eur.plot(title='cu')
    plt.figure()
    volume.eur.plot(title='volume')
    plt.figure()
    diff.eur.plot(title='diff')
    
    
        

    ratios_drop = ratios.filter(regex='JPY').columns.tolist() \
                + ratios.filter(regex='HKD').columns.tolist()
    
    ratios = ratios.drop(ratios_drop, axis=1)
    print(ratios.columns)
    
    plot_index = np.arange(ratios.first_valid_index(), 
                           ratios.last_valid_index() + 1)   
    
    currency = 'eur'

    # Get Insturment List and which direction to align instrument
    pair_list = []
    shape_list = []
    for pair in ratios.columns:
        if currency.upper() in pair.split('_'):
            pair_list.append(pair)
            if currency.upper() == pair.split('_')[0]:
                shape_list.append(1)
            else:
                shape_list.append(-1)
                
    # Get Slope position for all values (instruments)
    currency_set = pd.DataFrame()
    for i in range(len(pair_list)):
        instrument = pair_list[i]
        shape = shape_list[i]
        if shape == 1:
            values = ratios.loc[plot_index, instrument]
        else:
            values = (ratios.loc[plot_index, instrument] * -1)\
                   + (2 * ratios.loc[plot_index,  instrument].values[-1])
        currency_set[pair_list[i]] = values
    
    
    plt.figure()
    currency_set.plot(title='set', subplots=True)
    
    '''
    order_list = []
    for pair in pair_list:
        find = pair.split('_')
        find.remove(currency.upper())
        order_list.append(currencies.index(find[0].lower()))
        
    # Assign Column to currency
    ax_col = currencies.index(currency)
    
    # Plot all currency Pairs and Channels
    for i in range(len(order_list)):
        p = pair_list[i]
        o = order_list[i]
        ax[o, ax_col].cla()
        currency_set[p].plot(ax=ax[o, ax_col])
        

        # get channels for currency
        coll = []
        for s in range(20, interval + normaltest_step, normaltest_step):
            channel = Channel(currency_set[p].values)
            k2, _p = normaltest(channel.flattened / channel.channel_deviation)
            if _p < normaltest_alpha:
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
            channel = Channel(currency_set.loc[currency_set.last_valid_index() - s: \
                                     currency_set.last_valid_index(), p].values)
            currency_set.loc[currency_set.last_valid_index() - (k + 1):, p]\
              .plot(ax=ax[o, ax_col], marker='|', markersize=.5)
            line = channel.line
            line_x = np.arange(currency_set.last_valid_index() - (k), 
                               currency_set.last_valid_index() + 1)
            line_y_up = line + (np.ones(line.shape[0]) * 2 \
                                * channel.channel_deviation)
            line_y_down = line + (np.ones(line.shape[0]) * -2 \
                                  * channel.channel_deviation)
            #axr[i-1].plot(line_x, line, color='grey', linewidth=.5)
            ax[o, ax_col].plot(line_x, line_y_up, 
                                 color='black', linewidth=.75)
            ax[o, ax_col].plot(line_x, line_y_down, 
                                 color='black', linewidth=.75)

    # Add Row Labels to first column
    for i in range(df.columns.shape[0] - 1):
        ax[i, 0].set_ylabel(currencies[i].upper(), rotation=90, size='large', color='white')
        
    # Add Columns Labels to first Row
    for i in range(df.columns.shape[0] - 1):
        ax[0, i].set_title(currencies[i].upper(), size='large', color='white')
    
    # Set plot x Time markers
    x_ticks_count = plt.xticks()[0].shape[0]
    x_tick_locations = np.linspace(plot_index[0], plot_index[-1], x_ticks_count).astype(int)
    x_ticks_markers = ratios.loc[x_tick_locations, 'timestamp']
    x_ticks_markers = x_ticks_markers.apply(lambda x: x - pd.Timedelta(hours=hours_subtraction_from_utc)).dt.time
    x_ticks_markers = x_ticks_markers.astype(str).apply(lambda x: x[:5]).values.astype(str)
    
    # Set x Grid
    for a in ax.ravel():
        a.set_xticks(x_tick_locations)
        a.set_xticklabels(x_ticks_markers, color='white', rotation=270)
        a.grid(which='major', axis='x', linestyle= '-',linewidth=.2,
                                   color='#343837', b=True) 
            
    ## Set Y Grid
    #for i in range(1, df.columns.shape[0]):    
    #    max_ticks = df.iloc[plot_index, i].max()
    #    min_ticks = df.iloc[plot_index, i].min()
    #    # Different ticks for jpy and hkd
    #    if df.columns[i] == 'jpy':
    #        ticks = np.arange(min_ticks, max_ticks, .000001).round(6)
    #    elif df.columns[i] == 'hkd':
    #        ticks = np.arange(min_ticks, max_ticks, .00001).round(5)
    #    else:
    #        ticks = np.arange(min_ticks, max_ticks, .0001).round(4)
    #        # Create ticks 
    #    axr[i - 1].grid(linestyle= '-', axis='y', linewidth=.2, 
    #                               color='#343837', b=True)    
    #    axr[i - 1].set_yticks(ticks)
    
    # Remove Y Ticks
    for a in ax.ravel():
        a.set_yticklabels([])
    
    # Print Recent Location
        print(df.last_valid_index())
    
    
    

    '''

    
    

    
    
