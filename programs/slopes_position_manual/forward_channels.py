###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   
    
    # Imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from time import sleep
    from scipy.stats import normaltest
    from scipy.io.wavfile import write
    import os; os.chdir('/northbend')
    from libraries.currency_universe import backfill_with_singular
    from libraries.currency_universe import get_universe_singular
    from libraries.oanda import market
    from libraries.oanda import get_time
    from libraries.oanda import get_multiple_candles_spread
    from classes.channel import Channel 

    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15) 
    np.warnings.filterwarnings('ignore')
    plt.rcParams['figure.figsize'] = [14, 4]
    
    
    
    
        
    df = pd.read_pickle('/northbend/tmp/currencies_M1.pkl')
    df_price = df.copy()
#    df = df.loc[30000:]
    df = df - df.shift(1)
    df = df.dropna()
    
    currency = 'cad'
    
    # Normal Test
    normaltest_alpha = 1e-10
    normaltest_step = 1
    
###############################################################################
# Front fill normal test
###############################################################################    
if 0:   
    
    
    
    
    breaks = []
    start = df.first_valid_index()
    for step in range(df.first_valid_index() + 10, 
                      df.last_valid_index() - 10, 
                      normaltest_step):
        try:
            channel = Channel(df.loc[start: step, currency].values)
            k2, p = normaltest(channel.flattened / channel.channel_deviation)
            #print("p = {:g}".format(p))
            if p < normaltest_alpha:
                # "The null hypothesis can be rejected")
                start = step
                breaks.append(step)
            else:
                pass
                # coll.append(step) # "The null hypothesis cannot be rejected")
        except Exception as e:
            print(start, step, e)
    breaks = np.array(breaks)
    
    
    
    df[currency].plot()
    for _break in breaks:
        plt.plot(_break, df.loc[_break, currency], 'o')
    plt.figure()
    df_price.loc[df.index.values, currency].plot()
    for _break in breaks:
        plt.plot(_break, df_price.loc[_break, currency], 'o')    
    
    
    
    
    for _break in breaks:    
        pass

    '''
    # Get Positions for channel breaks
    keep = []
    for k in range(coll.shape[0] - 1):
        if coll[k + 1] > coll[k] + normaltest_step:
            keep.append(coll[k])
    if coll.shape[0] > 0: keep.append(coll[-1])
    keep = np.array(keep)
    '''
    
    
    
    '''
    # Plot chnnels by normal break
    for k in keep[::-1]:
        channel = Channel(df.loc[df.last_valid_index() - k:, 
                                 currency].values)
        df.loc[df.last_valid_index() - (k + 1):, currency]\
          .plot(ax=ax.ravel()[i-1], marker='|', markersize=.5)
        line = channel.line
        line_x = np.arange(df.last_valid_index() - (k), 
                           df.last_valid_index() + 1)
        line_y_up = line + (np.ones(line.shape[0]) * 2 \
                            * channel.channel_deviation)
        line_y_down = line + (np.ones(line.shape[0]) * -2 \
                              * channel.channel_deviation)
        ax.ravel()[i-1].plot(line_x, line, color='grey', linewidth=.5)
        ax.ravel()[i-1].plot(line_x, line_y_up, 
                             color='black', linewidth=.5)
        ax.ravel()[i-1].plot(line_x, line_y_down, 
                             color='black', linewidth=.5)
    '''
    
  
###############################################################################
# Front fill normal test
###############################################################################    
if 1:   
    
    currency = 'aud'
    
    m30 = pd.read_pickle('/northbend/tmp/currencies_M30.pkl')
    m15 = pd.read_pickle('/northbend/tmp/currencies_M15.pkl')
    m5 = pd.read_pickle('/northbend/tmp/currencies_M5.pkl')
    m1 = pd.read_pickle('/northbend/tmp/currencies_M1.pkl')
    
    m30.set_index('timestamp', inplace=True, drop=True)
    m15.set_index('timestamp', inplace=True, drop=True)
    m5.set_index('timestamp', inplace=True, drop=True)
    m1.set_index('timestamp', inplace=True, drop=True)
    
    m30 = m30.resample('T').ffill()
    m15 = m15.resample('T').ffill()
    m5 = m5.resample('T').ffill()
    m1 = m1.resample('T').ffill()

    comb = pd.DataFrame(m1[currency]).join(pd.DataFrame(m5[currency]), lsuffix='.')
    comb = comb.join(pd.DataFrame(m15[currency]), lsuffix='.')
    comb = comb.join(pd.DataFrame(m30[currency]), lsuffix='.')
    comb.dropna(inplace=True)
    
    
    
