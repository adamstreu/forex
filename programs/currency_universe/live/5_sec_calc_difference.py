###############################################################################
# Notes
###############################################################################
if True:

    
    '''
    Explore: Difference between given and calculated
    
        5 sec differences.
        the movements will not be great.
        But can we get some predictions?
    
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
    _from = '2018-12-01T00:00:00Z'
    _to   = '2019-01-01T00:00:00Z'
    granularity = 'S5'
    # Currencies to use
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 
                  'hkd', 'jpy', 'nzd', 'sgd', 'usd']   
    currencies = ['eur','usd']  
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

    # Export



###############################################################################
# Update All Info in real time
###############################################################################
if 1:
    


    calc = cu.eur / cu.usd
    given = ratios.EUR_USD.astype(float)
    
    diff = given - calc
    
    given = given.reset_index(drop=True)
    calc = calc.reset_index(drop=True)
    diff = diff.reset_index(drop=True)


    bound = .05
    upper_bound = 1 - bound
    lower_boundm = bound
    index = given.loc[diff > diff.quantile(upper_bound)].index.values

    for i in range(1, 120, 10):
        step = i
        a = given.loc[index]
        b = given.loc[index + step]
        difference = (a.values - b.values)[~(np.isnan(a.values - b.values))]
        sns.distplot(difference)
        print(pd.Series(difference).kurtosis())
    
    
    given.loc[index].plot(style='o')
    calc.loc[index].plot(style='o')
    given.plot()
    
    
    plt.figure()
    for i in range(10):
        plt.plot(given.loc[index + i])








































