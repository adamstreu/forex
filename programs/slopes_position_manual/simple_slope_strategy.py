###############################################################################
# Notes
###############################################################################
if True:

    
    """
    TONIGHT:
        Prep for slopes and channels tomorrow:
            
    
            
    """
    
    
    
    
    
    
    '''
    NOW
        
        by using mulit winodos on mulitple instrumetn  ( currencies)
        we were able to predict well the slopes of a currency in shifted 30
        
        the smae for positions
        
        So i used both slopes and windows from mulitplt instruments and 
        currency levels to predict a currency shifted 30.
        
        The results were very good.
        
        Explore this.  Find issues, add more data, perios, etc.
        
    '''
    
    
    '''
    Next:  
        add slopes to dashboard and watch.
        Make placements live.
        
    Today:
        Keep working on the simple slopes strategy
        
        try:
            expand index a few paces
            short and long
            test on outcomes
            look at live
            expand to all currencies
    
    '''
    
    '''
    
    MUCH TO DO:
        FIND THE BEST CHANNEL.
        
                will this function to search through all realted instruments
                to find likely currency turn arounds (or breakouts - keep an open eye.)
                
                
                # Get Channel (multiple timelines ? )
                # Score Fit (  do not have a method for doing so yet - )
                # Get position - using 2 * channel deviation for turn around points
                
                
                Well - is this true?
                    any currency slope can be found from just a few instruments?
                    I wonder - can we predict the currency itself?
        
        Simple slope prediction:
            
            
            
            
            
            
        All is strung together:
            Can I really get an currencies slope predicted perfectly from 
                just a handful or EUR____ instruments?
            is there an ml iteration sequence to converge on these things?
        
        
    
    
    '''


###############################################################################
# Imports 
###############################################################################
if 0:

    # Imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from time import sleep
    from scipy.optimize import leastsq
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    import os; os.chdir('/northbend')
    from classes.channel import Channel
    from classes.wave import Wave
    from libraries.currency_universe import get_universe_singular
    from libraries.currency_universe import backfill_with_singular
    from libraries.indicators import get_rolling_currency_correlation
    from libraries.indicators import get_rolling_mean_pos_std
    from libraries.indicators import get_channel_mean_pos_std
    from libraries.indicators import get_rolling_waves
    from libraries.oanda import get_time
    from libraries.oanda import market
    from libraries.correlation import get_autocorrelation
    from libraries.stats import get_distribution_boundary
    from libraries.taps import get_taps
    
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
    _from = '2018-10-01T00:00:00Z'
    _to   = '2019-12-01T00:00:00Z'
    granularity = 'H1'
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
# Get indicators
###############################################################################    
if 0:

    # Parameters
    currencies = ['aud', 'cad', 'eur', 'gbp', 'nzd', 'usd']
    windows = np.array([180])

    # Get slopes indicator
    positions = pd.DataFrame()
    deviations = pd.DataFrame()
    slopes = pd.DataFrame()
    for currency in currencies:
        pos = get_channel_mean_pos_std(cu[currency].values.astype(float), windows)
        positions[currency] = pos['pos'].values.ravel()
        deviations[currency] = pos['std'].values.ravel()
        slopes[currency] = pos['slope'].values.ravel()
        
    corr = get_rolling_currency_correlation(cu.eur.values, cu.usd.values, windows)

     
    # Get Multislope windows for EUR and USD
    windows = np.array([30, 60, 120])
    pos = get_channel_mean_pos_std(cu['eur'].values, windows)
    eur_slopes = pos['slope']
    pos = get_channel_mean_pos_std(cu['usd'].values, windows)
    usd_slopes = pos['slope']

    
    
    
###############################################################################
# Simplest slope predictions - just watch and do live 
# Aim for larger windows (120 seems good so far) over a few days (5 ish)
# Looking at 50 - 200 pips
# With larger risk rewards than the short stuff goes.
###############################################################################    
if 0:    
    
    
    # Plot single slope windows for two currencies
    plt.figure()
    plt.plot(StandardScaler().fit_transform(slopes.fillna(0).eur.values.reshape(-1, 1)).ravel(), label='eur')    
    plt.plot(StandardScaler().fit_transform(slopes.fillna(0).usd.values.reshape(-1, 1)).ravel(), label='usd')    
    plt.plot(StandardScaler().fit_transform(ratios.EUR_USD.values.reshape(-1, 1)).ravel())    
    plt.plot(np.zeros(slopes.shape[0]))
    plt.legend()
    plt.show()
  
    
    # PLot long placements
    slopes.loc[slopes.eur > 0, 'eur'].abs().plot()
    slopes.loc[slopes.eur < 0, 'usd'].abs().plot()
    
    
    # Plot Multi window slopes for both currencies
    plt.figure()
    plt.plot(StandardScaler().fit_transform(eur_slopes.fillna(0)), label='eur', color='blue')    
    plt.plot(StandardScaler().fit_transform(usd_slopes.fillna(0)), label='usd', color='orange')
    plt.plot(StandardScaler().fit_transform(ratios.EUR_USD.values.reshape(-1, 1)).ravel(), color='black')    
    plt.plot(np.zeros(slopes.shape[0]), color='grey')
    plt.legend()
    plt.show()
    # AGAIN _ Plot Multi window slopes for both currencies
    eur_slopes.plot()  
    plt.plot(np.zeros(slopes.shape[0]), color='grey')
    usd_slopes.plot()
    plt.plot(np.zeros(slopes.shape[0]), color='grey')
    plt.figure()
    plt.plot(StandardScaler().fit_transform(ratios.EUR_USD.values.reshape(-1, 1)).ravel())    
    plt.legend()
    plt.show()    
    
    
    
    
    
    plt.figure()
    ratios.EUR_USD.plot()    
    
    
    
    
    
    
    
    
###############################################################################
# Simple slope Prediction
###############################################################################    
if 0:
       
    # Parameters
    rolling_window = 10
    index_extension = 2
    currency = 'eur'
    
    # Get turn around points.
    eur_high = (slopes.rolling(rolling_window).mean().eur.shift(1) < \
                slopes.rolling(rolling_window).mean().eur) \
             & (slopes.rolling(rolling_window).mean().eur.shift(-1) < \
                slopes.rolling(rolling_window).mean().eur)
    usd_low = (slopes.rolling(rolling_window).mean().usd.shift(1) > \
               slopes.rolling(rolling_window).mean().usd) \
             & (slopes.rolling(rolling_window).mean().usd.shift(-1) > \
                slopes.rolling(rolling_window).mean().usd)    
    usd_high = (slopes.rolling(rolling_window).mean().usd.shift(1) < \
                slopes.rolling(rolling_window).mean().usd) \
             & (slopes.rolling(rolling_window).mean().usd.shift(-1) < \
                slopes.rolling(rolling_window).mean().usd)
    eur_low = (slopes.rolling(rolling_window).mean().eur.shift(1) > \
               slopes.rolling(rolling_window).mean().eur) \
             & (slopes.rolling(rolling_window).mean().eur.shift(-1) > \
                slopes.rolling(rolling_window).mean().eur) 
    # Get Index values.
    eur_high = eur_high[eur_high].index.values
    eur_low  = eur_low[eur_low].index.values
    usd_high = usd_high[usd_high].index.values
    usd_low  = usd_low[usd_low].index.values
    
    # Add to index (make width larger (& forward only of course))
    for i in range(1, index_extension):
        eur_high = np.union1d(eur_high, eur_high + i)
        eur_low  = np.union1d(eur_low, eur_low + i)
        usd_high = np.union1d(usd_high, usd_high + i)
        usd_low  = np.union1d(usd_low, usd_low + i)
     
    # Find intersecting indexes        
    short = np.intersect1d(eur_high, usd_low)
    long = np.intersect1d(eur_low, usd_high)
    
        
    # Make outcomes.  Over under or in bandwidth on values _shift_ away
    cross_zero = pd.DataFrame(slopes[currency].copy())   
    cross_zero[currency] = 0
    over_up = (slopes[currency].shift(-1) > 0) & (slopes[currency] <= 0) 
    over_down = (slopes[currency].shift(-1) < 0) & (slopes[currency] >= 0) 
    over_up_index = over_up[over_up].index.values
    over_down_index = over_down[over_down].index.values
    cross_zero.loc[over_up_index] = 1
    cross_zero.loc[over_down_index] = -1

    
    '''
    # Plot Results and points
    plt.figure()
    slopes.rolling(rolling_window).mean().loc[:, ['eur', 'usd']].plot()
    plt.plot(np.zeros(slopes.shape[0]), color='grey')
    slopes.eur.loc[eur_high].plot(style='o')
    slopes.usd.loc[usd_low].plot(style='o')        
    slopes.eur.loc[eur_low].plot(style='o')
    slopes.usd.loc[usd_high].plot(style='o')            
    
    plt.figure()
    ratios.EUR_USD.plot()
    ratios.loc[long, 'EUR_USD'].plot(style='o', label='long')
    ratios.loc[short, 'EUR_USD'].plot(style='o', label='short')
    plt.title('eur_usd')
    plt.legend()
    plt.show()
    
    plt.figure()
    corr.plot()
    
    
    long_again = corr.loc[long][corr.loc[long] < -.5].dropna().index.values
    plt.figure()
    ratios.EUR_USD.plot()
    ratios.loc[long_again, 'EUR_USD'].plot(style='o', label='long')
    plt.title('eur_usd')
    plt.legend()
    plt.show()
    '''
    
    plt.figure()
    plt.plot(StandardScaler().fit_transform(cu[currency].values.reshape(-1, 1)).ravel())
    plt.plot(StandardScaler().fit_transform(slopes.fillna(0)[currency].values.reshape(-1, 1)).ravel())    
    plt.plot(np.zeros(slopes.shape[0]))
    plt.plot(over_up_index, StandardScaler().fit_transform(cu[currency].values.reshape(-1, 1)).ravel()[over_up_index], 'o', label = 'up')
    plt.plot(over_down_index, StandardScaler().fit_transform(cu[currency].values.reshape(-1, 1)).ravel()[over_down_index], 'o', label = 'down')
    plt.legend()
    plt.show()
  
    plt.figure(); ratios.EUR_USD.plot()
    
###############################################################################
# Plot mulitple slope windows - get high points
###############################################################################    
if 0:

    # Parameters
    currencies = ['aud', 'cad', 'eur', 'gbp', 'nzd', 'usd']
    windows = np.array([ 60, 120, 240])
    rolling_window = 10
  
    eur_channel_stats = get_channel_mean_pos_std(cu.eur.values, windows)
    eur_channel_std   = eur_channel_stats['std']
    eur_channel_pos   = eur_channel_stats['pos']
    eur_channel_mean  = eur_channel_stats['mean']
    eur_slopes        = eur_channel_stats['slope']
        
    
        
    # Get turn around points.
#    high1 = (eur_slopes[15].rolling(rolling_window).mean().shift(1) < \
#             eur_slopes[15].rolling(rolling_window).mean()) \
#          & (eur_slopes[15].rolling(rolling_window).mean().shift(-1) < \
#             eur_slopes[15].rolling(rolling_window).mean())
#    high2 = (eur_slopes[30].rolling(rolling_window).mean().shift(1) < \
#             eur_slopes[30].rolling(rolling_window).mean()) \
#          & (eur_slopes[30].rolling(rolling_window).mean().shift(-1) < \
#             eur_slopes[30].rolling(rolling_window).mean())   
    high3 = (eur_slopes[60].rolling(rolling_window).mean().shift(1) < \
             eur_slopes[60].rolling(rolling_window).mean()) \
          & (eur_slopes[60].rolling(rolling_window).mean().shift(-1) < \
             eur_slopes[60].rolling(rolling_window).mean())
    high4 = (eur_slopes[120].rolling(rolling_window).mean().shift(1) < \
             eur_slopes[120].rolling(rolling_window).mean()) \
          & (eur_slopes[120].rolling(rolling_window).mean().shift(-1) < \
             eur_slopes[120].rolling(rolling_window).mean())
    high2 = (eur_slopes[240].rolling(rolling_window).mean().shift(1) < \
             eur_slopes[240].rolling(rolling_window).mean()) \
          & (eur_slopes[240].rolling(rolling_window).mean().shift(-1) < \
             eur_slopes[240].rolling(rolling_window).mean())   
          
          
    eur_slopes.plot()
#    eur_slopes[15][high1].plot(style='o')
    eur_slopes[240][high2].plot(style='o')    
    eur_slopes[60][high3].plot(style='o')    
    eur_slopes[120][high4].plot(style='o')    
    
    plt.figure()
    cu.eur.plot()
    
    
        
###############################################################################
# Ml on many many slopes (shifted)
###############################################################################    
if 0:

    windows = np.array([5, 10, 15, 30, 45, 60, 90, 120])
    currency_window = np.array([60])
    
    slopes = pd.DataFrame(ratios.iloc[:, 0].copy())
    positions = pd.DataFrame(ratios.iloc[:, 0].copy())
    
    instrument_positions = pd.DataFrame()
    instrument_deviations = pd.DataFrame()
    instrument_slopes = pd.DataFrame()    
    currency_positions = pd.DataFrame()
    currency_deviations = pd.DataFrame()
    currency_slopes = pd.DataFrame()   

    instruments = ['EUR_USD', 'USD_CAD', 'NZD_USD', 
                   'GBP_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD']
    instruments = ['EUR_USD', 'EUR_CHF', 'EUR_GBP', 
                   'EUR_JPY', 'EUR_AUD', 'EUR_USD', 'USD_CAD', 'NZD_USD', 
                   'GBP_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD']
    instruments = ['AUD_CHF',
                    'AUD_JPY',
                    'AUD_NZD',
                    'AUD_USD',
                    'CAD_CHF',
                    'CAD_JPY',
                    'CHF_JPY',
                    'EUR_AUD',
                    'EUR_CAD',
                    'EUR_CHF',
                    'EUR_GBP',
                    'EUR_JPY',
                    'EUR_NZD',
                    'EUR_USD',
                    'GBP_AUD',
                    'GBP_CAD',
                    'GBP_CHF',
                    'GBP_JPY',
                    'GBP_NZD',
                    'GBP_USD',
                    'NZD_CAD',
                    'NZD_CHF',
                    'NZD_JPY',
                    'NZD_USD',
                    'USD_CAD',
                    'USD_CHF',
                    'USD_JPY']
    currency = 'eur'
    
    for instrument in instruments:
        print(instrument)
        pos = get_channel_mean_pos_std(ratios[instrument].values.astype(float), windows)
        instrument_positions[instrument] = pos['pos'].values.ravel()
        instrument_deviations[instrument] = pos['std'].values.ravel()
        instrument_slopes[instrument] = pos['slope'].values.ravel()
    
        slopes = slopes.join(pos['slope'], lsuffix='.')
        positions = positions.join(pos['pos'], lsuffix='.')
        
    slopes.drop('timestamp', inplace=True, axis=1)
    positions.drop('timestamp', inplace=True, axis=1)
    
    pos = get_channel_mean_pos_std(cu[currency].values.astype(float), currency_window)
    currency_positions[currency] = pos['pos'].values.ravel()
    currency_deviations[currency] = pos['std'].values.ravel()
    currency_slopes[currency] = pos['slope'].values.ravel()
    
    


    
    # Run (shifted) machine learning on just slopes or positions (good results)
    if False:
        
        positions_or_slopes = 'positions'
        outcome_shift = 30
        
        if positions_or_slopes == 'slopes':
            ml_index = slopes.dropna().index.values
            ml = slopes.loc[ml_index]
            ml_outs = currency_slopes.loc[ml_index, currency]
            '''   Look ahead - try to predict slope in 15 moves '''
            ml_outs = currency_slopes.shift(-outcome_shift).loc[ml_index, currency]
        else:
            ml_index = positions.dropna().index.values
            ml = positions.loc[ml_index]
            ml_outs = currency_positions.loc[ml_index, currency]
            '''   Look ahead - try to predict slope in 15 moves '''
            ml_outs = currency_positions.shift(-outcome_shift).loc[ml_index, currency]            

        ml_index = ml_outs.dropna().index.values
        ml = ml.loc[ml_index]
        ml_outs = ml_outs.loc[ml_index]
        
        # Split  
        train_index, test_index = train_test_split(ml.index.values, 
                                                   train_size = .8, 
                                                   shuffle = False)
        x_train    = ml.loc[train_index]
        y_train    = ml_outs.loc[train_index]
        x_test     = ml.loc[test_index]
        y_test     = ml_outs.loc[test_index]
        
        
        # Scale 
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    
        
        # Implemnent Model 
        linreg = LinearRegression()
        linreg.fit(x_train, y_train)
        predictions = linreg.predict(x_test)
        
        plt.figure()
        plt.plot(predictions, label = 'predictions')
        plt.plot(y_test.values, label = 'currency')
        plt.plot(np.zeros(predictions.shape[0]))
        plt.legend()
        plt.show()
        
        score = linreg.score(x_test, y_test)
        print('SCORE: {:.2f}'.format(score))
        
    
        plt.figure(); 
        plt.plot(cu.loc[test_index, currency].values)
        plt.plot(cu.shift(-outcome_shift).loc[test_index, currency].values)
    
    
    
    

###############################################################################
# Get all slopes.  standardize.  Plot.  over 1 window
###############################################################################    
if 1:

    # Parameters
    currencies = ['aud', 'cad', 'eur', 'gbp', 'nzd', 'usd']
    windows = np.array([360])

    # Get slopes indicator
    positions = pd.DataFrame()
    deviations = pd.DataFrame()
    slopes = pd.DataFrame()
    for currency in currencies:
        pos = get_channel_mean_pos_std(cu[currency].values.astype(float), windows)
        positions[currency] = pos['pos'].values.ravel()
        deviations[currency] = pos['std'].values.ravel()
        slopes[currency] = pos['slope'].values.ravel()
        
    # Standardize  
    slopes = StandardScaler().fit_transform(slopes.fillna(0))
    slopes = pd.DataFrame(slopes, columns = currencies)
    
    # Plot
    slopes.plot()
    plt.plot(np.zeros(slopes.shape[0]), color='grey')
    
#    # Plot an instrument
#    plt.figure()
#    ratios.EUR_USD.plot()
    
    
    
