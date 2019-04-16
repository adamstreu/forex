###############################################################################
# Notes
###############################################################################
if True:

    '''
    Machine Learning.  Predict currency (in shifted amount) from
        currencies, 
        slopes from mulitple instrument, windows
        positions from mulitple instrument, windows  
        
        
    Next:

        
        try with currencie inds instead of instrumetns
        add rolling pos ?
        Can predictions on mulitple windows be 'averaged'
        or use diff windows.
        Have what 
        Watch live.
       
        
        
    Analysis notes:
        at 10 windows (hours) ahead it does great.
            Drop to poor by 60 (hours)  
            At least using 'these' windows
        Can't use currency.  convergess to strongly on column.
            
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Lars
    from sklearn.linear_model import LassoLarsCV
    from sklearn.linear_model import LarsCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import r2_score
    from sklearn.metrics import classification_report as cr
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
    _from = '2018-01-01T00:00:00Z'
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

    # Indicator Parameters
    windows = np.arange(5, 300, 10)
    windows = np.array([30, 60, 120, 240])
    
    # Currencies
    currency = 'usd'
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 
                  'hkd', 'jpy', 'nzd', 'sgd', 'usd'] 
    
    # Choose Instruments    
    instruments = ['EUR_USD', 'USD_CAD', 'NZD_USD', 
                   'CAD_JPY','CHF_JPY', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF',
                   'EUR_GBP', 'EUR_JPY', 'EUR_NZD', 'EUR_USD', 'GBP_AUD',
                   'GBP_CAD', 'GBP_CHF', 'GBP_JPY', 'GBP_NZD', 'GBP_USD',
                   'NZD_CAD', 'NZD_CHF', 'NZD_JPY', 'NZD_USD', 'USD_CAD']
    instruments = ['EUR_USD', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY', 
                   'EUR_AUD', 'EUR_USD', 'EUR_CAD']    
    instruments = ['EUR_USD', 'USD_CAD', 'NZD_USD', 'GBP_USD', 'USD_JPY', 
                   'AUD_USD', 'USD_CHF' ]
    # Get Channel position and slope indicators
    slopes = pd.DataFrame(ratios.iloc[:, 0].copy())
    positions = pd.DataFrame(ratios.iloc[:, 0].copy())
    mean = pd.DataFrame(ratios.iloc[:, 0].copy())
    deviation = pd.DataFrame(ratios.iloc[:, 0].copy())
    for instrument in instruments:
        print(instrument)
        pos = get_channel_mean_pos_std(ratios[instrument].values, windows)
        slopes = slopes.join(pos['slope'], lsuffix='.')
        positions = positions.join(pos['pos'], lsuffix='.')
        mean = mean.join(pos['mean'], lsuffix='.')
        deviation = deviation.join(pos['std'], lsuffix='.')
    slopes.drop('timestamp', inplace=True, axis=1)
    positions.drop('timestamp', inplace=True, axis=1)
    mean.drop('timestamp', inplace=True, axis=1)    
    deviation.drop('timestamp', inplace=True, axis=1)    

    
    # Get Rolling position and slope indicators
    rolling_deviations = pd.DataFrame(ratios.iloc[:, 0].copy())
    rolling_positions = pd.DataFrame(ratios.iloc[:, 0].copy())
    for instrument in instruments:
        print(instrument)
        pos = get_rolling_mean_pos_std(ratios[instrument], windows)
        rolling_deviations = rolling_deviations.join(pos['std'], lsuffix='.')
        rolling_positions = rolling_positions.join(pos['pos'], lsuffix='.')
    rolling_deviations.drop('timestamp', inplace=True, axis=1)
    rolling_positions.drop('timestamp', inplace=True, axis=1)
    
    
    # Try channel stuff with currencies
    currency_slopes = pd.DataFrame(ratios.iloc[:, 0].copy())
    currency_positions = pd.DataFrame(ratios.iloc[:, 0].copy())
    currency_mean = pd.DataFrame(ratios.iloc[:, 0].copy())
    for currency in currencies:
        print(currency)
        pos = get_channel_mean_pos_std(cu[currency].values, windows)
        currency_slopes = slopes.join(pos['slope'], lsuffix='.')
        currency_positions = positions.join(pos['pos'], lsuffix='.')
        currency_mean = positions.join(pos['mean'], lsuffix='.')
    


        
###############################################################################
# ML.  Find Currency in shifted amount
###############################################################################    
if 1:

    # Parameters
    outcome_shift = - 120

    # Create ml df
    ml = positions.join(slopes, lsuffix= 'o')
    ml = ml.join(mean, lsuffix='er')
    ml = ml.join(deviation, lsuffix='er')
    #ml = ml.join(cu.loc[:, 'eur'], lsuffix='l')
    ml = ml.join(cu.iloc[:, 1:], lsuffix='l')
    ml = ml.join(rolling_positions, lsuffix='a')
    ml = ml.join(rolling_deviations, lsuffix='b')
    ml = ml.join(currency_slopes, lsuffix='c')
    ml = ml.join(currency_positions, lsuffix='d')
    ml = ml.join(currency_mean, lsuffix='e')
    ml = ml.join(ratios.iloc[:, 1:], lsuffix='qre')
    
    
    # Assign y_values
    ml_outs = cu.shift(outcome_shift)[currency]

    # drop na's from ml on both dfs
    ml_index = ml.dropna().index.values
    ml = ml.loc[ml_index]
    ml_outs = ml_outs.loc[ml_index]
    
    # Drop na's from (shifted) ml_outs on both dfs
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
    linreg = Lars()                      # Better
    linreg = LarsCV()             
         # one Better
    linreg = LassoLarsCV()                   # Same
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    predictions = linreg.predict(x_test)
    
    # Plot predictions and y_test
    plt.figure()
    plt.plot(predictions, label = 'Predictions')
    plt.plot(pd.Series(predictions).rolling(5).mean(), label='rolling predictions')
    plt.plot(y_test.values, label = 'Shifted Currencies ( y_test values', color='grey')
    plt.plot(cu.loc[test_index, currency].values, label = 'UNSHIFTED')
    plt.legend()
    plt.show()
    
    # Print Score and summary
    score = linreg.score(x_test, y_test)
    print('SCORE: {:.2f}'.format(score))
    print('Explained Variance: {}'\
          .format(explained_variance_score(y_test, predictions)))



###############################################################################
# Use two currency predictions to predict instrument
###############################################################################    
if 0:

    # Get Parameters
    if False:        
        cur1 = predictions
        cur2 = predictions
        
    # Choose Instrument
    instrument = 'EUR_USD'
    y_values = ratios.loc[test_index, instrument]
    
    # Score outcome
    score = r2_score(y_values, cur1 / cur2)
    
    # Plot prediction and instrument
    plt.figure()
    plt.plot(y_values.values, label = 'instrument')
    plt.plot(cur1 / cur2, label = 'predictions')
    plt.plot(ratios.loc[test_index, instrument].values)
    plt.title('Instrument Predictions')
    
    # Print Summary and Stats
    print('Instrument score: {}'.format(score))
    


    
    
    
    
    
    
    
    
    


        
###############################################################################
# Logistic regression - guess in band, over or under
###############################################################################    
if 0:

    # Parameters
    outcome_shift = - 60
    band_width = .0002

    # Make outcomes.  Over under or in bandwidth on values _shift_ away
    outs = np.zeros(cu.shape[0])
    over_index = (cu.shift(outcome_shift)[currency] - cu[currency]) > band_width
    under_index = (cu.shift(outcome_shift)[currency] - cu[currency]) < -band_width
    outs[over_index.values] = 1
    outs[under_index.values] = -1
    outs = pd.DataFrame(outs.astype(int), index = cu.index.values)
    ml_outs = outs.copy()

    # Create ml df
    ml = positions.join(slopes, lsuffix= 'o')
    ml = ml.join(mean, lsuffix='t')
    ml = ml.join(deviation, lsuffix='esr')
    ml = ml.join(cu.loc[:, 'eur'], lsuffix='l')
    ml = ml.join(cu.iloc[:, 1:], lsuffix='ld')
    ml = ml.join(rolling_positions, lsuffix='a')
    ml = ml.join(rolling_deviations, lsuffix='b')
    ml = ml.join(currency_slopes, lsuffix='c')
    ml = ml.join(currency_positions, lsuffix='d')
    ml = ml.join(currency_mean, lsuffix='e')

    # drop na's from ml on both dfs
    ml_index = ml.dropna().index.values
    ml = ml.loc[ml_index]
    ml_outs = ml_outs.loc[ml_index]
    
    # Drop na's from (shifted) ml_outs on both dfs
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
    linreg = LogisticRegression(solver = 'saga', multi_class ='multinomial')
    linreg.fit(x_train, y_train)
    predictions = linreg.predict(x_test)

    # Print Score and summary
    print(cr(y_test, predictions))
    
    
    
    
    
    
    
    
    
    
    
