import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from scipy.stats import normaltest
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pandas.plotting import autocorrelation_plot as ac
import os; os.chdir('/northbend')
from libraries.outcomes import get_outcomes
from libraries.oanda import get_candles
from libraries.taps import get_taps
from libraries.currency_universe import get_currencies
from classes.channel import Channel
from classes.wave import Wave

# Set Environment
pd.set_option('display.width', 1000)
pd.set_option('max.columns', 15)
np.warnings.filterwarnings('ignore')


###############################################################################
# Get instruments.  Call currency Matrix
###############################################################################
if False:

    granularity = 'S5'
    _from = '2018-01-01T00:00:00Z'
    _to   = '2018-02-01T00:00:00Z'
    currency_dictionary = get_currencies(granularity, _from, _to)
    cur = currency_dictionary['currencies']
    curdiff = currency_dictionary['currencies_delta']
    ratios = currency_dictionary['ratios']
    






###############################################################################
# Correlation Tests
###############################################################################

# =============================================================================
# Determine correlation between all currencies over entire DF timeframe
# =============================================================================
if False:
    corrs = []
    for cur1 in cur.columns:
        tmp = []
        for cur2 in cur.columns:
            corr = (np.corrcoef(cur[cur1], cur[cur2]))[0, 1]
            tmp.append(corr)
        corrs.append(tmp)
    long_corrs = pd.DataFrame(corrs, index = cur.columns, columns = cur.columns)
    # Plot correltions with clustering
    sns.clustermap(long_corrs,linecolor='black', linewidth=3)


# =============================================================================
# Take n samples on correlation between two variables of set window length
# Calculate mean.  Do for all currencies on multiple window lengths
# =============================================================================
if False:
    samples = 100
    final_collection = []
    intervals = np.array([15, 30, 40, 50])#, 360, 500, 750, 1000, 1500, 2000, 3000])
    for interval in intervals:
        print(str(interval) + ' ----------------- ' )
        interval_collection = []
        for cur1 in cur.columns:
            cur_cur_collection = []
            for cur2 in cur.columns:
                location_collection = []
                print(cur1, cur2)
                for location in np.linspace(interval, 
                                            cur.shape[0] - 1, 
                                            samples):
                    cor = np.corrcoef(cur.loc[location - interval: location, cur1],
                                      cur.loc[location - interval: location, cur2])
                    cor = cor[0, 1]
                    location_collection.append(cor)
                cur_cur_collection.append(np.array(location_collection).mean())
            print(len(location_collection))
            interval_collection.append(cur_cur_collection)
        print('Interval')
        final_collection.append(interval_collection)
    # Make into Df
    final_collection = np.array(final_collection)
    final_collection = final_collection.reshape(1, -1, cur.columns.shape[0])
    interval = np.repeat(intervals, cur.columns.shape[0])
    columns = np.tile(cur.columns, intervals.shape[0])
    final_index = [columns, interval]
    final_columns = cur.columns
    interval_corrs = pd.DataFrame(final_collection[0],
                                    index = final_index,
                                    columns = final_columns)
    # Plot
    sns.heatmap(interval_corrs,linecolor='black', linewidth=3)
    if True:
        # Make into another type of Df
        int_corrs = pd.DataFrame()
        for cur1 in interval_corrs.columns:
            for cur2 in interval_corrs.columns:
                int_corrs[str(cur1) + '_' + str(cur2)] = interval_corrs.loc[cur1, cur2]
        # Plot
        sns.heatmap(int_corrs,linecolor='black', linewidth=3)


# =============================================================================
# Rolling Correlation on two currencies over all df for one corr interval
# =============================================================================
if False:    
    cur1 = 'cad'
    cur2 = 'eur'
    interval = 60
    rolling_window = 120
    stop = 960
    location_collection = []
    # Get ratios column name
    if str(cur1) + '_' + str(cur2) in ratios.columns:
        ratios_col = str(cur1) + '_' + str(cur2)
    else:
        ratios_col = str(cur2) + '_' + str(cur1)
    for location in range(interval, stop): #cur.shape[0] - 1):
        cor = np.corrcoef(cur.loc[location - interval: location, cur1],
                          cur.loc[location - interval: location, cur2])
        cor = cor[0, 1]
        location_collection.append(cor)
    fig, ax = plt.subplots(4, 1, figsize=(9,6))
    ax[0].plot(cur.loc[interval: stop, cur1].values, label=cur1)
    ax[1].plot(cur.loc[interval: stop, cur2].values, label=cur2)
    ax[2].plot(location_collection, '+')
    ax[2].plot(location_collection, label='correlation')
    ax[2].plot(pd.Series(location_collection).rolling(rolling_window).mean().values)
    ax[3].plot(ratios.loc[interval: stop, ratios_col].values, label=ratios_col)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Print long term correlation on the two
    print('Mean window correlation: {}'.format(np.array(location_collection).mean()))
            
    
# =============================================================================
# Plot the aurtocorrelation from same interval on all primaries
# =============================================================================
if False:
    start = 300
    end = 375
    endier = 500
    x = np.arange(end - start)
    plt.figure(figsize=(12, 5))
    for col in cur.columns:
        ac(cur.loc[start: end, col].values, label=col)
    plt.legend()
    
    # Plot corresponding values
    plt.figure(figsize=(12, 5))
    for col in cur.columns:
        plt.plot(cur.loc[start: end, col].values \
                 - cur.loc[start: end, col].values[0], label=col)
    plt.legend()
    

# =============================================================================  
# Plot Autocorrelation on currencies, both Flattened and Wave. Plot with values.  
# =============================================================================
if False:

    start = 10000
    window = 120
    
    # Positions
    # ---------------------------------------------------------------------
    # Regular
    for col in cur.columns:
        plt.figure(figsize=(12,2))            
        plt.plot(cur.loc[start: start + window, col].values)
        plt.plot(window, cur.loc[start + window, col] , '|', markersize=85, color='black')
        plt.title(col)
        plt.show()
    # Flattened         
    plt.figure(figsize=(12,2))
    for col in cur.columns:
        plt.plot(Wave(cur.loc[start:start+window*2, col].values\
                 - cur.loc[start:start+window*2, col].values[0]).channel.flattened, label=col)
    plt.plot(window, 0, '|', markersize=85, color='black')
    plt.title('Flattened')
    plt.legend()
    # Waved
    plt.figure(figsize=(12,2))
    msg = ''
    for col in cur.columns:
        wave = Wave(cur.loc[start:start+window*2, col].values\
                 - cur.loc[start:start+window*2, col].values[0])
        plt.plot(wave.wave, label=col)
        msg += '{}:  {:.2f}\n'.format(col, wave.frequency)
    plt.title('Waved')
    plt.plot(window, 0, '|', markersize=85, color='black')
    plt.legend()
    print(i)
    
    # Autocorrelation
    # ---------------------------------------------------------------------
    # Regular
    plt.figure(figsize=(12,2))
    for col in cur.columns:    
        ac(cur.loc[start: start + window, col].values)
    plt.title('Standard')
    # Flattened         
    plt.figure(figsize=(12,2))
    for col in cur.columns:    
        ac(Wave(cur.loc[start: start + window, col].values).channel.flattened)
    plt.title('Flattened')
    # Waved
    plt.figure(figsize=(12,2))
    for col in cur.columns:    
        ac(Wave(cur.loc[start: start + window, col].values).wave, label=col)
    plt.title('Waved')
    plt.legend()
    
    # Print Stats
    # ---------------------------------------------------------------------
    plt.show()
    print('Primary, Frequency\n-------------------')
    print(msg)


# =============================================================================
# Plot Rolling Variance on one currency over all df
# =============================================================================
if False:
    a = cur.loc[: ,'aud'].rolling(20).cov() \
        / cur.loc[: ,'aud'].rolling(20).std()
    plt.plot(a)
    






groups = get_groups(group, 600)




###############################################################################
# Plotting experiments
###############################################################################   


if True:
    # =============================================================================
    # Plot given and calculated ratios as well as differnce
    # =============================================================================
    start = 1000
    window = 1000
    calculated = cur.loc[start: start + window, 'eur'].values \
               / cur.loc[start: start + window, 'usd'].values
    given = ratios.loc[start: start + window, 'eur_usd'].values
    given_minus_calc = given - calculated
    # Double axis
    fig, ax = plt.subplots(2, 1, figsize=(10,8))
    ax[0].plot(cur.loc[start: start + window, 'eur'].values \
             / cur.loc[start: start + window, 'usd'].values, 
             label = 'calculated')
    ax[0].plot(ratios.loc[start: start + window, 'eur_usd'].values, label='given')
    ax[0].legend()
    # Plot the difference between given and calculated
    ax[1].plot(given_minus_calc, 'o', label='given - calc')
    ax[1].plot(pd.Series(given_minus_calc).rolling(60).mean().values)
    ax[1].legend()
    ax[1].plot(np.zeros(window), color='black')
    plt.tight_layout(); plt.show()

    
if False:
    for i in range(len(pair_values)):
        pair = pair_names[i].split('_')
        created = cur.loc[pair[0], pair[0]].values / cur.loc[pair[1], pair[1]].values
        given = pair_values[pair_names.index(pair_names[i])]
        plt.figure(figsize=(14,4))
        plt.title(str(pair_names[i]).upper() + ' created and givenby primary unscaled and Uncentered')
        plt.plot(created, color = 'blue', label= 'created')    
        plt.plot(given, color = 'orange', label= 'given') 
        plt.legend()
        plt.tight_layout()
        plt.show()    
    # Ratio with mean unscaled.  But centered first value at zero
    for i in range(len(pair_values)):
        pair = pair_names[i].split('_')
        created = cur.loc[pair[0]].mean(axis=1).values / cur.loc[pair[1]].mean(axis=1).values
        given = pair_values[pair_names.index(pair_names[i])]
        plt.figure(figsize=(14,4))
        plt.title(str(pair_names[i]).upper() + ' created and given by mean unscaled.  First values centered at zero')
        plt.plot(created - created[0], color = 'blue', label= 'created')    
        plt.plot(given - given[0], color = 'orange', label= 'given') 
        plt.legend()
        plt.tight_layout()
        plt.show()
    # Ratio with mean scaled minmax.  But centered first value at zero
    for i in range(len(pair_values)):
        pair = pair_names[i].split('_')
        created = cur.loc[pair[0]].mean(axis=1).values / cur.loc[pair[1]].mean(axis=1).values
        created = StandardScaler().fit_transform(created.reshape(-1, 1)).ravel()
        given = pair_values[pair_names.index(pair_names[i])]
        given = StandardScaler().fit_transform(given.reshape(-1, 1)).ravel()
        plt.figure(figsize=(14,4))
        plt.title(str(pair_names[i]).upper() + ' created and givenby mean SCALED (first value centered at zero')
        plt.plot(created - created[0], color = 'blue', label= 'created')    
        plt.plot(given - given[0], color = 'orange', label= 'given') 
        plt.legend()
        plt.tight_layout()
        plt.show()
        

if False:
    
    # Plot all currencies, with Aa, a's mean, and other seperated out]
    # -------------------------------------------------------------------------
    for c0 in currencies:
        plt.figure(figsize=(14,4))
        plt.title(str(c0) + ' - All created values')
        plt.plot(cur.loc[c0, c0].values - cur.loc[c0, c0].values[0], color='black', label = c0, linewidth=4)    
        for c1 in currencies:
            if c1 != c0:
                plt.plot(cur.loc[c0, c1].values - cur.loc[c0, c1].values[0], label= c1)    
        plt.plot(cur.loc[c0].mean(axis=1).values - cur.loc[c0].mean(axis=1).values[0], color='black', label='mean')
        plt.legend()
        plt.tight_layout()
        plt.show()

if False:
     
    # Plot together the primaries, the ratio, and the means
    # ---------------------------------------------------------------------
    plt.figure()
    plt.plot(cur.loc['aud'].mean(axis=1).values - cur.loc['aud'].mean(axis=1).values[0], color='c', label='aud_mean')
    plt.plot(cur.loc['cad'].mean(axis=1).values - cur.loc['cad'].mean(axis=1).values[0], color = 'darkorange', label='cad_mean')
    plt.figure()
    plt.plot(cur.loc['aud', 'aud'].values - cur.loc['aud', 'aud'].values[0], color='cadetblue', label='aud_primary')
    plt.plot(cur.loc['cad', 'cad'].values - cur.loc['cad', 'cad'].values[0], color='coral', label='cad_primary')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(pair_values[instrument_list.index('AUD_CAD')] - pair_values[instrument_list.index('AUD_CAD')][0])
    plt.title('The Ratio.  AUD CAD')
    plt.legend()
    plt.show()
    
    
    
if False:
    
    # Anything interesting changing right now ? 
    # ---------------------------------------------------------------------
    stop = 240
    # cur.index = cur.index.swaplevel()
    collection = cur.loc[1] - cur.loc[0]  
    for i in cur.index.levels[0].values[2:stop]:
        collection = collection.append((cur.loc[i] - cur.loc[i - 1]).abs())
    locations = np.repeat(cur.index.levels[0].values[2:stop + 1], cur.columns.shape[0])
    curs = np.tile(cur.columns, int(collection.shape[0] / cur.columns.shape[0]))        # Drop 1st row placeholder
    collection.index = [locations, curs]
    # turn into df
    collection = pd.DataFrame(collection, 
                              columns = cur.columns)
    collection.index = collection.index.swaplevel() 
    collection.loc['usd'].mean(axis=1).plot()
    collection.loc['eur'].mean(axis=1).plot()
    plt.figure()
    plt.plot(pair_values[0][1:stop])



    # Turn back index levels
    #cur.index = cur.index.swaplevel()        











if False:
    fig, ax = plt.subplots(cur.columns.shape[0], 1, figsize=(16,10), 
                           sharex=True)
    for i in range(cur.columns.shape[0]):
        ax[i].plot(cur.loc[0:100, cur.columns[i]].values)


if False:
    plt.figure(figsize = (15, 12))
    for col in cur.columns:
        plt.plot(cur.loc[0:1000, col].values - cur.loc[0:1000, col].values[0],
                 label = col)
        plt.plot(cur.loc[0:1000, col].values - cur.loc[0:1000, col].values[0],
                 '+')    
    plt.legend()
    plt.tight_layout()
    plt.show()
        
        
        
if False:
    plt.figure(figsize = (15, 12))
    for col in cur.columns:
        autocorrelation_plot(cur.loc[:1500, col].values)


    
    
    
    
    
    
    