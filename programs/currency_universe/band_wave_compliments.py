import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
from scipy.stats import rv_discrete
from pandas.plotting import autocorrelation_plot as ac
from sklearn.preprocessing import StandardScaler
import os; os.chdir('/northbend')
from libraries.currency_universe import get_currencies
from libraries.currency_universe import get_currency_rolling_correlation_waves
from libraries.currency_universe import get_currency_rolling_correlations
from libraries.transformations import get_groups
from classes.channel import Channel
from classes.wave import Wave
from libraries.bands import get_bollinger_bands
from libraries.bands import get_channel_bands
from libraries.bands import get_band_diff
from libraries.bands import get_two_curr_diff
from libraries.bands import get_correlation 
from libraries.bands import get_rolling_two_curr_diff
from libraries.bands import get_rolling_correlation_waves
from libraries.bands import get_channel_waves
from libraries.bands import get_bollinger_waves
from libraries.bands import get_bollinger_position_waves
from libraries.bands import get_channel_position_waves
from libraries.bands import get_variance_waves                 
from libraries.bands import get_position_waves  
from libraries.bands import get_mean_waves 
from libraries.bands import get_position_waves_df
from libraries.bands import get_slope_waves_mean_df
from libraries.bands import get_channel_bands_mean_df


# Set Environment
pd.set_option('display.width', 1000)
pd.set_option('max.columns', 15)
np.warnings.filterwarnings('ignore')


'''
GOAL:
    
    To build a prediction model,
    Using the currency universe,
    That uses a few (more or less) standard TA techniuqes on each currency.
    That provides measures of predictability:
        per currency    
        per indicator.
    Where we can Match up Cycles (moments) in Indicators:
        And by comparing their measures and predictive scores, 
        between every currencies,
    Make placements.  Or not.
    
    
    Looking at One currency at a time:
        Bring in all indicators and measures on currency
        
        
TO DO:    
    
    Work with One Currency
       Graph all indicators on one currency.
>           Intuit Patterns.  
            Additional (related) indicator creation ? ? ? ?
            Intuit different strategies provided by indicators.
        Analyze indicators:
            how to measure wave tops --- grumble
            can we use sklearn log regress to see what variables work well?
            what do we correlate it with to see if it  will work?
            Double check calculations
            Do they offer predictive value?
            Create predictive scoring model based on frequency (and solution)
    Work with all Currencies 
        How to define and recognize (cyclic) matches
        How to recognize best placements
    Develope stategy.
        Later.....


Now:

    Analyze.
    Does posiiton and channel mean have predictive power ?




NoOTES AND QUESTIONS:
    
    Pos mean is diff from channel mid.  Why?
    
    position wave mean on both touching -2 is not a bad indication of
    currency going jup
    
    Check out the correlation between the mean position wave and currency
    
    it appear that the rolling/  flattened diff wave, when corr i low, 
    indicates a spike or return    

    don't nmeed to find 80% predictive power for any ind on any curr....
    
    Channel mean - bollinger mean makes a nice wave.
    Can we get just a little predictive power out of this new indicator
    Can we match wave parameters between currencies for predictve power

    Maybe std waves  on multiple window - overkill ?

    why does the ind have a properties similar to log !!!!!!!!!
        ind(cur1) - ind(cur2) = ind(cur1 / cur2)   (!)


Basic Startegies:

    How to choose right window for


    When the mean of position waves is above 2 and below 2 for respective currencies
        Analysis: 
            make mean mean a df, mean pos a df, 
            if above 2, prob, it hits 60% of way before hitting %30 up
            same for opposite
            
            prob ratio moves if both are in opposite.\
            


    '''

"""
STRATEGY:  Currency Collection Comprimise:
    
    Currencies move mirror to the (1 - sum) of every other.
    How to make two collections, each missing one (different currnecy)
        these currencies make the ratio.
        
    There are (at least for now i see) two things:
        Scoring location potential.
        Choosing locations to make play
        
        I don't have eaither yet but am working towards some things
        Particularly, where currencies seperate into clumped positions and  
        slopes.
        
    
    So far, we are using:
        Slope Waves (mean) 
        Position Waves (mean) - don't yet know from normal or channel
        Mean Waves ( not using right now)
        Correlation - to help decide groups
        
        What windows work best?
        How do I combine windows with granularity.
        Do not yet even close to have thresholds made  (fixed or relative)


   just poked doesn't mean much but:
       (channel) position * slope * sign(slope) = predicted movement
       
    Rolling Correlation changes nice and smooth.
        It appears that we can somewhat count on it.... :>
        
    What's goin on wiht scaled_df sum axis = 1 ?
        
    For currencies that are wild and highly variable at moment, putting 
        them in each collection negates thier volatility. (maybe ? ? )
        Or not just wild, perhaps the ones that have just spiked. 
        
    spikes have higher (lower) slope ( and change......  )
    
    can we seperate by local correlation?
        meaning: things that move apart from each other are in seperate groups
        
    Wait - For ourt collections - we want to choose the group that is
    going highest and the group that is going lowest.  Again - 
        that really just means take the two highest and lowest ?! ?
    
    Placement Strategy:
        Perhaps if I am able to score things with a prediction 
            prob of moving _ given _ per currency.
        then I might be able to combine them.
        Otherwise certainly using the std ratio would be a good thing.
            or evena channel on the ratio itsefl.   no.
            
            
    who spend most of the longer interval time (say, 1000) uncorrelated
    but are highly correlated at position _
    
    Get when slope is not large for any of them ?  ?
        
"""



'''

Next:
    
    rank columns of correlation
    Build Results df, calculating indicator on entire frame
    like usual:
        score each parameter seperetely then filter on each and get the:
            index location
            ratio
            direction
            stop and target (later)
            
    Filter to find locations, placements, etc≥

'''     


    

###############################################################################
if 0:  # Call currency matrix.
###############################################################################
    
    # Get currency universe
    granularity = 'M1'
    _from = '2018-01-01T00:00:00Z'
    _to   = '2018-04-01T00:00:00Z'
    currency_dictionary = get_currencies(granularity, _from, _to)
    cur     = currency_dictionary['currencies']
    curdiff = currency_dictionary['currencies_delta']
    ratios  = currency_dictionary['ratios']
    cur_set = currency_dictionary['currency_set']
    

    
################################################################################
if  1: # Specify Intervals, Construct Indicators, etc.
###############################################################################

    

    
    # Wave Parameters
    indicator_std = 2    
    windows = np.array([5, 10, 20, 30, 60])
    windows = np.arange(5, 90, 5)
    windows = np.arange(15, 600, 25)
    windows = np.array([30, 60, 120, 240, 480])
    
    # Plotting Paramenters
    interval = 1000
    end =  random.choice(np.arange(interval + windows.max(), cur.shape[0]))
    color_list = plt.cm.Blues(np.linspace(.25, .75, windows.shape[0]))    
    
    # Reduce size of cur to working interval values
    df = cur.loc[end - (interval + windows.max()): end]
    
    # Get Waves df based on windows array
    correlation_waves = get_currency_rolling_correlation_waves(df, ratios.columns, windows)
    slopes_waves = get_slope_waves_mean_df(df, windows)
    channel_mean_wave = get_channel_bands_mean_df(df, windows)
    mean_position_waves = get_position_waves_df(df, False, windows)
    chan = 0
    
    # reduce dataframe to not include beggining window.max() values
    df = df.loc[end - interval: end]
    correlation_waves = correlation_waves.loc[end - interval: end]
    slopes_waves = slopes_waves.loc[end - interval: end]
    mean_position_waves = mean_position_waves.loc[end - interval: end]
    channel_mean_wave = channel_mean_wave.loc[end - interval: end]
    # Get Ratios on interval
    rat = ratios.loc[end - interval: end]
    
    # Scaled Dataframes
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df), columns = df.columns, index=df.index)
    channel_mean_wave.fillna(0, inplace=True)
    channel_mean_scaled = pd.DataFrame(StandardScaler().fit_transform(channel_mean_wave), columns = df.columns, index=df.index)
    
    
    
###############################################################################
if 1: # Calculate measures and analytics.  Build Results DF.
###############################################################################

    
    # To build : How to measure location
    #    location_measure = abs((slopes_waves * mean_position_waves ) )
    #    location_measure *= np.sign(mean_position_waves)
    slope_measure = StandardScaler().fit_transform(slopes_waves)
    position_measure = StandardScaler().fit_transform(mean_position_waves)
    location_measure = abs(slope_measure * position_measure)
    location_measure *= np.sign(slopes_waves)

    location_measure = slope_measure * channel_mean_scaled
    location_measure = StandardScaler().fit_transform(location_measure)
    
    #location_measure *= np.sign(mean_position_waves)
    location_measure = pd.DataFrame(location_measure, 
                                    columns = df.columns,
                                    index = df.index)
    
    # Get smallest and largest measure and put together as ratio     
    largest = location_measure.idxmax(axis=1)
    smallest = location_measure.idxmin(axis=1)
    measure_coupling_1 = (largest + '_' + smallest).astype(str)
    measure_coupling_2 = (smallest + '_' + largest).astype(str)    
    # make sure ratio is put together in correct order
    isin = measure_coupling_2.isin(ratios.columns)
    measure_coupling = measure_coupling_1
    measure_coupling[isin] = measure_coupling_2[isin]
    
    ''' Try Smallest correlation '''    
    # Get smallest Correlation
    smallest_correlation = correlation_waves.idxmin(axis=1)
    
    ''' Try No correlation '''
    smallest_correlation = correlation_waves.idxmax(axis=1)
    
    ''' Try Smallest Correlation with smallest pairs '''
    smallest_correlation = correlation_waves.idxmin(axis=1)
    
    
    first = pd.DataFrame(location_measure.aud)
    first.columns = ['first']
    second = pd.DataFrame(location_measure.aud)
    second.columns = ['second']
    first['location'] = first.index
    second['location'] = second.index
    for col in location_measure.columns:
        first_index = location_measure[(location_measure.abs().rank(axis=1)[col] == 1)].index.values
        second_index = location_measure[(location_measure.abs().rank(axis=1)[col] == 2)].index.values
        first.loc[first_index] = str(col)
        second.loc[second_index] = str(col)
    first['location'] = first.index
    second['location'] = second.index
    first_with_slope = slopes_waves.stack().loc[[tuple(x) for x in first[['location', 'first']].values]]
    second_with_slope = slopes_waves.stack().loc[[tuple(x) for x in second[['location', 'second']].values]]
    rank_and_slope = first_with_slope
    rank_and_slope.columns = ['first', 'first_slope']
    rank_and_slope['second'] = second_with_slope
    
    
    # Anywhere they match? 
    match = measure_coupling == smallest_correlation
    match_locations = largest.loc[match].index.values
    match_ratios = measure_coupling[match].values
    
    results = pd.DataFrame()
    results['location'] = match_locations
    results['ratios']   = match_ratios
    results['direction'] = 1
    results.index = match_locations
    results.loc[isin[match].values, 'direction'] = 0
    
    

        
    ################################################################################
    # Plot Strategy Results
    ################################################################################
    if 1: 
        
        '''
        Plot every placement that the strategy has choosen:
            Plot each ratio seperetely
            color location dot as long (green) or short (red)
        '''
        for r in results.ratios.unique():
            plt.figure(figsize=(10, 10))
            rat[r].plot()
            df_ind = results[results.ratios == r]
            # PLot long
            df_win = df_ind.loc[df_ind.direction == 1]
            plt.plot(df_win['location'].values, 
                     rat.loc[df_win['location'].values, r].values, 
                     'o', color = 'green')
            # Plot short
            df_lose = df_ind.loc[df_ind.direction == 0]
            plt.plot(df_lose['location'].values, 
                     rat.loc[df_lose['location'].values, r].values, 
                     'o', color = 'red')
            plt.title(r)
    


        
        '''
        column = 'gbp_nzd'
        plt.figure()
        ratios.loc[end - interval:end, column].plot()
        plt.title('Ratio oan ' + str(column))
        plt.figure()
        correlation_waves.loc[end - interval:end, column].plot()
        plt.title('Correlation Waves on ' + str(column))
        '''
    
###############################################################################
# Plot strategy indicators over interval with currencies, ratios, etc.
###############################################################################
if 1:      
    
    '''
    Plots:
        1.  
            1.  Every Currency
            2.  Every Slope                 (by mean waves function) 
            3.  Every Channel Position      (by mean waves function) 
            4.  Every channel               (by mean waves function) 
            5.  Every                       (by mean waves function) 
            6.  Location Measure
        2. Every Ratio 
        3. Every correlation                (by mean waves function) 
    ''' 
    
    

    
    # Plot 1 - every currency, all indicators
    # -------------------------------------------------------------------------

    # Call Subplots
    fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    
    # Plot currencies
    df_scaled.plot(ax=ax[0])
    # Plot Slopes waves
    slopes_waves.plot(ax=ax[1])
    # Plot mean position Waves
    mean_position_waves.plot(ax=ax[2])
    # Plot sum (slopes  * mean)
    location_measure.plot(figsize=(10, 3), ax=ax[3])
    location_measure.mean(axis=1).plot(ax=ax[3], color='black')
    # Plot Channel Mean Position (c3)
    channel_mean_scaled.plot(ax=ax[4])
    

    # get std lines on  channel position
    x = np.arange(end - interval, end + 1)
    ax[1].plot(x, np.zeros(df.shape[0]), color='black')
    ax[2].plot(x, np.ones(df.shape[0]) * indicator_std, color='black')
    ax[2].plot(x, np.ones(df.shape[0]) * -indicator_std, color='black')
    # Legends
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    ax[4].legend()
    # Name Rows
    for row in range(3):
        # Name columns and Rows
        rows = ['Currency', 
                'Slope', 
                'Mean Position', 
                'Location measure']
        for a, row in zip(ax[:], rows):
            a.set_ylabel(row, rotation=90, size='large')   
            
    # Show Plot
    plt.tight_layout()
    plt.show()
    
    
    # Plot 2 & 3 - Ratios and Correlations
    # -------------------------------------------------------------------------
    
    # Plot 2: Ratios
    plt.figure(figsize=(10, 10))
    for column in rat:
        plt.plot(rat[column] - rat[column].values[0], label=column)
    plt.title('Ratios')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: Correlations
    correlation_waves.plot(figsize=(10, 10))
    plt.plot(x, np.zeros(correlation_waves.shape[0]))
    plt.title('Correlations')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    




































































###############################################################################
if 0:  # Set up intervals on Currencies and ratios.  Call all Indicators.
###############################################################################


    # Plotting Paramenters
    windows = np.array([5, 10, 20, 30, 60, 90, 120])
    interval = 1000
    indicator_std = 2
    end =  random.choice(np.arange(interval, cur.shape[0]))    
    # Currencies 
    currency1 = 'eur'
    currency2 = 'usd'
    cur1 = cur.loc[end - interval: end, currency1].values
    cur2 = cur.loc[end - interval: end, currency2].values
    cur1_diff = curdiff.loc[end - interval: end, currency1].values
    cur2_diff = curdiff.loc[end - interval: end, currency2].values
    
    # For renaming
    boll_waves_1 = get_bollinger_waves(cur1, windows)
    chan_waves_1 = get_channel_waves(cur1, windows)
    boll_waves_2 = get_bollinger_waves(cur2, windows)
    chan_waves_2 = get_channel_waves(cur2, windows)
    # Mean Waves
    mean_waves = boll_waves_1.copy()
    mean_waves_flat = chan_waves_1.copy()
    mean_waves_difference = mean_waves - mean_waves_flat
    # Variance Waves
    variance_waves = get_variance_waves(cur1, flattened = False, windows=windows)
    variance_waves_flat = get_variance_waves(cur1, flattened = True, windows=windows)
    variance_waves_difference = variance_waves - variance_waves_flat
    # Position Waves
    mean_position_waves = get_position_waves(cur1, False, windows=windows)
    mean_position_waves_flat = get_position_waves(cur1, True, windows=windows)
    mean_position_waves_difference = mean_position_waves - mean_position_waves_flat

    
    

###############################################################################
if 0:  # Plot all Indicators for ONE currency.  Develope movement theories from here.
###############################################################################    

    '''
        We are comparing Flattened and Normal and difference of Values for each:
            Rolling mean.    ( and mean/variance)
            Rolling Std.     ( and mean/variance)
            Rolling Position ( and mean/variance)
        Against Currency 
        
    '''
    
    # Plot
    # -----------------------------------------------------------------------------
    fig, ax = plt.subplots(4, 3, figsize=(10, 10), sharex = True)    
    color_list = plt.cm.Blues(np.linspace(.25, .75, windows.shape[0]))

    # Currency (All Top Row)
    # -----------------------------------------------------------------------------
    ax[3, 0].plot(cur1)
    ax1 = ax[3, 0].twinx()
    ax1.plot(Channel(cur1).flattened, color='orange')
    ax[3, 1].plot(cur1_diff)
    ac(cur1, ax=ax[3,2])

    # Mean (Left Column)
    # -----------------------------------------------------------------------------
    # Normal 
    mean_waves.plot(color = color_list, ax=ax[0, 0])
    mean_waves.mean(axis=1).plot(color='black', ax=ax[0, 0])
    ax1 = ax[0, 0].twinx()
    mean_waves.std(axis=1).plot(color='orange', ax=ax1)
    # Flattened
    mean_waves_flat.plot(color = color_list, ax=ax[1, 0])
    mean_waves_flat.mean(axis=1).plot(color='black', ax=ax[1, 0])
    ax1 = ax[1, 0].twinx()
    mean_waves_flat.std(axis=1).plot(color='orange', ax=ax1)
    # Difference
    mean_waves_difference.plot(color = color_list, ax=ax[2, 0])
    mean_waves_difference.mean(axis=1).plot(color='black', ax=ax[2, 0])
    ax1 = ax[2, 0].twinx()
    mean_waves_difference.std(axis=1).plot(color='orange', ax=ax1)
    
    # Variance
    # -----------------------------------------------------------------------------
    # Normal
    variance_waves.plot(color=color_list, ax=ax[0, 1])
    variance_waves.mean(axis=1).plot(color='black', ax=ax[0, 1])
    ax1 = ax[0, 1].twinx()
    variance_waves.std(axis=1).plot(color='orange', ax=ax1)
    # Flattened
    variance_waves_flat.plot(color=color_list, ax=ax[1, 1])
    variance_waves_flat.mean(axis=1).plot(color='black', ax=ax[1, 1])
    ax1 = ax[1, 1].twinx()
    variance_waves_flat.std(axis=1).plot(color='orange', ax=ax1)
    # Difference
    variance_waves_difference.plot(color=color_list, ax = ax[2, 1])
    variance_waves_difference.mean(axis=1).plot(color='black', ax=ax[2, 1])
    ax1 = ax[2, 1].twinx()
    variance_waves_difference.std(axis=1).plot(color='orange', ax=ax1)    
    
    
    # Position
    # -----------------------------------------------------------------------------
    # Normal
    mean_position_waves.plot(color=color_list, ax=ax[0, 2])
    mean_position_waves.mean(axis=1).plot(color='black', ax=ax[0, 2])
    ax1 = ax[0, 2].twinx()
    mean_position_waves.std(axis=1).plot(color='orange', ax=ax1)
    # Flattened
    mean_position_waves_flat.plot(color=color_list, ax=ax[1, 2])
    mean_position_waves_flat.mean(axis=1).plot(color='black', ax=ax[1, 2])
    ax1 = ax[1, 2].twinx()
    mean_position_waves_flat.std(axis=1).plot(color='orange', ax=ax1)
    # Difference
    mean_position_waves_difference.plot(color=color_list, ax = ax[2, 2])
    mean_position_waves_difference.mean(axis=1).plot(color='black', ax=ax[2, 2])
    ax1 = ax[2, 2].twinx()
    mean_position_waves_difference.std(axis=1).plot(color='orange', ax=ax1) 
    # Plot the std_ratio
    ax[0, 2].plot(np.ones(cur1.shape[0]) * indicator_std, color='grey')
    ax[0, 2].plot(np.ones(cur1.shape[0]) * -indicator_std, color='grey')
    ax[1, 2].plot(np.ones(cur1.shape[0]) * indicator_std, color='grey')
    ax[1, 2].plot(np.ones(cur1.shape[0]) * -indicator_std, color='grey')    
    ax[2, 2].plot(np.ones(cur1.shape[0]) * indicator_std, color='grey')
    ax[2, 2].plot(np.ones(cur1.shape[0]) * -indicator_std, color='grey')       
 
    # Plot
    # -----------------------------------------------------------------------------
    # Remove Legends on waves
    for row in range(3):
        for col in range(3):
            ax[row, col].legend_.remove()
    # Name columns and Rows
    columns = ['Mean', 'Variance', 'Position']
    rows    = ['Regular', 'Flattened', 'Difference', 'Currency']
    for a, col in zip(ax[0], columns):
        a.set_title(col)
    for a, row in zip(ax[:,0], rows):
        a.set_ylabel(row, rotation=90, size='large')
    # Plot
    fig.tight_layout()
    plt.tight_layout()
    plt.show()
            
    
    
    
    
    

    
    
    
###############################################################################
if 0:  # Opposting Position Strategy (predictability measure)
###############################################################################


    # Plotting Paramenters
    windows = np.arange(10, 240, 15)
    windows = np.arange(10, 511, 50)
    windows = np.arange(5, 555, 5)
    windows = np.array([5, 10, 20, 30, 60, 90, 120])
    interval = 1000
    indicator_std = 2
    end =  random.choice(np.arange(interval, cur.shape[0]))    
    # Currencies 
    currency1 = 'eur'
    currency2 = 'usd'
    cur1 = cur.loc[end - interval: end, currency1].values
    cur2 = cur.loc[end - interval: end, currency2].values
    cur_diff_1 = cur_diff.loc[end - interval: end, currency1].values
    cur_diff_2= cur_diff.loc[end - interval: end, currency2].values
    
    flat_mean_1 = get_position_waves(cur1, True, windows=windows).mean(axis=1).values
    flat_mean_2 = get_position_waves(cur2, True, windows=windows).mean(axis=1).values
    norm_mean_1 = get_position_waves(cur1, False, windows=windows).mean(axis=1).values
    norm_mean_2 = get_position_waves(cur2, False, windows=windows).mean(axis=1).values
    channel_pos_1 = get_channel_position_waves(cur1, windows, indicator_std).mean(axis=1).values
    channel_pos_2 = get_channel_position_waves(cur2, windows, indicator_std).mean(axis=1).values


    # JUST BY POSITION WAVES AND RATIO
    fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex = True)    
    
    
    # Ax 2 - Flatttened Position
    ax[0].plot(flat_mean_1, label='cur 1')
    ax[0].plot(flat_mean_2, label='cur 2')
    ax[0].plot(np.ones(cur1.shape[0]) * indicator_std, color='grey')
    ax[0].plot(np.ones(cur1.shape[0]) * -indicator_std, color='grey')
    ax[0].legend()
    ax[0].set_title('Flat position')
    
    
    windows = np.arange(5, 555, 5)
    # Ax 2 - Mean Position

#    ax[1].plot(norm_mean_1, label='cur 1')
#    ax[1].plot(norm_mean_2, label='cur 2')
    ax[1].plot(pd.Series(norm_mean_1).rolling(5).mean().values, label='rolling 1')
    ax[1].plot(pd.Series(norm_mean_2).rolling(5).mean().values, label='rolling 2')
    ax[1].plot(np.ones(cur1.shape[0]) * indicator_std, color='grey')
    ax[1].plot(np.ones(cur1.shape[0]) * -indicator_std, color='grey')
    ax[1].legend()
    ax[1].set_title('Normal position')
    
    
    # Channel Position
    ax[2].plot(channel_pos_1, label='cur 1')
    ax[2].plot(channel_pos_2, label='cur 2')
    ax[2].plot(np.ones(cur1.shape[0]) * indicator_std, color='grey')
    ax[2].plot(np.ones(cur1.shape[0]) * -indicator_std, color='grey')
    ax[2].legend()
    ax[2].set_title('Channel Position')
    
    
    # Diff in mean, flat, channel positions
    ax[3].plot(channel_pos_1 - channel_pos_2 + .002, label='channel')
    ax[3].plot(flat_mean_1 - flat_mean_2, label='flat')
    ax[3].plot(norm_mean_1 - norm_mean_2 - .002, label='norm')
    ax[3].legend()
    ax[3].set_title('Difference in Currency Positions')
    
    ax[4].plot(cur_diff_1)
    ax[4].plot(cur_diff_2)
    ax[4].plot(cur_diff_1 - cur_diff_2)    
    
    fig.tight_layout()
    plt.show()
    


    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex = True)    
    ax[0].plot(cur1)
    ax[1].plot(cur2)
    ax[2].plot(cur1 / cur2)
    plt.show()
    
    
    short = (norm_mean_1 > indicator_std) & (norm_mean_2 < -indicator_std)
    long = (norm_mean_1 < -indicator_std) & (norm_mean_2 > indicator_std)
    long = np.arange(long.shape[0])[long]
    short = np.arange(short.shape[0])[short]
    
    
    
    
    
        
    '''
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex = True)    
    color_list = plt.cm.Blues(np.linspace(.25, .75, windows.shape[0]))



    get_channel_waves(cur1, windows).plot(ax = ax[0, 0], color=color_list)
    ax[0, 0].plot(cur1, color='orange')
    get_channel_waves(cur1, windows).mean(axis=1).plot(ax = ax[0, 0], color='black')
    
    
    get_channel_waves(cur2, windows).plot(ax = ax[0, 1], color=color_list)
    ax[0, 1].plot(cur2, color='orange')
    get_channel_waves(cur2, windows).mean(axis=1).plot(ax = ax[0, 1], color='black')


    
    get_position_waves(cur1, False, windows=windows).plot(ax = ax[1, 0], color=color_list)
    get_position_waves(cur1, False, windows=windows).mean(axis=1).plot(ax = ax[1, 0], color='black')
    ax[1, 0].plot(np.ones(cur1.shape[0]) * indicator_std, color='grey')
    ax[1, 0].plot(np.ones(cur1.shape[0]) * -indicator_std, color='grey')
    
    
    
    get_position_waves(cur2, False, windows=windows).plot(ax = ax[1, 1], color=color_list)
    get_position_waves(cur2, False, windows=windows).mean(axis=1).plot(ax = ax[1, 1], color='black')
    ax[1, 1].plot(np.ones(cur1.shape[0]) * indicator_std, color='grey')
    ax[1, 1].plot(np.ones(cur1.shape[0]) * -indicator_std, color='grey')


    plt.show()
    '''
    

    
###############################################################################
# Analyze movement theories.  Develop statistical tests long term.
###############################################################################


# Currency at end of channel by Mean Waves
# -----------------------------------------------------------------------------
if 0:
    windows = np.arange(10, 500, 15)
    
    # Get mean waves per currency
    all_mean_waves = pd.DataFrame()
    for column in cur.columns:
        wave = 0
    
    
    
    # Get flattened waves per currency
    all_mean_waves_flat = pd.DataFrame()  
    
    
    
    windows = np.arange(5, 101, 5)
    
    a = get_channel_bands(cur1, 30)
    b = get_mean_waves(cur1, True, [30])
    c = get_mean_waves(cur1, False, [30])
    
    a = get_channel_waves(cur1, windows).mean(axis=1).values
    b = get_mean_waves(cur1, False, windows).mean(axis=1).values
    c = get_mean_waves(cur1, True, windows).mean(axis=1).values
    
    
    fig, ax = plt.subplots(2, 1,figsize=(10, 10), sharex=True)
    
    #ax[0].plot((b -b.loc[30]).values, label='get mean waves FLAT')
    #ax[0].plot((c -c.loc[30]).values, label='get mean waves')
    #ax[0].plot(a['middle'] - a['middle'][30], label= 'get channel waves')
    ax[0].plot(cur1, label='Currency')
    ax[0].plot(a, label='channel_waves')
    ax[0].plot(b, label='mean_waves')
    ax[0].plot(c + c[30], label='mean_waves_flat')
    ax[0].legend()





















'''
# Plot it up    
fig, ax = plt.subplots(3, 1,figsize=(10, 10), sharex=True)
# Plot that shit
ax[0].plot(cur1, label='currency')
ax[0].legend()
# Plot that shit
ax[1].plot(adjusted - adjusted[0], label='adjusted')
ax[1].legend()
# Plot that Shit
ax[2].plot(vol, label='Volume')
ax[2].legend()
# PLot it up
plt.tight_layout()
plt.show()
'''

'''
So.
It appears that the middle channel is not the same as flattened mean.
Does it contain useful information?
It appears that it might hit its highs and lows earlier - which , if that is
    true - could be super important.
    
    
'''































    
###############################################################################
if 0: # Quick Side Step into Volume
###############################################################################
    
    if 0:
        
        
        # Get Volume universe
        granularity = 'M1'
        _from = '2018-01-01T00:00:00Z'
        _to   = '2018-04-01T00:00:00Z'
        currency_dictionary = get_currencies(granularity, _from, _to, 'volume')
        volume = currency_dictionary['currencies']
        volume_difference = currency_dictionary['currencies_delta']
        volume_ratios = currency_dictionary['ratios']
    
    # Create dAdjusted price by volume
    change_by_volume = cur_diff / volume
    currency = 'aud'
    interval = 1000
    end = random.choice(np.arange(interval, cur.shape[0]))    
    curs = cur.loc[end - interval :end, currency].values
    adjusted = change_by_volume.loc[end - interval: end, currency].cumsum().values
    vol = volume.loc[end - interval :end, currency].values

    # Plot it up    
    fig, ax = plt.subplots(3, 1,figsize=(10, 10), sharex=True)
    # Plot that shit
    ax[0].plot(curs - curs[0], label='currency')
    ax[0].legend()
    # Plot that shit
    ax[1].plot(adjusted - adjusted[0], label='adjusted')
    ax[1].legend()
    # Plot that Shit
    ax[2].plot(vol, label='Volume')
    ax[2].legend()
    # PLot it up
    plt.tight_layout()
    plt.show()

'''
'Wont Need the rest of this in the form it is in.
Will Save for later I suppose

    




# Indicator Paramters

# Create Ratio
ratio = cur.loc[end - interval: end, currency1] \
      / cur.loc[end - interval: end, currency2]
ratio = ratio.values
    # Get Indicators
ind1 = get_band_diff(cur1, indicator_window, indicator_std)
ind2 = get_band_diff(cur2, indicator_window, indicator_std)  
ind3 = get_band_diff(ratio, indicator_window, indicator_std) 
ind1_ind2_diff = get_two_curr_diff(cur1, cur2, indicator_window, indicator_std)
correlations = get_correlation(cur1, cur2, indicator_window)
Second Currency Stuff.  Hold aside for now.
boll_waves_2 = get_bollinger_waves(cur2, windows)
chan_waves_2 = get_channel_waves(cur2, windows)
Channel_position_waves_2 = get_channel_position_waves(cur2, windows)
bolling_position_waves_2 = get_bollinger_position_waves(cur2, windows)
# Get Correlations on Two Currencies
variance_waves = get_rolling_correlation_waves(cur1, cur2, windows)  
rolling_diff = get_rolling_two_curr_diff(cur1, cur2, windows)     

# Get Final Postion on Bollinger and Channel in terms of window deviation (from mean)
channel_position_waves_1 = get_channel_position_waves(cur1, windows)
bolling_position_waves_1 = get_bollinger_position_waves(cur1, windows)

    
    
    
    
###############################################################################
# Visually Compare one ratio, two instruments, 1 window length, 1 std_ratio
###############################################################################
if 0:
   
    fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    # Plot Currencies
    ax[0, 0].plot(cur1 - cur1[0], label='currency 1')
    ax[0, 0].plot(cur2 - cur2[0], label='currency 2')
    ax[0, 0].legend()
    ax[0, 0].set_title('Currencies')
    # Plot ratio
    ax[1, 0].plot(ratio, label='Ratio')
    ax[1, 0].plot(ratios.loc[end - interval: end, 'eur_usd'].values)
    ax[1, 0].set_title('Ratio')
    # Plot indicator 1
    ax[0, 1].plot(ind1['lower'],  color = 'blue')
    ax[0, 1].plot(ind1['middle'], color = 'cadetblue')    
    ax[0, 1].plot(ind1['upper'],  color= 'cyan')
    ax[0, 1].plot(np.zeros(cur1.shape[0]), color='grey')
    ax[0, 1].set_title('Indicatio - Currency 1')
    # Plot indicator 2
    ax[1, 1].plot(ind2['lower'],  color = 'blue')
    ax[1, 1].plot(ind2['middle'], color = 'cadetblue')
    ax[1, 1].plot(ind2['upper'],  color= 'cyan')    
    ax[1, 1].plot(np.zeros(cur1.shape[0]), color='grey')
    ax[1, 1].set_title('Indicator - Currency 2 ')
    # Plot indicator 3
    ax[2, 1].plot(ind3['lower'],  color = 'blue')
    ax[2, 1].plot(ind3['middle'], color = 'cadetblue')
    ax[2, 1].plot(ind3['upper'],  color= 'cyan')    
    ax[2, 1].plot(np.zeros(cur1.shape[0]), color='grey')
    ax[2, 1].set_title('Indicator - Ratio')
    # Plot Difference between Indicator 1 and Indicator 2
    ax[2, 0].plot(ind1_ind2_diff['middle_middle'], color='black')
    ax[2, 0].plot(np.zeros(cur1.shape[0]), color='grey')
    ax[2, 0].set_title('Differnece between Ind1 and Ind2')
    # Corrrelations between two currencies
    ax[3, 0].plot(correlations)
    ax[3, 0].plot(np.zeros(cur1.shape[0]), color='grey')
    ax[3, 0].set_title('Currency Correrlation')
    # Both Indicator middles
    ax[3, 1].plot(ind1['middle'], color = 'blue')
    ax[3, 1].plot(ind2['middle'], color = 'orange')
    ax[3, 1].plot(np.zeros(cur1.shape[0]), color='grey')
    ax[3, 1].set_title('All Three Indicator Middles')
    # Plot
    plt.tight_layout()
    plt.show()


###############################################################################
# Compare many window for Two currencies correlation, boll/channel diff 
###############################################################################
if 0:
    
    mark_position = 680
    # Plot
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    color_list = plt.cm.Blues(np.linspace(.25, .75, windows.shape[0]))
    # Plot Currencies
    ax[0].plot(cur1 - cur1[0], label='currency 1')
    ax[0].plot(cur2 - cur2[0], label='currency 2')
    ax[0].legend()
    for window in windows:
        ax[0].plot(window, 0, '|', markersize=100, color='darkslategray')
    ax[0].set_title('Currencies and Ratio')
    # Plot ratio
    ax[1].plot(ratio - ratio[0], color='grey')
    ax[1].set_title('Ratio')
    # Plot Correlations
    rolling_corr.plot(title='Correlation Windows', ax=ax[2], colors=color_list) 
    ax[2].plot(rolling_corr.mean(axis=1), color='black')
    ax[2].plot(rolling_corr.std(axis=1), color='orange')
    ax[2].plot(np.zeros(cur1.shape[0]), color='grey')
    ax[2].legend_.remove()
    # Plot Difference of Channel / bollinger
    rolling_diff.plot(title='Difference Windows',  ax=ax[3], colors=color_list)
    ax[3].plot(rolling_diff.mean(axis=1), color='black')
    ax[3].plot(rolling_diff.std(axis=1), color='orange')
    ax[3].plot(np.zeros(cur1.shape[0]), color='grey')
    ax[3].legend_.remove()
    # PLot position
    for i in range(4):
        ax[i].plot(mark_position, 0, '|', markersize=100, color='black')
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
###############################################################################
# Plot Channel and Bollinger Waves on two Currencies 
###############################################################################
if 0:

    # Plot
    fig, ax = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
    color_list = plt.cm.Blues(np.linspace(.25, .75, windows.shape[0]))
    # Plot Currency 1 Bollinger 
    boll_waves_1.plot(ax=ax[0, 0], color = color_list)
    boll_waves_1.mean(axis=1).plot(ax=ax[0, 0], color = 'black')  
    ax0 = ax[0, 0].twinx()
    boll_waves_1.std(axis=1).plot(ax=ax0, color = 'orange')
    ax[0, 0].legend_.remove()
    ax[0, 0].set_title('Currency 1 Bollinger Waves')
    # Plot Currency 1 Channel
    chan_waves_1.plot(ax=ax[1, 0], color = color_list)
    chan_waves_1.mean(axis=1).plot(ax=ax[1, 0], color = 'black')  
    ax1 = ax[1, 0].twinx()
    chan_waves_1.std(axis=1).plot(ax=ax1, color = 'orange')
    ax[1, 0].set_title('Currency 1 Channel Waves')
    ax[1, 0].legend_.remove()
    # Plot Currency 2 Bollinger 
    boll_waves_2.plot(ax=ax[0, 1], color = color_list)
    boll_waves_2.mean(axis=1).plot(ax=ax[0, 1], color = 'black')  
    ax2 = ax[0, 1].twinx()
    boll_waves_2.std(axis=1).plot(ax=ax2, color = 'orange')
    ax[0, 1].set_title('Currency 2 Bollinger Waves')
    ax[0, 1].legend_.remove()
    # Plot Currency 2 Channel
    chan_waves_2.plot(ax=ax[1, 1], color = color_list)
    chan_waves_2.mean(axis=1).plot(ax=ax[1, 1], color = 'black')  
    ax3 = ax[1, 1].twinx()
    chan_waves_2.std(axis=1).plot(ax=ax3, color = 'orange')
    ax[1, 1].set_title('Currency 2 Channel Waves')
    ax[1, 1].legend_.remove()
    # Plot Currency 1
    ax[2, 0].plot(cur1)
    ax[2, 0].set_title('Currency 1')
    # Plot Currency 2
    ax[2, 1].plot(cur2)
    ax[2, 1].set_title('Currency 2')
    # Plot
    plt.tight_layout()
    plt.show()
    
    
    
'''
    
    
    
    
