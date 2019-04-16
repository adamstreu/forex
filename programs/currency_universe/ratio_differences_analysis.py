import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
from scipy.stats import rv_discrete
import os; os.chdir('/northbend')
from libraries.currency_universe import get_currencies
from libraries.transformations import get_groups

# Set Environment
pd.set_option('display.width', 1000)
pd.set_option('max.columns', 15)
np.warnings.filterwarnings('ignore')

'''
Purpose:
    Does the difference between the calculated ratio nad the one given
    provide any predictive power?


Questions:
    Single Instance:
        Does the presence of larger differences ACCOMPANY a movement in the RATIOS
        Does the presence of larger differences PREDICT a movement in the RATIOS
        Does the presence of larger differences ACCOMPANY a movement in the CURRENCIES
        Does the presence of larger differences PREDICT a movement in the CURRENICES
    Universe:
        Does the presence of the SUM (MEAN) of larger differences 
            accompany / predict a change in the ratios / currencies
        Does the slope of the rolling mean tell us anything?
    Variables to consider:
        Granularity
            Does small granularity allow any room for placement
    
        
To Do:
    
        Might wanbt to change all of this to percent points for all diff ratios


Notes:
    there is no correlation between ratio diff and (currencies or ratios)
    
    The rolling ratio diff does not appear to add much info for long windows
    
    The rolling mean can have a slope.
    
    Big shifts do happen very quickly.  
        At S5, Can we catch them with causality time remaining?


'''

###############################################################################
# Get instruments.  Call currency Matrix.
###############################################################################
if False:

    # Get currency universe
    granularity = 'S5'
    _from = '2018-01-01T00:00:00Z'
    _to   = '2018-02-01T00:00:00Z'
    currency_dictionary = get_currencies(granularity, _from, _to)
    cur = currency_dictionary['currencies']
    curdiff = currency_dictionary['currencies_delta']
    ratios = currency_dictionary['ratios']
    
    # Create Calculated DataFrames
    calculated = pd.DataFrame()
    for column in ratios.columns:
        cur1, cur2 = str(column).split('_')
        calculated[column] = cur[cur1].values / cur[cur2].values
        
    # Create Diff DataFrame
    diff = pd.DataFrame()
    for column in ratios.columns:
        diff[column] = ratios[column].values - calculated[column].values
    diff['average'] = diff.sum(axis=1).values
    diff['mean_diff_delta'] = diff['average'].rolling(window=2) \
                           .apply(lambda x: x[1] - x[0]).values
          
            
            
###############################################################################
# Get Distribution on mean differences Delta
###############################################################################
if True:
    
    # Create Distribution for changes
    dist_values = (diff.loc[1: ,'mean_diff_delta'].values,
                   np.ones(diff.loc[1:].shape[0]) / diff.loc[1:].shape[0])
    dist = rv_discrete(values=dist_values)
    
    # Get bounding values on mean delta
    bound = .001
    bound_lower = dist.ppf(bound)
    bound_upper = dist.ppf(1 - bound)
    



###############################################################################
# Plot mean of all differences (w/ a roll) with every ratio over same interval
###############################################################################
if True:
    
    end = 16800 * 60 / 5   # 1700
    window = 4000  # 1000   don't touch, though no slope....
    roll_window = 20 # 20 
    plt.figure(figsize=(12,3))
    x = np.arange(end - window, end + 1)
    # Plot mean of all differences
    diff.loc[end - window: end].mean(axis=1).plot()
    # Plot Rolling mean of all differences
    diff.loc[end - window: end].mean(axis=1).rolling(roll_window).mean().plot()
    plt.title('Mean and Rolling mean Difference of Given and Calculated Ratios')
    plt.figure(figsize=(12,3))
    diff.loc[end - window: end, 'mean_diff_delta'].plot()
    plt.plot(x, np.ones(x.shape[0]) * bound_upper, color='black')
    plt.plot(x, np.ones(x.shape[0]) * bound_lower, color='black')
    plt.title('1 delta Change in mean difference')
    # Plot all the ratios
    plt.figure(figsize=(12,3))
    for column in ratios.columns:
        plt.plot(ratios.loc[end - window: end, column] \
                 - ratios.loc[end - window: end, column].values[0],
                 label=column)
    plt.title('Given Ratios')
    plt.legend()
    plt.show()
    # Plot all the Currencies
    plt.figure(figsize=(12,3))
    for column in cur.columns:
        plt.plot(cur.loc[end - window: end, column] \
                 - cur.loc[end - window: end, column].values[0],
                 label=column)
        plt.plot(cur.loc[end - window: end, column] \
                 - cur.loc[end - window: end, column].values[0],
                 '+')
    plt.title('Currencies')
    plt.legend()
    plt.show()
    

# Plot each given and calculated Ratio
if True:
    for column in ratios.columns:
        plt.figure(figsize=(12,3))
        plt.plot(ratios.loc[end - window: end, column], label='given')
        plt.plot(calculated.loc[end - window: end, column], label='calc')
        plt.title(column)
        plt.legend()
        plt.show()
        













'''



###############################################################################
# Get calculated ratios, given ratios and differences for two currencies
###############################################################################
if False:
    
    # Choose which currencies to work with
    cur1 = 'eur'
    cur2 = 'usd'
    # Get ratios column name
    if str(cur1) + '_' + str(cur2) in ratios.columns:
        ratios_col = str(cur1) + '_' + str(cur2)
        currency1 = cur1
        currency2 = cur2
    else:
        ratios_col = str(cur2) + '_' + str(cur1)
        currency1 = cur2
        currency2 = cur1
    # Build new DataFrame with all info that I need
    df = pd.DataFrame()
    df['given'] = ratios.loc[:, ratios_col]
    df['calculated'] = cur.loc[:, currency1].values \
                     / cur.loc[:, currency2].values
    df['cur1'] = cur.loc[:, currency1].values
    df['cur2'] = cur.loc[:, currency2].values
    df['ratio_diff'] = df.given - df.calculated



###############################################################################
# Exam difference distribution
###############################################################################
if False:
    
    #-----------------------------------------------------------------------------
    # Is the Distribution Normal ?
    #-----------------------------------------------------------------------------
    k2, p = normaltest(df.ratio_diff.values)
    alpha = 1e-3
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")
    
    
    #-----------------------------------------------------------------------------
    # Get Ratio Distribution
    #-----------------------------------------------------------------------------
    # create Distribution from data
    dist = rv_discrete(values=(df.ratio_diff.values, 
                                         np.ones(df.ratio_diff.values.shape[0])\
                                         / df.ratio_diff.values.shape[0]))    
    # Get bounding of distribution
    bound = .001
    upper = dist.ppf(1 - bound)
    lower = dist.ppf(bound)
    # Get index where values are past bounds (grouped)
    grouping_interval = 120
    upper_index = df[df.ratio_diff > upper].index.values
    lower_index = df[df.ratio_diff < lower].index.values
    upper_index = get_groups(upper_index, grouping_interval)
    lower_index = get_groups(lower_index, grouping_interval)
    # Examine Distibution of indexes
    bounds_distribution = np.zeros(df.shape[0])
    bounds_distribution[upper_index] = 1
    bounds_distribution[lower_index] = -1
    df['distribution'] = bounds_distribution
    
    
    #-----------------------------------------------------------------------------
    # Get Ratio Rolling Distribution
    #-----------------------------------------------------------------------------
    # Create rolling ratio_diff column.  Get Bounds for this as well
    rolling_window = 15
    df['ratio_rolling'] = df.ratio_diff.rolling(rolling_window).mean().values
    # Create Distribution
    dist_rolling = rv_discrete(values=(df.ratio_rolling.values, 
                                       np.ones(df.ratio_rolling.values.shape[0])\
                                       / df.ratio_rolling.values.shape[0])) 
    # Get bounding of distribution
    bound_rolling = .01
    upper_rolling = dist_rolling.ppf(1 - bound_rolling)
    lower_rolling = dist_rolling.ppf(bound_rolling)
    # Get index where values are past bounds (grouped)
    grouping_interval = rolling_window
    upper_index_rolling = df[df.ratio_rolling > upper_rolling].index.values
    lower_index_rolling = df[df.ratio_rolling < lower_rolling].index.values
    upper_index_rolling = get_groups(upper_index_rolling, grouping_interval)
    lower_index_rolling = get_groups(lower_index_rolling, grouping_interval)
    # Examine Distibution of indexes
    bounds_distribution_rolling = np.zeros(df.shape[0])
    bounds_distribution_rolling[upper_index_rolling] = 1
    bounds_distribution_rolling[lower_index_rolling] = -1
    df['distribution_rolling'] = bounds_distribution_rolling
    
    
    
    #-----------------------------------------------------------------------------
    # Plot Some Shit (but really, don't)
    #-----------------------------------------------------------------------------
    # Plot difference distriubtiomn
    # sns.distplot(df.ratio_diff, bins=100)
    # plt.plot(bounds_distribution, 'o')

    


###############################################################################
# Correlations on 'variables' in df
###############################################################################
if False:
    collection = []
    for col1 in df.columns:
        tmp = []
        for col2 in df.columns:
            corr = np.corrcoef(df.dropna()[col1].values, 
                               df.dropna()[col2].values)
            tmp.append(corr[0,1])
        collection.append(tmp)
    correlations = pd.DataFrame(collection, 
                                columns=df.columns,
                                index=df.columns)
    sns.clustermap(correlations,linecolor='black', linewidth=3)
    



###############################################################################
# Plotting examples
###############################################################################
if True:
    
    # =============================================================================
    # Plot given and calculated ratios as well as difference
    # =============================================================================
    end = 30000
    window = 780
    # Plot the given and calculated ratios
    fig, ax = plt.subplots(2, 1, figsize=(10,8))
    ax[0].plot(df.loc[end - window: end, 'given'], 
               label = 'given')
    ax[0].plot(df.loc[end - window: end, 'calculated'], 
               label = 'calculated')
    ax[0].legend()
    # Plot the difference between given and calculated
    ax[1].plot(df.loc[end - window: end, 'ratio_diff'],
               'o', markersize=3, label='ratio difference')
    ax[1].plot(df.loc[end - window: end, 'ratio_rolling'])
    ax[1].plot(np.arange(end - window, end), np.ones(window) * upper, color='black')
    ax[1].plot(np.arange(end - window, end), np.ones(window) * lower, color='black')
    ax[1].plot(np.arange(end - window, end), np.ones(window) * upper_rolling, color='green')
    ax[1].plot(np.arange(end - window, end), np.ones(window) * lower_rolling, color='green')
    ax[1].legend()
    ax[1].plot(np.arange(end - window, end), np.zeros(window), color='black')
    # Plot bounds distribution on window
    plt.tight_layout(); plt.show()
    
    
    
if False:
    fig, ax = plt.subplots(2, 1, figsize=(10,8))
    ax[0].plot(df.ratio_rolling.values)
    ax[1].plot(df.given.values)
    ax[1].plot(df.calculated.values)

'''
    
    
