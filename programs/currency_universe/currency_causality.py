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
    Can we find a spike in one (etc.) currency that causes the others to 
        adjust in predictable ways?
    For small granularity.


Questions:
    How to relate changes ot rest of universe     
    
TO DO:
    inspect visually for patterns.
    Compare with correlation.
    Compare with local behavior for each currency.
    any correlatio, to calcualted differences?
    Correlation to volume?
    double check alone
    

Problems:
    Found bounds on all currency changes - this might belong to one
        currency more than others
    
Notes:
    all right.
    
    So.
    
        

'''

###############################################################################
# Get instruments.  Call currency Matrix.
###############################################################################
if 0:

    # Get currency universe
    granularity = 'M1'
    _from = '2018-01-01T00:00:00Z'
    _to   = '2018-04-01T00:00:00Z'
    currency_dictionary = get_currencies(granularity, _from, _to)
    cur = currency_dictionary['currencies']
    curdiff = currency_dictionary['currencies_delta']
    ratios = currency_dictionary['ratios']
        
    # Create Diff DataFrame for currencies
    cur_diff = pd.DataFrame()
    for column in cur.columns:
        roll = cur[column].rolling(window=2) \
                           .apply(lambda x: (x[1] - x[0])).values
        cur_diff[column] = roll



###############################################################################
# Get Distribution on mean differences Delta (as percentage points)
# Get Indexes where change is outside bounds of dist bound
###############################################################################
if 1:
    
    # Get Distribution
    dist_values = cur_diff.values.ravel()
    dist_probs = np.ones(cur_diff.values.ravel().shape[0]) \
               / cur_diff.values.ravel().shape[0]
    dist = rv_discrete(values=(dist_values, dist_probs))
    
    # Get bounding values on mean delta
    bound = .001
    bound_lower = dist.ppf(bound)
    bound_upper = dist.ppf(1 - bound)

    # Greate indexes for each currency where delta is above / below bound
    upper_indexes = {}
    for column in cur_diff.columns:
        upper_indexes[column] = cur_diff.loc[1:].loc[cur_diff[column] > bound_upper, column].index.values
    lower_indexes = {}
    for column in cur_diff.columns:
        lower_indexes[column] = cur_diff.loc[1:].loc[cur_diff[column] < bound_lower, column].index.values
    combined_indexes = {}
    for k in upper_indexes.keys():
        combined_indexes[k] = np.array(list(set(upper_indexes[k].tolist() +  
                                  lower_indexes[k].tolist())))
        
    # Get Indexes where bounding change happend on one currency but no other
    upper_alone = {}
    lower_alone = {}
    for k in combined_indexes.keys():
        all_others = []
        # Combine all indexes that are not main index into set list
        for j in combined_indexes.keys():
            if str(k) != str(j):
                all_others += combined_indexes[j].tolist()
        all_others = list(set(all_others))
        # if upper_alone is not in combined, keep
        alone = []
        for index in upper_indexes[k]:
            if index not in all_others:
                alone.append(index)
        upper_alone[k] = np.array(alone)
        # if lower_alone is not in combined, keep
        alone = []
        for index in lower_indexes[k]:
            if index not in all_others:
                alone.append(index)
        lower_alone[k] = np.array(alone)
        
        
        
###############################################################################
# Plot some shit based on alone values - see what there is to see
###############################################################################
if 1:
    
    middle = 35147
    window = 100
    end = middle + int(window / 2)
    fig, ax = plt.subplots(2, 1, figsize=(10,8), sharex=True)
    for currency in cur.columns:
        ax[0].plot(cur.loc[end - window: end, currency] \
                 - cur.loc[end - window: end, currency].values[0],
                 label=currency)
        ax[0].plot(cur.loc[end - window: end, currency] \
                 - cur.loc[end - window: end, currency].values[0],
                  '+', color='lightgrey')
    ax[0].plot(end - int(window/2), 0, '|', markersize=400, color='black')
    ax[0].legend()
    ax[0].set_title('Currencies')
    # Plot sum of all currencies
    ax[1].plot(cur.loc[end - window: end].sum(axis=1))
    ax[1].plot(end - int(window/2), 1, '|', markersize=400, color='black')
    ax[1].set_title('Sum of Currencies')
    # Show plot
    plt.legend()
    plt.tight_layout()
    plt.show()



'''

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
    
    
