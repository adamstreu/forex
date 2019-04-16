####################################################################
# Import
####################################################################
if False:
    
    # Imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import os; os.chdir('/northbend')
    from libraries.stats import get_distribution_boundary
    from libraries.transformations import get_groups
    from libraries.plotting import plot_index_distribution
    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15)
    np.warnings.filterwarnings('ignore')   
    plt.rcParams['figure.figsize'] = [14, 6]

    # Import df's 
    eur_long = pd.read_pickle('/Users/user/Desktop/6_month_steps/eur_long.pkl')
    eur_short = pd.read_pickle('/Users/user/Desktop/6_month_steps/eur_short.pkl')
    eur_down = pd.read_pickle('/Users/user/Desktop/6_month_steps/eur_down.pkl')
    eur_up = pd.read_pickle('/Users/user/Desktop/6_month_steps/eur_up.pkl')
    eur_minimums = pd.read_pickle('/Users/user/Desktop/6_month_steps/eur_minimums.pkl')
    usd_long = pd.read_pickle('/Users/user/Desktop/6_month_steps/usd_long.pkl')
    usd_short = pd.read_pickle('/Users/user/Desktop/6_month_steps/usd_short.pkl')
    usd_down = pd.read_pickle('/Users/user/Desktop/6_month_steps/usd_down.pkl')
    usd_up = pd.read_pickle('/Users/user/Desktop/6_month_steps/usd_up.pkl')
    usd_minimums = pd.read_pickle('/Users/user/Desktop/6_month_steps/usd_minimums.pkl')
    eur_usd_long = pd.read_pickle('/Users/user/Desktop/6_month_steps/eur_usd_long.pkl')
    eur_usd_short = pd.read_pickle('/Users/user/Desktop/6_month_steps/eur_usd_short.pkl')
    eur_usd_down = pd.read_pickle('/Users/user/Desktop/6_month_steps/eur_usd_down.pkl')
    eur_usd_up = pd.read_pickle('/Users/user/Desktop/6_month_steps/eur_usd_up.pkl')
    eur_usd_minimums = pd.read_pickle('/Users/user/Desktop/6_month_steps/eur_usd_minimums.pkl')
    cur = pd.read_pickle('/Users/user/Desktop/6_month_steps/cur.pkl')
    ratios = pd.read_pickle('/Users/user/Desktop/6_month_steps/ratios.pkl')
    high = pd.read_pickle('/Users/user/Desktop/6_month_steps/high.pkl')
    low = pd.read_pickle('/Users/user/Desktop/6_month_steps/low.pkl')
    volume = pd.read_pickle('/Users/user/Desktop/6_month_steps/volume.pkl')
    timestamp = pd.read_pickle('/Users/user/Desktop/6_month_steps/timestamp.pkl')
    aud = cur.loc[:, 'aud']
    cad = cur.loc[:, 'cad']
    eur = cur.loc[:, 'eur']
    gbp = cur.loc[:, 'gbp']
    nzd = cur.loc[:, 'nzd']
    usd = cur.loc[:, 'usd']
    eur_usd = ratios.loc[:, 'eur_usd']


 
####################################################################
# Notes
####################################################################
if False:
    
    '''
    Original backfill method but one at a time - 
        like i would retrive data in real life
        
    '''



####################################################################
# Definitions
####################################################################
if 1:  
    
    # Step Outcomes
    def linear_outcomes(closing_values, lookback):
        # Just some error protection
        try:
            vals = closing_values.values
        except:
            vals = closing_values
        coll = [np.nan] * lookback
        # Follow the next value up or down and see how far it gets in lookback 
        for i in range(lookback, vals.shape[0] - 1):    
            # If next values is positive
            if vals[i - (lookback - 1)] - vals[i - lookback] > 0:
                end_search = np.argmax(vals[i - lookback: i] < vals[i - lookback])
                if end_search == 0:
                    end_search = i
                value = vals[i - lookback : (i - lookback) + end_search].max()
                coll.append(value - vals[i - lookback])                                 # do i have this signed right?
            # If next values is negative
            elif vals[i - (lookback - 1)] - vals[i - lookback] < 0:
                end_search = np.argmax(vals[i - lookback: i] > vals[i - lookback])
                if end_search == 0:
                    end_search = i
                value = vals[i - lookback : (i - lookback) + end_search].min() 
                coll.append(value - vals[i - lookback])                                 # do i have this signed right?
            # If next values is nuetral (don't follow)    
            elif vals[i - (lookback - 1)] - vals[i - lookback] == 0:
                coll.append(0)
                end_search = i - lookback
                value = i - lookback
            #print(i, end_search, value, coll[-1])
        coll.append(0)
        return pd.DataFrame(np.array(coll))
    
    def linear_outcomes_singular(closing_values, lookback):
        # Just some error protection
        try:
            vals = closing_values.values
        except:
            vals = closing_values
        # See    
        if vals[1] - vals[0] > 0:
            end_search = np.argmax(vals < vals[0])
            if end_search == 0:
                end_search = vals.shape[0]
            value = vals[: end_search].max()
            distance = value - vals[0]                       # do i have this signed right?
        # If next values is negative
        elif vals[1] - vals[0] < 0:
            end_search = np.argmax(vals > vals[0])
            if end_search == 0:
                end_search = vals.shape[0]
            value = vals[: end_search].min() 
            distance = value - vals[0]                        # do i have this signed right?
        # If next values is nuetral (don't follow)    
        elif vals[1] - vals[0] == 0:
            distance = 0
        return distance
    
    def rolling_slope(df_values, slope_window):
        '''
        validation = make sure dataframe / series)
        '''
        index = np.arange(slope_window).reshape(-1, 1)
        values = df_values.values.reshape(-1, 1)
        coll = [0] * slope_window
        for i in range(slope_window, values.shape[0]):
            linreg = LinearRegression()
            linreg.fit(index, values[i - slope_window: i])
            coll.append(linreg.coef_[0][0])
        return np.array(coll)
    
    
    def main(currency, bounded_outcomes, direction, linear_lookback,
             slope_lookback, mask_threshold, boundary_threshold):
        # Get Linear Outcomes
        lin = linear_outcomes(currency, linear_lookback)
        # Get masked max or min over threshold
        if direction == 'long':
            lin_masked = lin.mask(lin > - mask_threshold)\
                        .fillna(method='backfill').fillna(0)
        else:
            lin_masked = lin.mask(lin < mask_threshold)\
                        .fillna(method='backfill').fillna(0)
        # Get rolling slope
        slopes = rolling_slope(lin_masked, slope_lookback)
        # Get bounded index.
        slope_bounds = get_distribution_boundary(slopes, boundary_threshold)
        if direction == 'long':
            index = slope_bounds['lower_index']
            threshold = slope_bounds['lower_bound']
        else:
            index = slope_bounds['upper_index']
            threshold = slope_bounds['upper_bound']
        # Test bounded index on ratio
        print(bounded_outcomes.loc[index].mean())
        return index, threshold
    

    

###############################################################################
# Call program through definitions.  Test long and short on instrument
###############################################################################
if 1:    
    
    # Windows 
    linear_lookback = 30
    slope_lookback = 30
    # thresholds
    mask_threshold = .0005
    boundary_threshold = .03

    # Currency 1 
    currency = eur
    bounded_outcomes = eur_long
    direction = 'long'


    distances = []
    

    for i in range(linear_lookback):
        

    







###############################################################################
# Go through without functions - keep all in memory.  Assemble DF.
###############################################################################  
if 0:
    
    # Windows 
    linear_lookback = 30
    slope_lookback = 30
    # thresholds
    mask_threshold = .0005
    boundary_threshold = .03
    vals = eur.values
    currency = eur
    bounded_outcomes = eur_short
    
    ''' 
    Step Outcomes
    Gets maximum signed distance over specified interval 
    (without falling back to original value)
    '''
    step_distances = [np.nan] * linear_lookback
    # Follow the next value up or down and see how far it gets in linear_lookback 
    for i in range(linear_lookback, vals.shape[0] - 1):    
        # If next values is positive
        if vals[i - (linear_lookback - 1)] - vals[i - linear_lookback] > 0:
            end_search = np.argmax(vals[i - linear_lookback: i] < vals[i - linear_lookback])
            if end_search == 0:
                end_search = i
            value = vals[i - linear_lookback : (i - linear_lookback) + end_search].max()
            step_distances.append(value - vals[i - linear_lookback])                                 # do i have this signed right?
        # If next values is negative
        elif vals[i - (linear_lookback - 1)] - vals[i - linear_lookback] < 0:
            end_search = np.argmax(vals[i - linear_lookback: i] > vals[i - linear_lookback])
            if end_search == 0:
                end_search = i
            value = vals[i - linear_lookback : (i - linear_lookback) + end_search].min() 
            step_distances.append(value - vals[i - linear_lookback])                                 # do i have this signed right?
        # If next values is nuetral (don't follow)    
        elif vals[i - (linear_lookback - 1)] - vals[i - linear_lookback] == 0:
            step_distances.append(0)
            end_search = i - linear_lookback
            value = i - linear_lookback
        #print(i, end_search, value, step_distances[-1])
    step_distances.append(0)
    step_distances = pd.DataFrame(np.array(step_distances))
    
    
    
    '''
    Mask Step Outcomes 
    (Assumes Long)  Neccesary ?   does ffill still work the same
    Because the above function more or less provides the slope at two locations,
    the mask function chooses the appropriate one.
    '''
    masked_steps = step_distances.mask(step_distances < mask_threshold)\
                                 .fillna(method='backfill').fillna(0)
                            
    
    '''
    Get Rolling Slopes of masked according to parameter
    As it says....
    This is the final value that is matched against 
    it's own distributional threshold
    '''
    
    index = np.arange(slope_lookback).reshape(-1, 1)
    values = masked_steps.values.reshape(-1, 1)
    coll = [0] * slope_lookback
    for i in range(slope_lookback, values.shape[0]):
        linreg = LinearRegression()
        linreg.fit(index, values[i - slope_lookback: i])
        coll.append(linreg.coef_[0][0])    
    slopes =  np.array(coll)
    
    
    
    '''
    The rest we don't need 
    we will already have the thresholds from backtesting - 
    we will just need to check them
    '''
    threshold = get_distribution_boundary(slopes, boundary_threshold)['upper_bound']
    
    
    
    '''
    Assemble into df for hand checking, etc.
    '''
    df = pd.DataFrame(currency).join(step_distances, lsuffix= '.')\
            .join(masked_steps, lsuffix = '.')
    df.columns = ['eur', 'steps', 'masked_steps']
    df['slopes'] = slopes
    df['outcomes'] = bounded_outcomes.iloc[:, 2]
    
    
    start = 1000
    end = 10000
    groups = get_groups(df.loc[(df.slopes > threshold)].index.values, 120)
    print('Placement Shape and Win %: \t{}\t{:.2f}'.format(df.loc[df.slopes > threshold, 'outcomes'].shape[0], df.loc[df.slopes > threshold, 'outcomes'].mean()))
    print('Group Shape and win %    : \t{}\t{:.2f}'.format(groups.shape[0], df.loc[groups, 'outcomes'].mean()))
    print('Placement Shape and Win % in first start: end: \t{}\t{:.2f}'.format(df.loc[start:end].loc[df.slopes > threshold, 'outcomes'].shape[0], df.loc[start:end].loc[df.slopes > threshold, 'outcomes'].mean()))    
    df.loc[start:end, ['steps', 'masked_steps']].plot()
    win_plot_index = df.loc[start:end].loc[(df.slopes > threshold) & (df.outcomes == True)].index.values 
    lose_plot_index = df.loc[start:end].loc[(df.slopes > threshold) & (df.outcomes == False)].index.values 
    plt.plot(win_plot_index, np.ones(win_plot_index.shape[0]) * .001, 'o', color='green')
    plt.plot(lose_plot_index, np.ones(lose_plot_index.shape[0]) * .001, 'o', color='red')
    plt.figure()
    currency.loc[start:end].plot()
    plot_index_distribution(df.loc[df.slopes > threshold].index.values, 'placement distribution')
































# =============================================================================
# Export
# =============================================================================
if False:
    
    eur_long.to_pickle('/Users/user/Desktop/6_month_steps/eur_long.pkl')
    eur_short.to_pickle('/Users/user/Desktop/6_month_steps/eur_short.pkl')
    eur_down.to_pickle('/Users/user/Desktop/6_month_steps/eur_down.pkl')
    eur_up.to_pickle('/Users/user/Desktop/6_month_steps/eur_up.pkl')
    eur_minimums.to_pickle('/Users/user/Desktop/6_month_steps/eur_minimums.pkl')

    usd_long.to_pickle('/Users/user/Desktop/6_month_steps/usd_long.pkl')
    usd_short.to_pickle('/Users/user/Desktop/6_month_steps/usd_short.pkl')
    usd_down.to_pickle('/Users/user/Desktop/6_month_steps/usd_down.pkl')
    usd_up.to_pickle('/Users/user/Desktop/6_month_steps/usd_up.pkl')
    usd_minimums.to_pickle('/Users/user/Desktop/6_month_steps/usd_minimums.pkl')

    eur_usd_long.to_pickle('/Users/user/Desktop/6_month_steps/eur_usd_long.pkl')
    eur_usd_short.to_pickle('/Users/user/Desktop/6_month_steps/eur_usd_short.pkl')
    eur_usd_down.to_pickle('/Users/user/Desktop/6_month_steps/eur_usd_down.pkl')
    eur_usd_up.to_pickle('/Users/user/Desktop/6_month_steps/eur_usd_up.pkl')
    eur_usd_minimums.to_pickle('/Users/user/Desktop/6_month_steps/eur_usd_minimums.pkl')

    cur.to_pickle('/Users/user/Desktop/6_month_steps/cur.pkl')
    ratios.to_pickle('/Users/user/Desktop/6_month_steps/ratios.pkl')
    high.to_pickle('/Users/user/Desktop/6_month_steps/high.pkl')
    low.to_pickle('/Users/user/Desktop/6_month_steps/low.pkl')
    volume.to_pickle('/Users/user/Desktop/6_month_steps/volume.pkl')
    timestamp.to_pickle('/Users/user/Desktop/6_month_steps/timestamp.pkl')



    
    
    
    
    








'''
    def linear_outcomes_singular(closing_values, lookback):
        # Just some error protection
        try:
            vals = closing_values.values
        except:
            vals = closing_values
        # See    
        if vals[1] - vals[0] > 0:
            end_search = np.argmax(vals < vals[0])
            if end_search == 0:
                end_search = vals.shape[0]
            value = vals[: end_search].max()
            distance = value - vals[0]                       # do i have this signed right?
        # If next values is negative
        elif vals[1] - vals[0] < 0:
            end_search = np.argmax(vals > vals[0])
            if end_search == 0:
                end_search = vals.shape[0]
            value = vals[: end_search].min() 
            distance = value - vals[0]                        # do i have this signed right?
        # If next values is nuetral (don't follow)    
        elif vals[1] - vals[0] == 0:
            distance = 0
        return distance
'''