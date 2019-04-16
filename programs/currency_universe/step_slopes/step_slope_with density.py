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
    Ok.
    Have to figure out a way around the backfill.
    more nmotes later.....
    
    This would be comsidered 
    
    WORK # 2.
    
    
    Got linear outcomes on _steps_
    Got rolling slope of (for example, ) of steps/
    got boundary on steps 
    took all locations greater than bound
    Plugged into currenecy outcomes.
    Excellent reuslts
    
    
    
    make sure ratios line up with calculated
    If we can't go through backfill we are fucked.
    It's the backfill on which it all depends now.
    
    ok.  even if we don't have a good backfill - we at least have a graph of 
    the wins and what we are after.  Perhaps there is another way.
    
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
    
    
    def rolling_slope_old(df_values, slope_window):
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
    
    
    def rolling_slope(df_values, slope_window):
        values = df_values.values.reshape(-1, 1)
        slopes = [0] * slope_window
        density = [0] * slope_window
        distribution = [0] * slope_window             
        for i in range(slope_window, values.shape[0]):
            vals = values[i - slope_window: i]
            non_nan_index = ~np.isnan(vals)
            vals_for_lin = vals[~np.isnan(vals)].reshape(-1, 1)
            density.append(vals_for_lin.shape[0])
            if vals_for_lin.shape[0] == 0:
                # Just append zeros if no values found
                slopes.append(0)    
                distribution.append(0)
            else: 
                # Get Slope
                index = np.arange(vals.shape[0]).reshape(-1, 1)[non_nan_index].reshape(-1,1)
                linreg = LinearRegression()
                linreg.fit(index, vals_for_lin)
                slopes.append(linreg.coef_[0][0])   
                # Calculate distribution
                hist = np.histogram(index.ravel(), bins =[0, dist1, dist2, dist3, slope_window])
                distribution.append(np.argmax(hist[0]) + 1)
        # Return Array dictionary        
        slopes =  np.array(slopes)
        distribution = np.array(distribution)
        return {'slopes': slopes,
                'density': density,
                'distribution': distribution}
    
    
    def main(currency, bounded_outcomes, direction, linear_lookback,
             slope_lookback, mask_threshold, boundary_threshold):
        # Get Linear Outcomes
        step_distances = linear_outcomes(currency, linear_lookback)
        # Get masked max or min over threshold
        if direction == 'long':
            masked_steps = step_distances.mask(step_distances > - mask_threshold)
        else:
            masked_steps = step_distances.mask(step_distances < mask_threshold)
        # Get rolling slope
        return_slopes = rolling_slope(masked_steps, slope_lookback)
        slopes = return_slopes['slopes']
        distribution = return_slopes['distribution']
        density = return_slopes['density']
        # Get bounded index.
        slope_bounds = get_distribution_boundary(slopes, boundary_threshold)
        if direction == 'long':
            index = slope_bounds['lower_index']
            threshold = slope_bounds['lower_bound']
        else:
            index = slope_bounds['upper_index']
            threshold = slope_bounds['upper_bound']
        # Create DF
        df = pd.DataFrame(currency).join(step_distances, lsuffix= '.')\
                .join(masked_steps, lsuffix = '.')
        df.columns = ['eur', 'steps', 'masked_steps']
        df['slopes'] = slopes
        df['density'] = density
        df['distribution'] = distribution
        # Test bounded index on ratio
        print(bounded_outcomes.loc[index].mean())
        return index, threshold, df
        
    

###############################################################################
# Call program through definitions.  Test long and short on instrument
###############################################################################
if 0:    
    
    # Windows 
    linear_lookback = 30
    slope_lookback  = 30
    # thresholds
    mask_threshold     = .0005
    boundary_threshold = .03
    # Distribution calculations
    dist1 = int(slope_lookback / 4)
    dist2 = dist1 * 2
    dist3 = dist1 * 3

    # Currency 1 
    currency = eur
    bounded_outcomes = eur_long
    direction = 'long'
    # Call all functions (main) (and print results)
    index_1, threshold_1, df_1 = main(currency,
                                      bounded_outcomes,
                                      direction,
                                      linear_lookback,                           
                                      slope_lookback,                            
                                      mask_threshold,
                                      boundary_threshold)
    
    
    # Currency 2
    currency = eur
    bounded_outcomes = eur_short
    direction = 'short'
    # Call all functions (main) (and print results)
    index_2, threshold_2, df_2 = main(currency,
                                      bounded_outcomes,
                                      direction,
                                      linear_lookback,
                                      slope_lookback,
                                      mask_threshold,
                                      boundary_threshold)


    # Measure Combined Results on Instrument
    instrument = eur_usd
    combined_outcomes = eur_usd_long
    # How did we do ? 
    combined_indexes = np.intersect1d(index_1, index_2)
    groups = get_groups(combined_indexes, 120)
    print('Placements: {}'.format(combined_indexes.shape[0]))
    print('groups: {}'.format(groups.shape[0]))
    print('\n Winnings from all placements\n----------------------------')
    print(combined_outcomes.loc[combined_indexes].mean())
    print('\n Winnings from groups\n----------------------------')
    print(combined_outcomes.loc[groups].mean())
    # Plot distribution
    plot_index_distribution(combined_indexes, 'combined')
    plot_index_distribution(groups, 'groups')
    # Plot instrument



###############################################################################
# Go through without functions - keep all in memory.  Assemble DF.
###############################################################################  
if 1:
    
    # Windows 
    linear_lookback = 30
    slope_lookback = 30
    # thresholds
    mask_threshold = .0005
    boundary_threshold = .03
    # Distribution calculations
    dist1 = int(slope_lookback / 4)
    dist2 = dist1 * 2
    dist3 = dist1 * 3
    
    # Currency 1 
    currency = eur
    vals = currency.values    
    bounded_outcomes = eur_short
    direction = 'short'
    target_column = 2

    
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
    Try to follow slope of top or bottom of step distribution
    '''
    if direction == 'long':
        masked_steps = step_distances.mask(step_distances > - mask_threshold)
    else:
        masked_steps = step_distances.mask(step_distances < mask_threshold)
    
    
    '''
    Get Rolling Slopes of masked according to parameter
    As it says....
    This is the final value that is matched against 
    it's own distributional threshold
    '''
    values = masked_steps.values.reshape(-1, 1)
    slopes = [0] * slope_lookback
    density = [0] * slope_lookback
    distribution = [0] * slope_lookback             #  numbers in First second or third (or none) of vals 
    for i in range(slope_lookback, values.shape[0]):
        vals = values[i - slope_lookback: i]
        non_nan_index = ~np.isnan(vals)
        vals_for_lin = vals[~np.isnan(vals)].reshape(-1, 1)
        density.append(vals_for_lin.shape[0])
        if vals_for_lin.shape[0] == 0:
            slopes.append(0)    
            distribution.append(0)
        else: 
            index = np.arange(vals.shape[0]).reshape(-1, 1)[non_nan_index].reshape(-1,1)
            linreg = LinearRegression()
            linreg.fit(index, vals_for_lin)
            slopes.append(linreg.coef_[0][0])   
            
            # Calculate distribution
            hist = np.histogram(index.ravel(), bins =[0, dist1, dist2, dist3, slope_lookback])
            distribution.append(np.argmax(hist[0]) + 1)
    slopes =  np.array(slopes)

        
    
    '''
    The rest we don't need 
    we will already have the thresholds from backtesting - 
    we will just need to check them
    '''
    if direction == 'long':
        threshold = get_distribution_boundary(slopes, boundary_threshold)['lower_bound']
        thresh = get_distribution_boundary(df.slopes[df.slopes > 0], boundary_threshold)
        threshold = get_distribution_boundary(df.slopes[df.slopes > 0], boundary_threshold)['lower_bound']
    else:
    
        threshold = get_distribution_boundary(slopes, boundary_threshold)['upper_bound']
        thresh = get_distribution_boundary(df.slopes[df.slopes <  0], boundary_threshold)
        threshold = get_distribution_boundary(df.slopes[df.slopes < 0], boundary_threshold)['lower_bound']    
    
    
    '''
    Assemble into df for hand checking, etc.
    '''
    df = pd.DataFrame(currency).join(step_distances, lsuffix= '.')\
            .join(masked_steps, lsuffix = '.')
    df.columns = ['eur', 'steps', 'masked_steps']
    df['slopes'] = slopes
    df['outcomes'] = bounded_outcomes.iloc[:, target_column]
    df['density'] = density
    df['distribution'] = distribution
    df['outcomes_opp'] = eur_long.iloc[:, target_column]
    
    
    
    if direction == 'long':

        start = 1000
        end = 10000
        groups = get_groups(df.loc[(df.slopes < threshold)].index.values, 120)
        print('Threshold: {}'.format(threshold))
        print('Placement Shape and Win %: \t{}\t{:.2f}'.format(df.loc[df.slopes < threshold, 'outcomes'].shape[0], df.loc[df.slopes < threshold, 'outcomes'].mean()))
        print('Group Shape and win %    : \t{}\t{:.2f}'.format(groups.shape[0], df.loc[groups, 'outcomes'].mean()))
        print('Placement Shape and Win % in first start: end: \t{}\t{:.2f}'.format(df.loc[start:end].loc[df.slopes < threshold, 'outcomes'].shape[0], df.loc[start:end].loc[df.slopes < threshold, 'outcomes'].mean()))    
        df.loc[start:end, ['steps', 'masked_steps']].plot()
        win_plot_index = df.loc[start:end].loc[(df.slopes < threshold) & (df.outcomes == True)].index.values 
        lose_plot_index = df.loc[start:end].loc[(df.slopes < threshold) & (df.outcomes == False)].index.values 
        plt.plot(win_plot_index, np.ones(win_plot_index.shape[0]) * .001, 'o', color='green')
        plt.plot(lose_plot_index, np.ones(lose_plot_index.shape[0]) * .001, 'o', color='red')
        plt.figure()
        currency.loc[start:end].plot()
        plot_index_distribution(df.loc[df.slopes < threshold].index.values, 'placement distribution')
        
        
        print('\nTrheshold and Density  \n'
              'i:  Placements,  Win % \n'
              '-----------------------')
        for i in range(1, df.density.max()):
            new_df = df.loc[(df.slopes < threshold) & (df.density > i), 'outcomes']
            print('{}:\t{}\t{:.2f}'.format(i, new_df.shape[0], new_df.mean()))
            
        print('\nTheshold and Distribution  \n'
              'i:  Placements,  Win % \n'
              '-----------------------')
        for i in range(df.distribution.max() + 1):
            new_df = df.loc[(df.slopes < threshold) & (df.distribution == i), 'outcomes']
            print('{}:\t{}\t{:.2f}'.format(i, new_df.shape[0], new_df.mean()))
                

        
    else:
            
        start = 1000
        end = 10000
        groups = get_groups(df.loc[(df.slopes > threshold)].index.values, 120)
        print('Threshold: {}'.format(threshold))
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
        
        
        print('\nTrheshold and Density  \n'
              'i:  Placements,  Win % \n'
              '-----------------------')
        for i in range(1, df.density.max()):
            new_df = df.loc[(df.slopes < threshold) & (df.density > i), 'outcomes']
            print('{}:\t{}\t{:.2f}'.format(i, new_df.shape[0], new_df.mean()))
            
        print('\nTheshold and Distribution  \n'
              'i:  Placements,  Win % \n'
              '-----------------------')
        for i in range(df.distribution.max() + 1):
            new_df = df.loc[(df.slopes < threshold) & (df.distribution == i), 'outcomes']
            print('{}:\t{}\t{:.2f}'.format(i, new_df.shape[0], new_df.mean()))
                

        plt.figure()
        df.loc[:, ['steps', 'masked_steps']].plot();
        (currency - currency.values[0]).plot()
        plt.plot(np.zeros(currency.shape[0]), color='grey')


        plt.figure()
        df.loc[:, ['steps', 'masked_steps']].plot();
        (currency - currency.values[0]).plot()
        plt.plot(np.zeros(currency.shape[0]), color='grey')
        plt.plot(np.ones(currency.shape[0]) * threshold, color='grey')
    





























"""
###############################################################################
# Go through without functions - keep all in memory.  Assemble DF.
###############################################################################  
if 1:


        
    ''' 
    Step Outcomes
    Gets maximum signed distance over specified interval 
    (without falling back to original value)
    '''
    coll = [np.nan] * linear_lookback
    # Follow the next value up or down and see how far it gets in linear_lookback 
    for i in range(linear_lookback, vals.shape[0] - 1):    
        # If next values is positive
        if vals[i - (linear_lookback - 1)] - vals[i - linear_lookback] > 0:
            end_search = np.argmax(vals[i - linear_lookback: i] < vals[i - linear_lookback])
            if end_search == 0:
                end_search = i
            value = vals[i - linear_lookback : (i - linear_lookback) + end_search].max()
            coll.append(value - vals[i - linear_lookback])                                 # do i have this signed right?
        # If next values is negative
        elif vals[i - (linear_lookback - 1)] - vals[i - linear_lookback] < 0:
            end_search = np.argmax(vals[i - linear_lookback: i] > vals[i - linear_lookback])
            if end_search == 0:
                end_search = i
            value = vals[i - linear_lookback : (i - linear_lookback) + end_search].min() 
            coll.append(value - vals[i - linear_lookback])                                 # do i have this signed right?
        # If next values is nuetral (don't follow)    
        elif vals[i - (linear_lookback - 1)] - vals[i - linear_lookback] == 0:
            coll.append(0)
            end_search = i - linear_lookback
            value = i - linear_lookback
        #print(i, end_search, value, coll[-1])
    coll.append(0)
    step_distances = pd.DataFrame(np.array(coll))
    
    
    
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
    eur.loc[start:end].plot()
    plot_index_distribution(df.loc[df.slopes > threshold].index.values, 'placement distribution')



"""



