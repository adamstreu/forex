'''

GOAL ==========================================================================
    
    Looking at One currency at a time
    Create Strategies:
        Score correlation / prediction based on a (set) of indicators
        Predict Currency Movement with above 60% success based on indicators


This Module ===================================================================

    Prep:
        Call currency Matrix 
        Set Parameters and Currency.
        Call indicators.
    Results:
        Assemble data into df as possible. 
        Calculate more into df as needed.
    Analysis:
        Explore and score relations between indicators
    Startegies:
        Create various Strategies
        Attempt to score predictions and or correlations.
    Plot:
        Plot all indicators together for one analysis
        Plot strategies as they are created.

    
NOTES =========================================================================
    
    position wave mean on both touching -2 is not a bad indication of
        currency going jup
    Check out the correlation between the mean position wave and currency
    it appear that the rolling/  flattened diff wave, when corr i low, 
        indicates a spike or return    
    don't nmeed to find 80% predictive power for any ind on any curr....
    Channel mean - bollinger mean makes a nice wave.
        Can we get just a little predictive power out of this new indicator
        Can we match wave parameters between currencies for predictve power
    My currency universe does not return high mid low correctly.
        Indication that I either need to:
            revisit my function  or
            explore the diff.
    
    
QUESTIONS =====================================================================
    
    Pos mean is diff from channel mid.  Why?
    How to choose right window.  How to combine with granularity.
    ind(cur1) - ind(cur2) = ind(cur1 / cur2)
        
        
Basic Strategies - ============================================================
    
    Volume correlations:
    Zeros variance for all then slope:
    Position variance low and mean near c0
    diff between high, close, low.
    Slope correlation 
    basic ml on 'all' indicators
        if we can get a score of 60 on every prediction
    if slope is in direction - will slope continue to be
    steering with position slope, std.  
    Channel position filtered on 3.5 or 4
        This is 
        can we try rolling rank on channel position?

    Scoring:
        Score Indicator (window) by bin membership
        CAN BE SCORED INDEPENDANTLY
        Want indicators ensemble to be uncorrelated
        Want bars to match
        scores can be summed as conditional probabilities
        score each 
        

TO DO:

    Scoring:
        
        typical binned dist index on long
        
        Summed correlations.
            Do summing of binned prob on windows 
            within one indicator have an corr with a higher win ratio

        Do what we are just doing (slope , pos, etc)
        just with ratios......
    
    
    
LATER:    
    
    hml:
        notes in strategy

    Outcomes:
        Relative Targets
        
    Currencies:
        Get all needed in first run


'''      


# =============================================================================
# Imports
# =============================================================================
if 0:
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from multiprocessing import Process
    from multiprocessing import Manager
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report as cr
    import os; os.chdir('/northbend')
    from libraries.plotting import plot_index_distribution
    from libraries.outcomes import get_outcomes_multi
    from libraries.outcomes import plot_outcomes
    from libraries.stats import get_distribution_boundary
    from libraries.currency_universe import get_currencies
    # Multiprocessing wrapped
    from libraries.waves import waves_wrapper
    from libraries.waves import get_rolling_rank
    # Multiprocessing unwrapped
    from libraries.waves import get_rolling_mean_pos_std
    from libraries.waves import get_channel_mean_pos_std
    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15)
    np.warnings.filterwarnings('ignore')   

    

# =============================================================================
# Call currency universe.  Multiprocessing Units.
# =============================================================================
if 0:  
    
    # Need to use because of bug in response / oanda / mutli
    ''' env no_proxy = '*' ipython '''
   
    # Parameters    
    granularity = 'M1'
    _from = '2015-01-01T00:00:00Z'
    _to   = '2018-01-01T00:00:00Z'

    # Call Currency Matrix with different Columns
    def get_universe(arg, procnum, return_dict):
        print('Get Universe {}'.format(procnum))
        print('Get Universe {}'.format(arg))
        currency_dictionary = get_currencies(granularity, _from, _to, arg)
        return_dict[str(arg)] = currency_dictionary    

    # Parameters and Mutliprocessing setup
    args = ['midclose', 'midlow', 'midhigh', 'volume']
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    # Call processes, join, wait to complete, etc.
    for i in range(4):
        p = Process(target=get_universe, args=(args[i], i, return_dict))
        p.start()
        jobs.append(p)
    for job in jobs:
        job.join()

    # Relabel Values from Multiprocessing Return Dict.
    print('\nAssembling dictionaries\n')
    cur     = return_dict['midclose']['currencies']
    curdiff = return_dict['midclose']['currencies_delta']
    ratios  = return_dict['midclose']['ratios']
    cur_set = return_dict['midclose']['currency_set']
    high    = return_dict['midhigh']['currencies']
    low     = return_dict['midlow']['currencies']
    volume  = return_dict['volume']['currencies']
    volume_difference = return_dict['volume']['currencies_delta'] 
    timestamp = return_dict['volume']['timestamps'] 



# =============================================================================
# Pick Currency and Parameters.  
# =============================================================================
if 0:    
    
    # Set Currency and Volume and ratios to use.
    currency = 'eur'
    ratio_use = 'eur_usd'
    c = cur.loc[:, currency]
    l = low.loc[:, currency]
    h = high.loc[:, currency]
    c_delta = curdiff.loc[:, currency]
    vol      = volume.loc[:, currency]
    vol_delta = volume_difference.loc[:, currency]
    ratio = ratios.loc[:, ratio_use]



# =============================================================================
# Outcomes.  Get em.
# =============================================================================
if 0:
    
    # Get Outcomes
    step = .0001
    target_max = .0012# .00081 # c.std() / 2
    targets = np.arange(step, target_max, step).round(6)
    targets = np.linspace(.0009, target_max , 4).round(6)
    outcomes = get_outcomes_multi(c.values, targets, 10000)
    up = outcomes['up']
    down = outcomes['down']
    # Build Results DataFrame
    long = up < down
    short = down < up
    minimums = pd.DataFrame(np.minimum(up.values, down.values), 
                                       columns = up.columns, index = up.index)
    
    # Some plotting verification for outcomes
    if 0:
        location = 12000
        plot_outcomes(location, c, up, down, targets) 
        
        

# =============================================================================
# Construct Indicators 
# =============================================================================
if 0:
     
    # Parameters 
    windows = np.array([1500, 2000, 2250, 2500])
    position_rank_window = 240
    hml_measure = c - h

    # Rolling mean, pos, std 
    rolling_stats = get_rolling_mean_pos_std(c, windows)
    rolling_std   = rolling_stats['std']
    rolling_pos   = rolling_stats['pos']
    rolling_mean  = rolling_stats['mean']
    # Channel mean, pos, std and Slope
    channel_stats = get_channel_mean_pos_std(c.values, windows)
    channel_std   = channel_stats['std']
    channel_pos   = channel_stats['pos']
    channel_mean  = channel_stats['mean']
    slopes        = channel_stats['slope']
    # Rolling Channel Difference
    roll_chann_diff_mean =  rolling_mean - channel_mean
    roll_chann_diff_pos  =  rolling_pos - channel_pos
    roll_chann_diff_std  =  rolling_std - channel_std
    # Hml Rolling rank
    hml_rolling_rank = waves_wrapper(hml_measure, windows, get_rolling_rank)
    # Volume    
    volume_adjusted = ((c_delta / vol).cumsum())
    

    
# =============================================================================
# General Analysis on Indicators.  Build Indicators Df.  
# =============================================================================
if 0:
    
    '''
    Notes:
        
        Rolling and channel pos at high windows are corr at ~ .55 
    
    
    '''



# =============================================================================
# Strategy: Score Indicator (window) by bin membership and bin outcomes mean
# =============================================================================
if 1:
    
    '''
    OK...so, withuog double triple quandruple checking or understantind 
        (cause I've had two drinks....)
        
        ok, so - the summed results are just fine.  Just ok
        In fact - they are similar t the results based n any ne.
        I never checked, but at thes level they are ceertainly almost all the smae.
        
    Now - It is time to sum the sums on  different indicators.
        
        
    The graph showing me (IT;S' ONLY FOR SLOPE !11!!!!!) sum of binned 
    wining percentage is aboslutely, 100% amazingly coorelated to win perc.
    
    In other wrds - it seems aweeomse.
    
    
    We need enough bins for granularity but also noo few for statistical resutls
    
    ABSOLUTELY FANTIC SEEMING RESULTS FOR SLOPE, POS DIFF........
    
    currency = 'eur'
    window = array([2000, 2250, 2500, 2750])
    targets = array([0.0009, 0.001 , 0.0011, 0.0012])
    SLOPES and POSITION DIFFERENCE:
        results are tryueally fantastic.
        channel_pos are not so much - they MIGHT be ok though
        

    For this level, all targets have same outcomes..... :()
    
    hml rolling rank is discrete and will need to be handled differently
    
    
    
    If resutls are this good should also try applying directly to ratios.
    

        
    
    TO DO NEXT:
        get distr n excellent slopes results.
        
        
    '''
    
    
    
    # ================= Put each slope in bin.  Put each  =====================

    results = pd.DataFrame()
    dfs = [(slopes, 'slopes'),
           (roll_chann_diff_pos, 'pos_difference'),
           (rolling_pos, 'rolling_pos'),
           (channel_std, 'channel_std'),
           (channel_pos, 'channel_pos')]
    for df in dfs:
        target = .001
        count = 0 
        ind = df[0].copy()
        ind_prob = ind.copy()
        bins = 60
        for window in windows:
            vals = ind.loc[window:, window].values
            hist = np.histogram(vals, bins = bins)
            probs = hist[0] / vals.shape[0]
            for i in range(bins):
                start = hist[1][i]
                end   = hist[1][i + 1]
#                index = ind.loc[ind.loc[:, window].apply(lambda x:  # THIS ONE DOES NOT WORK
#                        x >= start).values, window].index.values                
                index = ind.loc[ind.loc[:, window].apply(lambda x: 
                        x >= start and x <= end).values, window].index.values
                outs = long.loc[index, target].mean()
                ind_prob.loc[index, window] = outs
                # Double chekc total indexes changed = should equal
                count += index.shape[0]
        # Results
        results[df[1]] = ind_prob.sum(axis = 1)

                
    # So now we take the combined and do it for that...
    # So now we should have a slopes probability index.
    # Lets take a sum of the probabilities 
    # Lets then bin those and get the same probability measure on those.
    bins = 60
    ind_sum = results.sum(axis=1)# ind_prob.sum(axis=1)
    hist = np.histogram(ind_sum.loc[3000:], bins=bins)
    final_bin_probs = []
    indexes = []
    count = 0
    for i in range(bins):
        start = hist[1][i]
        end   = hist[1][i + 1]
        # Jsut use larger than results
        #index = ind_sum.loc[ind_sum.apply(lambda x: x >= start and x <= end)].index.values
        index = ind_sum.loc[ind_sum.apply(lambda x: x >= start)].index.values
        outs = long.loc[index, target].mean()
        final_bin_probs.append(outs)
        count += index.shape[0]
        indexes.append(index)
    plt.plot(final_bin_probs)
    for each in list(zip(hist[0], [len(x) for x in indexes], final_bin_probs)):
        print(each)
    final_bin_probs = np.array(final_bin_probs)
            
        
    
    


    '''
    # Experiment - do for a bunch of indicators........
    for inds in [(slopes, 'slopes'), 
                 (channel_pos, 'channel_pos'),
                 (channel_std, 'channel_std'),
                 (channel_mean,'channel_mean'),
                 (rolling_pos, 'rolling_pos'),
                 (rolling_std, 'rolling_std'),
                 (roll_chann_diff_pos, 'pos diff')]:
        inds = eur_slopes
        plt.figure()
        for window in windows:
            # Do first for one window
            ind = inds[0]
            bins = 60
            target = .0009
            vals = ind.loc[window:, window].values
            hist = np.histogram(vals, bins = bins)
            probs = hist[0] / vals.shape[0]
            start = hist[1][:-1]
            end = hist[1][1:]
            
            coll = []
            for i in range(bins - 1):
                gt = vals > hist[1][i]
                lt = vals < hist[1][i + 1]
                index = gt & lt
                outs = short.loc[index, target]
                coll.append([i, index.sum(), outs.mean()])
            for each in coll:
                # print(each)
            coll = np.array(coll)
            plt.plot(coll[:, 2], label=window)
        plt.title(inds[1])
        plt.xlabel('bins')
        plt.ylabel('prob of outcome on bin')
        plt.legend()
    '''
        
        
    






if 0:
    
    '''
    Simply getting a percent frequency measure of filtering all outcomes 
    on channel position greater than _x_.
    'Longer' windows (than what I've been using)are better.
    Allows us to just brush 65% win % on 4000 placements in 3 years.
    Best ex: window = 2500, target = .0009, placements = 4200, win% = .65
              
    
    Slopes and channel position at these windows (for all four windows current),
        have a corrcoef or -.22
        
    Intersection of slope and channel pos is creating high results:
        72% with 863 placements in 3 years. 
        
    First inspection on combined grouping - not great.  All in a row with many gaps


    Slope Filter of 6 or higher has excellent results - not sure ondistribution though
    
    
    
    
    
    
    to do:
        at some point will need to go through all filters, window, target, etc.
        inspect thier shape and distributions and pick
        
        Score each distribution, shape, etc.
        
        will have to get bars
        
    
    '''
    

    # plot distibution of one intersection

    
    # =============== Plot ====================================================
    
    # Channel Position
    plt.figure()
    chan_coll = []
    _filter = 4
    pos_index = []
    print(' Window, Target, Shape, Win %')
    for window in windows:
        for target in targets:
            index = channel_pos[channel_pos.loc[:, window] > _filter].index.values
            outs = long.loc[index, target].mean()
            shape = index.shape[0]
            print('\t{} - {}: \t{}\t{:.2f}'.format(window, target, shape, outs))
            chan_coll.append(outs)
            pos_index.append(index)

    # Slopes
    slope_coll = []
    slope_index = []
    _filter = 4
    print(' Window, Target, Shape, Win %')
    for window in windows:
        for target in targets:
            index = channel_pos[channel_pos.loc[:, window] > _filter].index.values
            outs = long.loc[index, target].mean()
            shape = index.shape[0]
            print('\t{} - {}: \t{}\t{:.2f}'.format(window, target, shape, outs))
            slope_coll.append(outs)
            slope_index.append(index)
    
    # Create Filtered df
    f_ind = pd.DataFrame()
    f_ind['targets'] = np.repeat(windows,windows.shape[0])
    f_ind['windows'] = np.tile(targets, targets.shape[0])
    f_ind['position'] = pos_index
    f_ind['slope'] = slope_index
    
    # Get Intersection df
    intersections = []
    int_index = []
    for i in f_ind.index:
        for j in f_ind.index:
            intersect = np.intersect1d(f_ind.loc[i, 'position'], 
                                       f_ind.loc[j, 'slope'])
            shape = intersect.shape[0]
            outs = long.loc[intersect].mean().values
            intersections.append([i, j] + [shape] + outs.tolist())
            int_index.append(intersect)
            print('\t{} - {}: \t\t{}\t{:.2f}'.format(i, j, shape, outs.mean()))
    columns = ['position', 'slope', 'shape'] + targets.tolist()
    f_int = pd.DataFrame(intersections, columns=columns)
    int_index = pd.Series(int_index)
   
    # Filtered COmbined Expected Value ( wweighting losses at 1.1)
    f_ex = f_int.iloc[:, 3:].apply(lambda x: x - 1.2*(1-x))
    tile_shape = np.tile(f_int['shape'].values.reshape(-1, 1), (4))
    f_ex = f_ex * tile_shape
    
    
    # =============== Plot ====================================================
    
    # Plot Window Target Winning Means
    plt.plot(chan_coll, label='channel_pos')
    plt.plot(slope_coll, label = 'slope')    
    plt.title('Window / Target Winning Percentages')
    plt.legend()
    plt.show()

    # Plot intersections
    plt.figure()
    f_int.iloc[:, 3:].plot()
    plt.title('Slope and Chanel Posiiton Intersection')
    
    # Plot Channel_pos
    index_to_plot = pos_index[4]
    plot_index_distribution(index_to_plot, 'Channel_pos')
    # Plot Slope
    index_to_plot = slope_index[4]
    plot_index_distribution(index_to_plot, 'Slope')    
    # Plot Intersections
    index_to_plot = int_index.loc[12]
    plot_index_distribution(index_to_plot, 'intersection')    
    
    # Plot expected values shape
    plt.figure()
    f_ex.plot()
    plt.title('Expected Values of combined filters')
    plt.show()
    
    
    
    
    '''
    Need to combine and process:
        window,
        target,
        filter,
        
    Then for each index_combination of above


    or do i just simply score each binned filter position and leave it at that
    
    
    OK - DOES A SCORE ON EACH FILTER COMBINE SOMEWHAT INTUITIVELY OR NO?
        in other words - does a better combined score corrrelate with a 
            higher winning percentage.  This is calculable
            
        steps:
            score bins by frequency
            
            score combination of each placemtne
            
            get location of placemtn
            
            bin combined scores
            
            get outcomes on binss...
        
    
    
    
    
    
    '''







    
















# =============================================================================
# Strategy: High mid low currency rolling rank
# =============================================================================

if 0:
    
    ''' 
    To DO:
            
        do freuency of win from rank humber
    
        Score he rolling positions seperetly then combine

        looop through to analyze expected return on all combinations of
            target, 
            window, 
            rolling, 
            etc.
            pos rank (on diff windowss ? )
        
        Then call a large df and see how we do.
        
        So - for combined - we might have to accept some loss in the perc win
            on each in order to get enough values where we still have enough 
            remaiing after we look for overlaps between the two (all)
            currencies
            
        Strategy:    
            If hml results can be (filtered ?) above 65% over a year atleast:
                if outcomes are checked:
                    if calculations are checked:
                        Switch to mulyi currency analysis on strategy.
                        do for all currencies and get results for ratios,  
                        can i get outcomes by percent movement?
    
    Notes:
        
        First try on combined score I was able to dig out some good results.
            Filtering on 0 was good (80%) - even just to try with ratio
            filter on 1 or two was great bu placemetns of course dropped
        
        ml did ok (60% precision with equal recall on largest target)
        would be worth it to contie with all currenceis and
        
        6 months data with ust ratio I was able to 
        
        
        Worled alright on just using c values - interesting 
        
        Does not seem to be muach (any0 correlation between)
            rolling_rank and rolling_position
        
        How to tell correlation between - if is it's in longer its' in shorter
        
        Interesting difference between:
            hml = c - h     better - which is smooth and well formed and similar for all
            hml = 2*c - h+l          jagged- a tiny bit higher (less placements at poitn)
    
        Have gotten goog results on smaller timesfrmaes but might have dropped
        on longer.  thismight be workable if it works on and off for periods of time.
        
    
        Window   Placements      Max  ------------
        
        60:     1615            0.64
        85:     1196            0.63
        110:    957             0.63
        135:    806             0.64
        160:    693             0.66
        185:    619             0.65
        210:    549             0.66
        235:    506             0.65
        260:    460             0.65
        285:    421             0.66
        310:    382             0.67
        335:    350             0.67
        360:    316             0.67
        385:    296             0.68
        410:    280             0.68
        435:    265             0.68
        460:    248             0.68
        485:    231             0.68
        510:    220             0.67
        535:    214             0.66
        560:    208             0.66
        585:    199             0.65
        610:    194             0.64
        635:    189             0.64
           
        
        6 minths = === not s good !!!!!       
        Window   Placements      Max  
        60:     2681            0.58
        110:    1598            0.59
        160:    1177            0.58
        210:    924             0.58
        260:    763             0.58
        310:    654             0.59
        360:    553             0.58
        410:    472             0.56
        460:    423             0.56
        510:    392             0.57
        560:    353             0.58
        610:    330             0.59
    
    '''





    '''
    This fitering is not correct.
    
    Try to score each indicator seperetely
    
    Do freq analysis
    '''

    # ============= Analyze Results ===========================================
    if 1:
        
        hml_index = []
        hml_minimums= []  
        hml_max_wins = []
        plt.figure()
        print('\nWindow \t Placements \t Max  \n-----------------\n')
        # For each window I need to print the mean stack of the df.
        for i in range(windows.shape[0]):
            # Try an index column and get ones index
            window_column = i

            # Use to set for highest values (1)
            hml_rank_index = hml_rolling_rank[hml_rolling_rank\
                               .iloc[:, window_column] == 1].index.values
            # Use to set for lowest value (windows[i])
            hml_rank_index = hml_rolling_rank[hml_rolling_rank\
                               .iloc[:, window_column] == windows[i]].index.values
            
                                              # Plot mean long for each target
            plt.plot(long.loc[hml_rank_index].mean().values, 
                     label=windows[window_column])
            
            # Print Max winning percentage at each window and at what target
            print('{}:\t{}\t\t{:.2f}'.format(windows[window_column], 
                                           hml_rank_index.shape[0],
                                           long.loc[hml_rank_index].mean().max()))
            hml_index.append(hml_rank_index)
            hml_minimums.append(minimums.loc[hml_rank_index]\
                            .mean().astype(int))
            hml_max_wins.append(long.loc[hml_rank_index].mean().max())
        plt.legend()
        plt.xlabel('Targets')
        plt.show()


        plt.figure()
        hml_minimums = pd.DataFrame(hml_minimums)    
        hml_minimums.mean().plot()        
        plt.title('Avergae Bars to Outcome')


        
        plt.figure()
        plt.plot(c)
        plt.title('Closing values')
        
        
    # ============= Filter on Additional Indicators ===========================
    if 1:
        
        # Get best performing target and winoow
        top_window = np.array(hml_max_wins).argmax()
        top_target = long.loc[hml_index[top_window]].mean().values.argmax()
        top_index  = hml_index[top_window]
        
        # Set aside hml outcomes Series
        hml_outcomes = long.iloc[top_index, top_target]
        win_hml = long.iloc[top_index, top_target][long.iloc[top_index, top_target]].index.values
        lose_hml = long.iloc[top_index, top_target][~long.iloc[top_index, top_target]].index.values
        win_hml = win_hml[win_hml > 1000]
        lose_hml = lose_hml[lose_hml > 1000]        
        
        
        # ============= Rolling Position ======================================
        
        # Set Filter on Rolling Pos for each window and graph winning perc =
        print('\n  ----------- Rolling Position ---------------------------\n')
        rolling_pos_filter = 4 # We are using less than.....
        plt.figure()
        print('\nwindow: Placements, winning % ----------------------------\n')
        for w in windows:
            rolling_window = w
            rolling_index = channel_pos.loc[top_index]
            combined_wins = rolling_index[rolling_index.loc[:, rolling_window] > rolling_pos_filter ]
            long.loc[combined_wins.index.values].mean().plot(label=rolling_window)
            plt.legend()
            print('{}: \t{}\t\t{:.2f}'.format(w, combined_wins.shape[0], long.loc[combined_wins.index.values].mean().max() ))

        '''
        # How do the win and lose indexes look on rolling_positions
        win_positions = rolling_pos.loc[hml_outcomes[win_hml].index.values]
        lose_positions = rolling_pos.loc[hml_outcomes[lose_hml].index.values]

        # Plot the winning and losing distibutions on Pos0itions
        plt.figure()
        sns.distplot(win_positions.values.ravel())
        sns.distplot(lose_positions.values.ravel())
        '''
        
        # ============= Rolling Slope =========================================
        
        # Set Filter on Rolling Slope
        print('\n\n\n  ----------- Slope ----------------------------------\n')
        slope_filter = 0 # greater than
        plt.figure()
        print('\nwindow: Placements, winning % ----------------------------\n')
        for w in windows:
            rolling_window = w
            rolling_index = channel_pos.loc[top_index]
            combined_wins = rolling_index[rolling_index.loc[:, rolling_window] > slope_filter ]
            long.loc[combined_wins.index.values].mean().plot(label=rolling_window)
            plt.legend()
            print('{}: \t{}\t\t{:.2f}'.format(w, combined_wins.shape[0], long.loc[combined_wins.index.values].mean().max() ))

        '''
        # How do the win and lose indexes look on rolling_positions
        win_positions = slopes.loc[hml_outcomes[win_hml].index.values]
        lose_positions = slopes.loc[hml_outcomes[lose_hml].index.values]

        # Plot the winning and losing distibutions on Pos0itions
        plt.figure()
        sns.distplot(win_positions.values.ravel())
        sns.distplot(lose_positions.values.ravel())
        '''
    
        # ============= Combined  =========================================

        # Set Filter on Rolling Slope
        print('\n\n\n  ----------- Combined Slope and  Posititon ----------\n')
        plt.figure()
        print('\nwindow: Placements, winning % ----------------------------\n')
        for w in windows:
            rolling_window = w
            rolling_index = slopes.loc[top_index]
            combined_wins = rolling_index[rolling_index.loc[:, rolling_window] < slope_filter ]
            rolling_index_1 = slopes.loc[top_index] 
            combined_wins_1 = rolling_index[rolling_index.loc[:, rolling_window] < slope_filter ]
            combined_index = np.intersect1d(combined_wins.index.values, 
                                          combined_wins_1.index.values)
            
            long.loc[combined_index].mean().plot(label=rolling_window)
            plt.legend()
            print('{}: \t{}\t\t{:.2f}'.format(w, combined_wins.shape[0], long.loc[combined_wins.index.values].mean().max() ))
        
        '''
        # How do the win and lose indexes look on rolling_positions
        win_positions = slopes.loc[hml_outcomes[win_hml].index.values]
        lose_positions = slopes.loc[hml_outcomes[lose_hml].index.values]

        # Plot the winning and losing distibutions on Pos0itions
        plt.figure()
        sns.distplot(win_positions.values.ravel())
        sns.distplot(lose_positions.values.ravel())
        '''
    
    
    
    
    
    
        '''
        # How do the win and lose indexes look on position rank
        win_positions_rank = position_rank.loc[hml_outcomes[win_hml].index.values]
        lose_positions_rank = position_rank.loc[hml_outcomes[lose_hml].index.values]
        # sns.heatmap(win_positions_rank)
        # sns.heatmap(lose_positions_rank)
    

        # Correlation between long window on rolling_position and rolling_rank
        pos_rank_corr = np.corrcoef(rolling_pos.loc[1000:, windows[-1]],
                                     hml_rolling_rank.loc[1000:, windows[-1]])
        pos_rank_1_corr = np.corrcoef(position_rank.loc[1000:, windows[-1]],
                                     hml_rolling_rank.loc[1000:, windows[-1]])  
        pos_rank_pos_rank = np.corrcoef(rolling_pos.loc[1000:, windows[-1]],
                                     position_rank.loc[1000:, windows[-1]])
        '''
        




        # Quick ML
        if 0:
            ml = rolling_pos.T.append(hml_rolling_rank.T).append(rolling_std.T).T.dropna()
            ml.columns = np.arange(ml.columns.shape[0])
            ml.drop(columns=[0], inplace=True) 
            ml = ml.sample(frac=1)
            ml_long = long.loc[ml.index.values]
            ml.reset_index(inplace=True)
            ml_long.reset_index(inplace=True)
            x = StandardScaler().fit_transform(ml.values)
            row = int(ml.shape[0] * .8)
            x_train = x[:row, :]
            x_test = x[row:, :]
            for t in range(6, targets.shape[0]):
                print(' ========= {} =========\n'.format(t))
                
                y_train = ml_long.iloc[:row, t].values
                y_test = ml_long.iloc[row:, t].values

                logreg = LogisticRegression(solver='sag',
                                            n_jobs=4,
                                            multi_class='ovr')
                
                logreg.fit(x_train, y_train )
                predictions = logreg.predict(x_test)
                
                print(cr(y_test, predictions))
                
            
                # Outcomes for different Predictions slices
                log_predictions = logreg.predict_proba(x_test)[:, 1]
                step = .05
                print('slice start, win_%, count')
                for start in np.arange(.5, 1, step).round(2):
                    pred_index_up = log_predictions > start
                    pred_index_down = log_predictions <= 10 #start + step
                    bool_index = pred_index_up & pred_index_down
                    log_index = np.arange(log_predictions.shape[0])[bool_index]
                    print('{}: \t {}\t\t{:.2f}'.format(start, 
                                                   log_index.shape[0], 
                                                   y_test[log_index].mean()))
            
                
            






# =============================================================================
# Strategy: High mid low currency difference
# =============================================================================
if 0:    
        
    '''
    Strategy:
        
        Get largest gaps between close and high values
        Assess outcomes 
        try other:
            granularity
        filter on other indicators:
            slope
            mean pricing 
            channel location
        As soon as I can get ov 65%, try with two currencies:
            get resulting positions for opposite directions
            get outcomes on proper ratio
            
    Notes:
        Perhaps it is more profitabel to use distnace of just low or high, etc
        Might want to try this on a different interval - short and Long
        How to set outcome targets correclty for diff currencies ?
        
    '''
    
    # ============= Get Data, Placement Locations and Outcomes ================

    # Parameters
    distribution_boundary = .03
    
    # How far is close from low and/or high
    hml = c - (h + l) / 2

    # Get Distribution Boundary and value
    hml_dist = get_distribution_boundary(hml, distribution_boundary)
    hml_diff_high = hml_dist['upper_index']
    hml_upper_bound = hml_dist['upper_bound']
    
    # Calculate Outcomes
    long = up < down
    short = down < up
    
    # Get and print mean minimum bars for outcomes
    minimums = pd.DataFrame(np.minimum(up.values, down.values), 
                            columns = up.columns, index = up.index)

    
    # ============= Add other Indicators and Filter Results ===================
    if 1:    
        
        # Parameters
        windows = np.array([3, 5, 15, 30, 45, 60, 90, 120, 240])

        # Choose a Direction and a target to work with
        direction = short
        outcomes_column  = np.argmax(direction.mean().values)
        outcomes = direction.iloc[:, outcomes_column]

        # ============= Get Some low values filters to test ===================

        # bollinger waves
        bollinger_dist = get_distribution_boundary(bollinger_waves.std(axis=1)\
                                              .values, distribution_boundary)
        bollinger_low = bollinger_dist['upper_index']

        # Channel waves 
        channel_dist = get_distribution_boundary(channel_waves.std(axis=1)\
                                              .values, distribution_boundary)
        channel_low = channel_dist['upper_index']
        
        # Slope waves 
        slope_dist = get_distribution_boundary(slope_waves.std(axis=1)\
                                              .values, distribution_boundary)
        slope_low = slope_dist['lower_index']
        
        # Bollinger Channel Difference
        boll_chan_diff_dist = get_distribution_boundary(mean_waves_difference.std(axis=1)\
                                              .values, distribution_boundary)
        boll_chan_diff_low = boll_chan_diff_dist['upper_index']

        # Get Intersection of Placements and Low bollinger waves mean
        intersections = np.intersect1d(outcomes.index.values, slope_low)
        
        # Get outcomes on filtered intersections
        filtered_outcomes = outcomes.loc[intersections]
        
        
        # ============= Analyze Results =======================================
        if 1:
            
            # Print Original Results
            msg = 'Original Winning % and Placements: {:.2f}\t{}'
            print(msg.format(outcomes.mean(), outcomes.shape[0]))
            # Print Filtered Results
            msg = 'Filtered Winning % and Placements: {:.2f}\t{}'
            print(msg.format(filtered_outcomes.mean(), filtered_outcomes.shape[0]))
        
        
        # ============= Graph Results  ========================================
        if 1:
            

            # Plot the first n placments
            placements_to_plot = 10
            end = hml_diff_high[placements_to_plot + 1]
            plt.figure()
            plt.plot(c[:end], label='close')
            plt.plot(h[:end], label='high')
            plt.plot(l[:end], label='low')
            plt.plot(hml_diff_high[: placements_to_plot], 
                     c[hml_diff_high[: placements_to_plot]], 
                     'o',color= 'black')
            plt.title('Closing Values with first few placements - COLOR')
            plt.legend()
            plt.show()
            
            
            # Plot long and short mean outcomes on target
            plt.figure()
            short.mean().plot(label='short')
            long.mean().plot(label='long')
            plt.legend()
            plt.title('Mean Short and long outcomes per target distance ')
            plt.show()
            
            # Plot the average bars to  outcome for each target distance
            plt.figure()
            plt.plot(minimums.mean())
            plt.title('Mean Minimum Outcome Bars')
        
            # Plot the Filtered Results
            plt.figure()
            plt.plot(c, label='close')
            plt.plot(h, label='high')
            plt.plot(l, label='low')
            plt.title('Plot All filtered Wins here')
            
            
    
'''
===============================================================================
Strategy.  Small variance for many indicators.  Add in Slope.
===============================================================================
'''
if 0:  
    
    '''
    boll and channel mena positiosn and position mean differnece around zero
    differecne btwwenm channe and bolinger mean near zero
    vol adjusted diff around zero
    maybe get mean of slope for direction
    doing pretty good so far with windows = array([  5,  15,  30,  60,  90, 120, 240])
    '''
    
    # ================= Zeros for all then slope ==============================
    
    # =============================================================================
    # I use the interval to create a distribution then select the bottom _x_ %.
    # I do this with each indicator and then see if there is an intersection.
    # Not a great way to do - it requires the intervcal instead of measuring
    # in real time (probably filtering on a constant ) eventually.
    
    # Also could weight slope
    
    # Also should incorporate channel and bollinger position somehow
    
    # Might look good when short slopes are opposite of long slopes
        # as a turn around point.
    # =============================================================================

    boundary = .05
    # Get lowest Bounded (percent) on both bollinger and channel mean and 
    boll_low = get_distribution_boundary(bollinger_waves.std(axis=1).values, boundary)
    chan_low = get_distribution_boundary(channel_waves.std(axis=1).values, boundary) 
    
    # Try the same on others as well
    diff_low = get_distribution_boundary(mean_waves_difference.std(axis=1).values, boundary)
    vol_diff_low = get_distribution_boundary(channel_waves.std(axis=1).values, boundary) 
    position_diff_low = get_distribution_boundary(position_difference_waves.std(axis=1).values, boundary) 
    chan_pos_low = get_distribution_boundary(channel_position_waves.std(axis=1).values, boundary) 
    boll_pos_low = get_distribution_boundary(mean_position_waves.std(axis=1).values, boundary) 
    std_low = get_distribution_boundary(std_waves.std(axis=1).values, boundary) 
    
    # Get time that are in low set on both bollinger and channel
    lower = np.intersect1d(boll_low['lower'], chan_low['lower'])
    lower = np.intersect1d(lower, diff_low['lower'])
    lower = np.intersect1d(lower, vol_diff_low['lower'])
    lower = np.intersect1d(lower, std_low['lower'])
    # lower = np.intersect1d(lower, position_diff_low['lower']) 
    
    ''' 
    Matching with the lower position difference does not always work
    '''
        
    # Print stats on lower location
    location = lower[0]
    print('Lower: {}'.format(lower)) 
    print('\n-----Mean Position--------')
    print(mean_position_waves.loc[location])
    print('\n-----Channel Position--------')
    print(channel_position_waves.loc[location])
    print('\n-----Slopes--------')
    print(slope_waves.loc[location])    

    # Plot location
    window_half = 200
    start = max(location - window_half, 0)
    end = min(location + window_half, c.shape[0] - 1 )
    x = np.arange(start, end)
    plt.figure()
    plt.plot(x, c[start: end])
    plt.plot(location, c[location], 'o')

    
    
    # ================= Position variance low and mean near c0 ==============================
    

    # Get where the variance is low on both channel and bollinger
    position_low_variance_intersection = np.intersect1d(chan_pos_low['lower'], boll_pos_low['lower'])
    print('\n\n')
    print('position_low_variance_intersection: {}'.format(position_low_variance_intersection))
    
    # Plot them - are they similar - near channels?  
    plt.figure()
    channel_position_waves.loc[position_low_variance_intersection].mean(axis=1).plot(style='o')
    mean_position_waves.loc[position_low_variance_intersection].mean(axis=1).plot(style = 'o')    
    plt.plot(np.ones(position_low_variance_intersection.max()) * 2, color='grey')
    plt.plot(np.ones(position_low_variance_intersection.max()) * -2, color='grey')    
    plt.figure()
    plt.plot(mean_position_waves.loc[position_low_variance_intersection[0]:\
                                     position_low_variance_intersection[0] + 100].mean(axis=1))
    plt.plot(channel_position_waves.loc[position_low_variance_intersection[0]:\
                                     position_low_variance_intersection[0] + 100].mean(axis=1))
    
    

    # ================= Both above combined (find low for everything ==============================

    combined = np.intersect1d(position_low_variance_intersection, lower)

    
'''  
===============================================================================
# Strategy.  Volume Correlations
===============================================================================
'''
if 0:  
    
    '''
    Volume correlations:
    volume correlation spikes.
    one is 'lower' than the other (not right exaclty - diff scales)
    Keeping the different scales for now - 
        WHEN THE DIFF BETWEEN ADJUSTED AND NORMAL GETS ' TOO LARGE ', 
        THE NORMAL WILL RISE OR FALL TO MATCH THE VOLUME ADJUSTED
    '''
    pass


'''
===============================================================================
# Strategy.  Slope Correlation
===============================================================================
'''
if 0:  

    '''
    some slpoed correlate, some dont.........
    Can one slope correlat the the futuer of another sloope?
    '''    
    pass


'''
===============================================================================
# Plot all Indicators, currency, etc. over small Interval currency.
===============================================================================    
'''
if 0:  
    

    '''
    Add location, interval, prediction values.
    Change (Fucking again) to use the dataframes above.
    '''
    
    
    # ================= Ready Plot ============================================
    
    # Set Subtitle in box as text
    def title(title, ax):
        ax.text(.5,.9,title,
            horizontalalignment='center',
            transform=ax.transAxes)
    
        
    # Plot Setup
    fig, ax = plt.subplots(5, 4, figsize=(10, 10), sharex = True)    
    plt.subplots_adjust(wspace=None, hspace=0)
    color_list = plt.cm.Blues(np.linspace(.25, .75, windows.shape[0]))
    x = np.arange(c.shape[0])
    c5 = np.ones(c.shape[0]) *  2
    c0 = np.ones(c.shape[0]) * -2
    zeros = np.zeros(x.shape[0])
    
    
    # ================= Currency ==============================================
    
    # Plot Currency and Taps and vollume adjusted
    ax[0, 0].plot(c)
    ax[0, 0].plot(x[taps['upper']], c[taps['upper']], 'o', color='green')
    ax[0, 0].plot(x[taps['lower']], c[taps['lower']], 'o', color='red')
    title('Currency and Taps', ax[0,0])
    
    # Plot Prediction Values
    a = ax[0, 3]
    a.plot(prediction_c)
    title('Prediction array', a)
    
    # Plot Delta
    ax[0, 1].plot(cur_diff)
    title('Currency Delta', ax[0,1])
    
    # Plot Autocorrelation
    ax[0, 2].plot(autocor['autocor'])
    ax[0, 2].plot(x[autocor['upper_cycle']], 
                  autocor['autocor'][autocor['upper_cycle']], 
                  'o', color='orange')
    ax[0, 2].plot(x[autocor['lower_cycle']], 
                  autocor['autocor'][autocor['lower_cycle']], 
                  'o', color='red')
    ax[0, 2].plot(zeros, color='grey')
    title('Currency Autocorrelation', ax[0, 2])
    

    # ================= Volume ================================================ 
    
    # Volume Adjusted
    ax[1, 0].plot(StandardScaler().fit_transform((c - c[0]).reshape(-1, 1)), 
                  label = 'normal')
    ax[1, 0].plot(StandardScaler().fit_transform((volume_adjusted \
                  - volume_adjusted[0]).reshape(-1, 1)), 
                  label = 'vol adjusted')
    ax[1, 0].legend()
    title('Volume Adjusted', ax[1,0])
    
    # Volume Adjusted Difference 
    a = ax[1, 1]
    vol_adjusted_difference.plot(ax=a, colors=color_list)
    a.plot(vol_adjusted_difference.mean(axis=1), color='black')
    a.plot(zeros, color='grey')
    b = a.twinx()
    b.plot(vol_adjusted_difference.std(axis=1), color='orange')
    title('Volume Adjsuted Difference', a)
    
    # volume Adjusted Correlation
    a = ax[1, 3]
    vol_adjusted_corr.plot(ax=a, colors = color_list)
    a.plot(vol_adjusted_corr.mean(axis=1), color='black')
    b = a.twinx()
    b.plot(vol_adjusted_corr.std(axis=1), color='orange')
    title('Volume Adjsuted Correlation', a)

    # Volume 
    ax[1, 2].plot(vol)
    title('Volume', ax[1, 2])
    
    # Volume Difference
    ax[2, 2].plot(vol_diff)
    title('Volume Delta', ax[2, 2])
    
    
    # ================= Mean ==================================================

    # Bollinger Waves
    a = ax[2, 0]
    bollinger_waves.plot(ax=a, color=color_list)
    a.plot(bollinger_waves.mean(axis=1), color='black')
    b = a.twinx()
    b.plot(bollinger_waves.std(axis=1), color='orange')
    title('Bollinger Wave', a)
        
    # Channel Waves
    a = ax[3, 0]    
    channel_waves.plot(ax=a, color=color_list)
    a.plot(channel_waves.mean(axis=1), color='black')
    b = a.twinx()
    b.plot(channel_waves.std(axis=1), color='orange')
    title('Channel Wave', a)
    
    # Mean Channel Difference
    a = ax[4, 0]
    mean_waves_difference.plot(ax=a, color=color_list)
    a.plot(mean_waves_difference.mean(axis=1), color='black')
    a.plot(zeros, color='grey')
    b = a.twinx()
    b.plot(mean_waves_difference.std(axis=1), color='orange')
    title('Bollinger Channel Difference', a)
    
    
    # ================= position ==============================================

    # Mean Position 
    a = ax[2, 1]
    mean_position_waves.plot(ax=a, color=color_list)
    a.plot(mean_position_waves.mean(axis=1), color='black')
    a.plot(c5, color='grey')
    a.plot(c0, color='grey')
    a.plot(mean_position_waves.std(axis=1), color='orange')
    title('Bollinger Position', a)
    
    # Channel Position 
    a = ax[3, 1]
    channel_position_waves.plot(ax=a, color=color_list)
    a.plot(c5, color='grey')
    a.plot(c0, color='grey')
    a.plot(zeros, color='grey')
    a.plot(channel_position_waves.mean(axis=1), color='black')
    a.plot(channel_position_waves.std(axis=1), color='orange')
    title(' Channel Position', a)
    
    # Mean Channel Position difference
    a = ax[4, 1]
    position_difference_waves.plot(ax=a, color=color_list)
    a.plot(zeros, color='grey')
    a.plot(position_difference_waves.mean(axis=1), color='black')
    a.plot(position_difference_waves.std(axis=1), color='orange')
    title('Position Difference', a)


    # ================= Slope =================================================

    # Slope Waves
    a = ax[3, 2 ]
    slope_waves.plot(ax=a, color = color_list)
    a.plot(zeros, color='grey')
    a.plot(slope_waves.loc[:, windows[2:]].mean(axis=1).values, color='black')
    b = a.twinx()
    b.plot(slope_waves.std(axis=1).values, color='orange')
    title('Slope Waves', a)
    
    # ================= Variance ==============================================
    
    # Variance Waves
    a = ax[2, 3]
    std_waves.plot(ax=a, color = color_list)
    a.plot(zeros, color='grey')
    a.plot(std_waves.mean(axis=1).values, color='black')
    a.plot(std_waves.std(axis=1).values, color='orange')
    title('Standard Deviation Waves', a)
    
    
    # ================= Variance ==============================================
    
    # Remove legend from waves
    for row in range(5):
        for col in range(4):
            try:
                ax[row, col].legend_.remove()
            except:
                pass
    
    # Add legends back in
    ax[1, 0].legend()
   
    # Plot Cast
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.show()


    
    
    
    
    
    
    
    
    
    se = StandardScaler().fit_transform(c.values)
    su = StandardScaler().fit_transform(cur.loc[:, 'usd'].values.reshape(-1, 1)).ravel()
    sr = StandardScaler().fit_transform(ratios.loc[:, 'eur_usd'].values.reshape(-1, 1)).ravel()
        
    plt.figure()
    plt.plot(se[:1000])
    plt.plot(su[:1000])    
    plt.plot(sr[:1000])
    
    plt.figure()
    plt.plot(se[:1000] - se[0], label='e')
    plt.plot(su[:1000] - su[0], label='u')    
    plt.plot(sr[:1000] - sr[0], label='rat')
    plt.plot((se/su)[:1000] - (se/su)[0], label='cal')
    
    plt.figure()
    plt.plot(se - se[0], label='e')
    plt.plot(su - su[0], label='u')    
    plt.plot(sr - sr[0], label='rat')
    plt.plot((se/su) - (se/su)[0], label='cal')
    
'''
===============================================================================
# Export
===============================================================================    
'''    
if 0:    
    
    df = pd.DataFrame(timestamp.copy())
    df = df.join(c, lsuffix='c')
    df = df.join(hml_rolling_rank, lsuffix='rolling_rank')
    df = df.join(long, lsuffix = 'long')
    df = df.join(channel_pos, lsuffix='channel_pos')
    df = df.join(slopes, lsuffix='slopes')
        
    columns = ['timestamp', 'currency']
    columns += ['hml_' + str(x) for x in windows]
    columns += ['long_' + str(round(x, 6)) for x in targets]
    columns += ['channel_pos_' + str(x) for x in windows]
    columns += ['slope_' + str(x) for x in windows]
    
    df.columns = columns
    
    df.to_csv('/Users/user/Desktop/results.csv')
    
    
    
































"""    
MORE MISC INDICATORS UNUSED SO FAR

from libraries.taps import get_taps
from libraries.correlation import get_autocorrelation
from libraries.correlation import get_rolling_correlation_waves
from libraries.waves import get_volume_adjusted_position_difference_waves
if 0:
    
    # Rolling Position Rank from window 240
    position_rank = waves_wrapper(rolling_pos.loc[:, position_rank_window], windows, get_rolling_rank)

    # Volume adjsuted price
    volume_adjusted = ((c_delta / vol).cumsum())

    # Volume Adjusted Scaled Difference    
    scaled_adjusted= StandardScaler().fit_transform(volume_adjusted.reshape(-1, 1)).ravel()
    scaled_values = StandardScaler().fit_transform(c.values.reshape(-1, 1)).ravel()
    volume_adjusted_scaled_difference = scaled_values - scaled_adjusted
    
    # Volume adjusted difference (position)
    vol_adjusted_difference = get_volume_adjusted_position_difference_waves(c.values, 
                              c_delta, vol, windows)
    # Volume Correlation
    vol_adjusted_corr = get_rolling_correlation_waves(c, volume_adjusted)

    # Autocorrelation values - plotted
    autocor = get_autocorrelation(c.values, 30, 30, 30, False)

    # Get taps
    taps_interval_left = 30
    taps_interval_right = 10
    taps = get_taps(c.values, taps_interval_left, taps_interval_right)

    # Get support - no point really though.
"""
