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
    from multiprocessing import cpu_count
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
    _from = '2017-01-01T00:00:00Z'
    _to   = '2017-02-01T00:00:00Z'

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
# Outcomes.  Get em.
# =============================================================================
if 1:
    
    
    

    up = cur.copy()
    down = cur.copy()
    targets = np.array([.001, .001, .001, .001, .001, .001]) 
    for i in range(cur.columns.shape[0]):
        col = cur.columns[i]
        target = np.array([targets[i]])
        outs = get_outcomes_multi(cur.loc[:, col].values, target, 10000)
        up.loc[:, col] = outs['up'].values
        down.loc[:, col] = outs['down'].values
    long = up < down
    short = down < up
    minimums = pd.DataFrame(np.minimum(up.values, down.values), 
                            columns = up.columns, 
                            index = up.index)
    print(minimums[minimums < 10000].dropna().mean())
    
        
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

    

# =============================================================================
# Strategy: Score Indicator (window) by bin membership and bin outcomes mean
# =============================================================================
if 0:
               
    
    
    '''
    Also missing - we might want to save the indexes as well (another df)
    Also         - we are redoing all bins / indexes adn everything when we don't have to
    '''    
    # ============ Get Bin lookup table =======================================    
    
                  
    ''' Get all bins for one window '''
    def get_bin_stats(ind, targets, window, long, short, return_bins, return_probs):
        bin_values = ind.loc[window:, window].values
        hist = np.histogram(bin_values, bins=bins)
        x = np.arange(direction[0].shape[0])
        long_probs  = long.copy()
        short_probs = short.copy()
        coll        = []
        for i in range(bins - 1):
            start = hist[1][i]
            end = hist[1][i+1]
            # To do for long
            cond1 = ind.loc[:, window].values >= start
            cond2 = True # ind.loc[:, window].values <= end # < < not using
            index = x[cond1 & cond2] 
            for target in targets:
                mean_outcome = long.loc[index, target].mean()
                long_probs.loc[index, target] = mean_outcome
                coll.append([window, 'long', target, i, index.shape[0], mean_outcome])
                # To do for short    
            cond1 = ind.loc[:, window].values <= end
            cond2 = True # ind.loc[:, window].values >= end # < < not using
            index = x[cond1 & cond2]   
            for target in targets:
                mean_outcome = short.loc[index, target].mean()            
                short_probs.loc[index, target] = mean_outcome
                coll.append([window, 'short', target, i, index.shape[0], mean_outcome])
        return_bins[window] = coll
        return_probs['long'] = long_probs
        return_probs['long'] = short_probs
        
            
    # Setup Parameters and loops
    windows = np.array([1500, 2000, 2250, 2500])
    currencies = cur.columns[0]#.tolist()
    directions   = [(long, 'long'), 
                    (short, 'short')]
    bins       = 60
    bin_coll   = []
    
    # Cycle through all values
    for currency in currencies:
        print(currency)
        '''
        # Get Currencies
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
        '''
        
        
        
        
        
        
        
        # Assign Indicators to use
        indicators = [(slopes, 'slopes')]#,
                     # (roll_chann_diff_pos, 'pos_diff'),
                     # (rolling_pos, 'rolling_pos'),
                     # (channel_std, 'channel_std'),
                     # (channel_pos, 'channel_pos')]
            
        for indicator in indicators:
            for direction in directions:
                for window in windows:
                    
                    # Call Multiprcessing entities
                    cpus = cpu_count()
                    manager = Manager()
                    return_bins = manager.dict()
                    return_probs = manager.dict()
                    jobs = [] 
                    # Run Jobs for first Rotations
                    for w in range(windows.shape[0]):
                        p = Process(target=get_bin_stats, 
                                                    args=(indicator[0],
                                                          windows[w], 
                                                          direction,
                                                          return_bins,
                                                          return_probs))
                        jobs.append(p)
                        p.start()
                        # Pause to join at number of cpus before moving on.
                        if (w % (cpus-1) == 0 and w > 0) \
                                or (w == windows.shape[0] - 1):
                            for job in jobs:
                                job.join()
                    # Add Results in order and return
                    to_append = [currency, 
                                 indicator[1]]
                    for w in sorted(return_dict.keys()):
                        to_append += [w, direction[1]] + targets.columns.to_list()
                        [bin_coll.append([to_append] + x) for x in return_dict[w]]
      
   
    
    # Create Results DF
    

                            
                        
    
    
    
    '''
    Try Just for one window, one target, just for slope.
    '''
    # Shorten data, set parameters
    long_probs = long.copy()
    short_probs = short.copy()
    window = 2000
    target = .0009
    bins = 40

    for currency in cur.columns:
        # SEt currerency.  Gather indicators, set indicators to use.
        c = cur.loc[:, currency]
        channel_stats = get_channel_mean_pos_std(c.values, np.array([window]))
        channel_std   = channel_stats['std']
        channel_pos   = channel_stats['pos']
        channel_mean  = channel_stats['mean']
        slopes        = channel_stats['slope']
        indicator = slopes.loc[window:, window].values
        # Get histogram
        hist = np.histogram(indicator, bins=bins)
        for i in range(bins - 1):
            start = hist[1][i]
            end = hist[1][i+1]
            # Create Index
            cond1 = indicator >= start 
            cond2 = indicator <= end
            index = cond1 & cond2
            # Long
            mean_outcome = long.loc[index, currency].mean()
            long_probs.loc[index, currency] = mean_outcome
            # Short
            mean_outcome = short.loc[index, currency].mean()            
            short_probs.loc[index, currency] = mean_outcome
        
    
    
    # Plot this strange data
    plt.figure()
    for col in long_probs.columns:
        plt.plot(long_probs.loc[:25000, col].values, label=col)
    plt.legend()
    plt.title('Long')
    plt.show()
    # Plot this strange data
    plt.figure()
    for col in short_probs.columns:
        plt.plot(short_probs.loc[:25000, col].values, label=col)
    plt.legend()
    plt.title('Short')
    plt.show()
    
    
    # When cur one above 80% on long, cur2 80% on short
    # Cur 1 goes up, cur 2 goes dows (not logical yet) = see rat.....
    cur1 = 'eur'
    cur2 = 'usd'
    cutoff = .7
    rat = '{}_{}'.format(cur1, cur2)
    try:
        rat = '{}_{}'.format(cur1, cur2)
        vals = ratios.loc[:, rat].values
    except:
        rat = '{}_{}'.format(cur2, cur1)
        vals = ratios.loc[:, rat].values   
    targets = np.array([.5, 1, 1.5]) * vals.std()
    outs = get_outcomes_multi(vals, targets, 10000)
    ratio_long = outs['up'] < outs['down']
    ratio_short = outs['up'] > outs['down']    
    
    long_filter = long_probs.loc[:, cur1].values > cutoff
    short_filter = short_probs.loc[:, cur2].values > cutoff
    index = long_filter & short_filter
    
    combined_mean = ratio_long.loc[index].mean()
    

            
            
            
            
    
    
    
    
    
    
    
            
            
        
        # For each window in indicator, get bins probabilities and shape
        for window in windows:
            vals = ind.loc[window:, window].values
            hist = np.histogram(vals, bins = bins)
            probs = hist[0] / vals.shape[0]    
            
            
            ind = df[0].copy()
            ind_prob = ind.copy()
    
            # Call Multiprcessing entities
            cpus = multiprocessing.cpu_count()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            jobs = [] 
            # Run Jobs for first Rotations
            for b in range(bins):
                p = multiprocessing.Process(target=score, 
                                            args=(values,
                                                  windows[b], 
                                                  return_dict))
                jobs.append(p)
                p.start()
                # Pause to join at number of cpus before moving on.
                if (b % (cpus-1) == 0 and b > 0) or (b == bins - 1):
                    for job in jobs:
                        job.join()
                     
            # Add Results in order and return
            return_df = pd.DataFrame()
            for w in windows:
                 return_df[w] = return_dict[w]
            mean.columns  = windows        
      
            
        
        
        
    
    
    
    
    
    
    
    
    
    
    
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
        
        
    











# =============================================================================
# Notes
# =============================================================================
'''   
    
    GOAL ======================================================================
        
    
        .  Score each window (seperetelu) of each ind by       
                bins and freq of target outcomes per bi
        .  Sum scores to contain final score per indicator
        .  Sum all indicators to obtain score for currency
                Questions:
                    Do any overide all others
                    Do we get better scores shrinking or exploding any measure?
        .  Do this for both long and short for each currency
                This will leave me with two df's - a long and short summed probs.
    
        .  
    
    
    
        how we slice it has a large effect (within bin or gt / lt
        
        
        
        Take score of each currency.
        Get outcomes of each ratio
        Create Strategies:
            Score correlation / prediction based on a (set) of indicators
            Predict Currency Movement with above 60% success based on indicators
    
    
    
    
        The reuslts are not as good as I hoped.  
        The grouped score drops significantly.
        
        Are results summed no better than just slope ? 
        
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
            Take the good index.  plot a few outcomes to make sure that it looks good.
            
            
            
            
            
'''
        
        
    
    
    
    
    
    
    
    
    
    
