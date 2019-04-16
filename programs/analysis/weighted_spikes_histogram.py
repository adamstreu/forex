import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os; os.chdir('/northbend')


'''
To do: (roughly in order)
    
    tommorrow: 
        
        Remove unneccesary directions
            
            
        Thourghly Check all computations, outcomes and results
            
            Double check my bars and groups - think i messed it up.
            and that it is indexing properly all the way from candles weighted
            Do i need to use both directions for all levels?
            Remember: stated distance isn;t right now 
                whould probably just be name of columns...duh.
            Took a lot longer since adding both sides to levels and group / bar reorg.
            double check histogram creation and resulting levels
    

        Do oucomes change if I do't get groups - why would they.
    
        If even descent, deploy both measured and unmeasure versions.
        
        
    Later in Week:
        
        Should we be computing stats not on group but on all ? ? ? ? 
        
        Add in filtering and unweighted closings movement

        Complete investment / return anlysis
            > bars, gain / loss / throughput, risk, shares, trades per day, 
            
        Plot multi-currency 'best' results of each and graph thier hits.
            Hopefully we don't want to see much correlation.
            
    Even Later:            

        Autocorrelation on differences might offer up something interesting


    Differnt Plans to follow:
        
        weighted following unweighted
        Gated:
            weighted followed by unweighted away then back, then placement


'''


if False:
    folder = 'EUR_USD_M1_2014-01-11_2019-01-01'
    path   = '/Users/user/Desktop/' + folder + '/'
    long_target = pd.read_csv(path + 'long_target.csv')
    long_loss = pd.read_csv(path + 'long_loss.csv')
    short_loss = pd.read_csv(path + 'short_loss.csv')
    short_target = pd.read_csv(path + 'short_target.csv')
    candles = pd.read_csv(path + 'candles.csv')
    candles.index.rename('location', inplace=True)
    for df in [long_target, short_target, long_loss, short_loss]:
        df.set_index('location', inplace=True, drop=True)


if False:
    # Parameters
    ###############################################################################
    closings = candles.midclose.values
    closings_shape = closings.shape[0]
    # Levels
    histogram_less_than    = 30000
    histogram_greater_than = 900    # WE want at least this many instance 
    histogram_bins         = 3000
    # Spike Interval
    spike_windows = np.arange(1, 300, 5)
    # Instantiations
    results = []
    indexes = []
    
    # Create Slopes
    slope_interval = 500
    slope          = np.insert(closings, closings_shape, np.zeros(slope_interval))\
                   - np.insert(closings, 0, np.zeros(slope_interval))
    slope[slope >= 0] = 1
    slope[slope <= 0] = 0
    slope             = slope[slope_interval:-slope_interval] 
    candles.loc[slope_interval: ,'slope'] = slope
    
    # Create Grouping interval based on bars
    bar_limit = 1500
    bars = []
    for t in long_target.columns:
        tmp = []
        for l in long_target.columns:             
            mins = np.minimum(short_target.loc[:, t].values, short_loss.loc[:, l].values)
            tmp.append(int(mins[mins < mins.max()].mean()))
        bars.append(tmp)
    bars = pd.DataFrame(bars, columns = long_target.columns)
    bars.index = long_target.columns
    
    
    ###############################################################################  
    # Beggin Iteration Sequence        
    ###############################################################################  
    
    ## For direction 
    #for direction in ['long', 'short']:
    #    if direction == 'long':
    #        target   = long_target
    #        loss     = long_loss
    #    else:
    #        target   = short_target
    #        loss     = short_loss
        
    target   = long_target
    loss     = long_loss
    # Spike Intervals
    ###########################################################################
    for spike_window in spike_windows:
        print('Spike: {}'.format(spike_window))
        # Get weighted Values 
        differences         = np.insert(closings, 
                                 closings_shape, 
                                 np.zeros(spike_window)) \
                            - np.insert(closings, 
                                0, 
                                np.zeros(spike_window))
        differences         = differences[spike_window:-spike_window]
        candles.loc[spike_window: ,'difference'] = differences
        candles['weighted'] = candles.difference / candles.volume
    
        
        # Get Levels
        ###############################################################################
        #  Get Negative Levels
        hist    = np.histogram(candles.weighted.values[spike_window:], 
                                      bins=histogram_bins)
        cumsum    = hist[0].cumsum()
        minimum = np.argmax(cumsum > histogram_greater_than)
        maximum = np.argmax(cumsum > histogram_less_than)
        step = np.floor((maximum - minimum) / 9)
        hist_index = (minimum + np.arange(10) * step).astype(int)
        levels = hist[1][hist_index].tolist()
    #    # Get Positive Levels
    #    hist0 = np.flip(hist[0], axis=0)
    #    hist1 = np.flip(hist[1], axis=0)
    #    cumsum = hist0.cumsum()
    #    minimum = np.argmax(cumsum > histogram_greater_than)
    #    maximum = np.argmax(cumsum > histogram_less_than)
    #    step = np.floor((maximum - minimum) / 9)
    #    hist_index = (minimum + np.arange(10) * step).astype(int)
    #    levels += hist1[hist_index].tolist()
        
        
        # Get Levels to filter spikes on.  Set long or short targets as well
        #######################################################################
        for level in levels:  
            # Set what we are looking for from the limit
            if level > 0:
                out_ind = candles.loc[candles.weighted > level].index.values
                #                target   = short_target
                #                loss     = short_loss
            else:
                out_ind = candles.loc[candles.weighted < level].index.values 
                #                target   = long_target
                #                loss     = long_loss 
            # print('Level: {:.6f}.  Shape: {}'.format(level, out_ind.shape[0]))  
            #print('T, L, Count, Returns')
    
            
            
            # Apply Filters
            #######################################################################
            # Well, not a filter maybe but yet another combination to take forever
          # for s in [1, 0]:
            slopes = candles[candles.slope == 1].index.values
            out_ind = np.intersect1d(out_ind, slopes)
            
            
    
            if out_ind.shape[0] > 0: 
                
                
                # Calculate Returns on iterators and parmeters
                ########################################################s######
                for t in target.columns[:3]:
                    for l in target.columns[:3]: 
                        # Only compute columns where bar < bar_limit and t > l
                        if float(t) >= float(l):# and bars.loc[t, l] < bar_limit:
        
         
                            # Get groups by using bar average spacing                            
                            ###################################################
                            groups = [out_ind[0]]
                            groups += out_ind[1:][(out_ind[1:] \
                                   - out_ind[:-1]) >= bars.loc[t, l]].tolist()
                            groups = np.intersect1d(target.index.values,
                                                    groups)
                            
                       # This undoes      what's above.  jusu checking
                            groups = out_ind
        
                            # Calculate Winnings and returns
                            ###################################################
                            # win percent
                            win_perc = (target.loc[groups, str(t)] \
                                     < loss.loc[groups, str(l)]).mean()
                            # spread
                            spread   = candles.loc[groups, 'spread'].sum()
                            # Wins and loss
                            winnings = win_perc * float(t)
                            losses   = (1 - win_perc) * float(l)
                            # returns
                            returns  = len(groups) * (winnings - losses)
                            returns  -= spread
                          
                            #print('{}, {}: \t{}\t{}'.format(t, l, groups.shape[0], returns))
                            
                            
                            # Append results to results
                            ###############################################
                            results.append([#direction,
                                            #s,
                                            round(level, 7),
                                            spike_window,
                                            len(groups),
                                            out_ind.shape[0],
                                            float(t), 
                                            float(l), 
                                            bars.loc[t, l],
                                            win_perc,
                                            returns])
                            indexes.append(groups)
                                                        
    ###############################################################################  
    # End Iteration Sequence        
    ###############################################################################  
        
        
    # Build Results DataFrame   
    ###############################################################################
    columns = [#'direction',
               #'slope',
               'level', 
               'spike_window',
               'group_count',
               'index_count',
               'target', 
               'loss',
               'bars',
               'winnings',
               'returns'
               ]
    results = pd.DataFrame(results, columns=columns)
    results['throughput'] = results.returns / results.bars
    groups = pd.DataFrame(np.array(indexes), columns=['indexes'])
    
    """
    # Filter Results and Print  
    ###############################################################################
    print()
    print('Highest Long Returns')
    print(results.loc[results.direction == 'long']\
                      .sort_values('returns', ascending=False).head(10))
    print()
    print('Highest Long Winnings with more than _ groups')
    print(results.loc[(results.group_count > 900) & (results.direction == 'long')]\
                      .sort_values('winnings', ascending=False).head(10))
    print()
    print('Highest Short Returns')
    print(results.loc[results.direction == 'short']\
                      .sort_values('returns', ascending=False).head(10))
    print()
    print('Highest Short Winnings with more than _ groups')
    print(results.loc[(results.group_count > 20) & (results.direction == 'short')]\
                      .sort_values('winnings', ascending=False).head(10))
    print()
    print('Highest  Throughput')
    print(results.sort_values('throughput', ascending=False).head(10))
    
    
    # Plot Results   
    ###############################################################################
    # Plot candles for reference
    plt.figure()
    candles.midclose.plot(figsize=(12, 4))
    # Plot distribution of top returning group
    ind = groups.loc[results.sort_values('returns', ascending=False)\
                     .head(1).index.values[0]].values[0]
    x   = np.zeros(max(ind) + 1) 
    x[ind] = 1
    plt.figure(figsize=(12, 1))
    plt.plot(x, 'o')
    
    
    
    
    # It appears, unfiliters, testing a few spikes, that these offer nothing.
    # If these are all near 50% I don't have anything.
    ###############################################################################
    for direction in ['long', 'short']:
        for spike in spike_windows:
            for l in results.level.unique():
                cond1 = (results.target == 0.0015)
                cond2 = (results.loss   == 0.0015 )
                cond3 = (results.level == l) 
                cond4 = (results.spike_window == spike)
                cond5 = (results.direction == direction)
                cond6 = (candles.slope)
                filters = cond1 & cond2 & cond3 & cond4 & cond5
                msg = '{}\t{}\t{:.6f}\t{:.2f}'
                print(msg.format(direction,
                                 spike,
                                 l, 
                                 results[filters].winnings.mean()))
    
    
    
    
    # Filter on up slopes with one group (comparison for now only)
    ###############################################################################
    if False:
        indexes = groups.loc[2554].values[0]
        for s in [1, 0]:
            for t in long_target.columns:
                for l in long_target.columns:
                    if float(t) >= float(l) and bars.loc[t, l] < bar_limit:
                        slope_up = candles[candles.slope == s].index.values
                        inds = np.intersect1d(indexes, slope_up)
                        target = '0.0010'
                        loss   = '0.0010'
                        print('{}\t{}\t{}\t{}\t{:.2f}'\
                              .format(s,
                                      t,
                                      l,
                                      inds.shape[0],
                                      (long_target.loc[inds, t] < long_loss.loc[inds, l]).mean()))
    
    
    
    dist = np.zeros(candles.shape[0])
    dist[inds] = 1
    plt.plot(dist)
    
    
    print('Here is How many I am really interested in')
    results[(results.winnings > .65) & (results.returns > 0)] # and this isn;t even group number....
    
    
    
    
    
    # How long after weighted spike is large unweighted move
    ###############################################################################
    rezzies = []
    
    for level in levels:
        if level > 0:
            ind = candles.loc[candles.weighted > level].index.values
        else:
            ind = candles.loc[candles.weighted < level].index.values
        print('Level: {:.6f}.  Shape: {}'.format(level, ind.shape[0]))
        for move in [.01, -.01]:
            tmp = []
            for index in ind:
                try:
                    if move > 0:
                        tmp.append(np.argmax(candles.difference.values[index:] > move ))
                    else:
                        tmp.append(np.argmax(candles.difference.values[index:] < move ))
                except:
                    pass
            movements = np.array(tmp)
            rezzies.append([level, move, movements.mean(), movements.std()])
    
    
    
    
    # Plot all ot them
    ###############################################################################
    level = levels[0]
    ind = candles.loc[candles.weighted < level].index.values
    for each in ind:
        fig, ax = plt.subplots(2, 1, figsize=(20,14))
        ax[0].plot(np.arange(1001), candles.loc[each - 1000:each, 'midclose'].values)
        ax[0].plot(np.arange(1001) + 1000, candles.loc[each: each+1000, 'midclose'].values)
        ax[1].plot(np.arange(1001), candles.loc[each - 1000:each, 'weighted'].values)
        ax[1].plot(np.arange(1001) + 1000, candles.loc[each: each+1000, 'weighted'].values) 
        ax[1].plot(np.ones(2000) * level, '_')
        plt.show()
        print(each)
        cont = input('type touch')
        if cont == 'd':
            break
    """
    
###############################################################################
# K_mean 
###############################################################################
    
from sklearn.cluster import k_means
clusters = 5
k = k_means(results.values, clusters)
for i in range(clusters):
    k_ind = results[k[1] == i].index.values
    print(k_ind.shape)




