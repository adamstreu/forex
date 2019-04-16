###############################################################################
# Notes
###############################################################################
if True:

    '''
    By Sunday evening:
        Continuos Update.
        Pl
        
        

        Ongoing:
            
            Data:
                
                Just add to obviously.
                redo when want new currency or diff interval
                
            Plot:
                plot everything as floating windows.
                Then just add new data to plot (if best way) or clear
                just call up, etc.  Then can arrange as desiredd.
                
            Organization:
                Try - a seperate window (subplotted) for each currency.
                    contains all info for each (with data printed as desired as well)


        other indicators:
            Like I do - the channel position of each in currency set.
            Then take the average for each winow.  
            Put each average into a chart.
            
            
            
        Currency Chart:
            
            Parameters:
                Intervals for plotting
                windows for calculations
                granularity
                
            Plots:    
                Price
                chifted price currency set
                
                
            Data:
                
                
            Notes:
                Whole thing should update in real time.  multiprocess if needed
                Make sure to grid every plot
            


        Last positions vlues in subplots
        Volume universe. ( more interset in one grwoing ? )
        add jpy and hkd back in - correct them
        
        Method: 
            measure instrument in set that moves against rest of set
        
        
        
        a sum of instrument  slopes, etc.  forget remember later

        Simultion till 11:00:
            
        
        
        Backstudy Graphs on Slopes / position:
            Hone in on ONE method that seems to be reliable:
                understand why, name it and look for it tomorrow.
                

        Prepare Study:
            feed then keep updated:
                scale slopes. etc on entrire dataset ( previous year, say)
                (by granularity)
                mke afaster
    
            I wonder if I can calculate channel in tableau and 
        just adjust length in real time with parameter ( much later)
        
        keep it live.  graph updates (in system.- f tableau)
    
    NOTE TO SELF:
        
        Did a bad job of watching the market today.  Something I 
        nee to learn to do well.  Should be able to nurture a feel for 
        where the market is for every relevant instrument / currency.
        
        After 11 or so no point in putting down positions.  Spread is
        too high.
        
        Need to track carefully, remember well, be focused and log shit
        and place things in correct locations, unhurried, with limits, 
        with stops and targets.
        



    Method 1:  (described for short for one currency (match two of course)
        Position:
            instruments grouped well.  
            At top of channel on multiple timeframes
            (?) did not spike its way up too much
        Slope:
            Nada ? ? ? ?
        Placement:
            Wait for a final spike against position, set limit high, etc.
            Maybe kill when small window hits c0 ?
            
    Method 2:
        When positions are relavitely tight in one currency cluster,
            but then two branch out in seperate directions:
                place if they hit the outside channels near the same time.
                    (or turn around together)
                    
    Method 3:
        Slope peaks.  Long play
    
    '''


###############################################################################
# Imports 
###############################################################################
if 1:

    # Imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    import os; os.chdir('/northbend')
    from libraries.currency_universe import backfill_with_singular
    from libraries.indicators import get_channel_mean_pos_std
    from libraries.oanda import market
    from libraries.oanda import get_multiple_candles_spread
    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15) 
    np.warnings.filterwarnings('ignore')   
    plt.rcParams['figure.figsize'] = [14, 4]
    
   

-1###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   

    # General Parameters
    _from = '2018-12-10T00:00:00Z'
    _to   = '2019-12-01T00:00:00Z'
    granularity = 'M15'
    # Currencies to use
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'hkd', 'jpy', 'nzd', 'usd']     
    
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

    # Print average sum of Currencies ( must equal 1)
    print(cu.sum(axis=1).mean())

     


###############################################################################
# Set Parameters
###############################################################################    
if 1:     

    # Only use these currencies for placements (high leverage)
    currencies = ['aud', 'cad', 'nzd', 'chf', 'eur', 'usd']
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'hkd', 'jpy', 'nzd', 'usd']     
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'usd']     

    
    # Windows and calculation interval
    windows = np.array([30, 60, 120])   
    end = ratios.last_valid_index()
    start = end - (windows[-1] * 2)
    
    # standardize position and slope values ?
    standardize = False

    # Solo interval for graphing
    interval = 240





###############################################################################
# Print Currencies AND Currency Set
###############################################################################    
if 1:
    
    # Print what I need !
    print('\n\n\n\n\n')
    spread = get_multiple_candles_spread(instrument_list, granularity)

    # Print prices and granularity
    print()
    print(granularity)
    print()
    print(cu.loc[cu.last_valid_index(), 'timestamp'])
    print()

    for each in cu.iloc[cu.last_valid_index(), 1:].astype(float).round(6).values:
        print(each)
    print()
    for each in sorted(spread.keys()):
        print(spread[each])



    #o the rest
    for currency in currencies:
        # Get ticks by pip for currencies
        max_ticks = cu.loc[cu.last_valid_index() - interval:,
                           currency].values.max()
        min_ticks = cu.loc[cu.last_valid_index() - interval:,
                          currency].values.min()
        if currency in  ['hkd', 'jpy']:
            ticks = np.arange(min_ticks, max_ticks, .00001).round(4)
        else:
            ticks = np.arange(min_ticks, max_ticks, .0001).round(4)

        # Plot currencies
        fig = plt.figure(figsize=(14,4))
        if currency in  ['hkd', 'jpy']:
            fig.set_facecolor('xkcd:salmon')
        cu.loc[cu.last_valid_index() - interval: ,  currency].plot()
        plt.tight_layout()
        plt.title(str(currency) + '        ' +str(cu.loc[cu.last_valid_index(),  currency].round(5)))
        plt.yticks(ticks)
        plt.grid(which='both')
        plt.show()  
        


        # Get Insturment List and which direction to align instrument
        pair_list = []
        shape_list = []
        for pair in ratios.columns[1:]:
            if pair.split('_')[0].lower() in currencies \
                and pair.split('_')[1].lower() in currencies:
                if currency.upper() in pair.split('_'):
                    pair_list.append(pair)
                    if currency.upper() == pair.split('_')[0]:
                        shape_list.append(1)
                    else:
                        shape_list.append(-1)
                        
        # Get Slope position for all values (instruments)
        currency_set = pd.DataFrame()
        for i in range(len(pair_list)):
            instrument = pair_list[i]
            shape = shape_list[i]
            if shape == 1:
                values = ratios.loc[end - interval: end, instrument]
            else:
                values = (ratios.loc[end - interval: end, instrument] * -1)\
                       + (2 * ratios.loc[start: end,  instrument].values[-1])

            currency_set[pair_list[i]] = values
            currency_set[currency] = cu.loc[cu.last_valid_index() - interval: ,  currency]
            
        # Standardize in weird way
        scaler = StandardScaler().fit(currency_set.values)
        standardized_set = scaler.transform(currency_set.values) \
             - scaler.transform(currency_set.values)[0]
        std_set = pd.DataFrame(standardized_set, columns=pair_list + [currency])
                 
        # Plot Standardized Values
        std_set.plot()
        plt.title('Standardized.  Values not accurate:     '+ str(currency))
        plt.tight_layout()
        plt.legend()
        plt.show()
        
        # Non Standardized
        (currency_set - currency_set.loc[currency_set.first_valid_index()]).plot()
        plt.title('Shifted.       '+ str(currency))
        plt.tight_layout()
        plt.legend()
        plt.show()
    
        # Give it some space
        print('\n\n')
        
        
###############################################################################
# Print Up close position movements
###############################################################################    
if 1:
            
    interval = 10
    window = np.array([60])

    for currency in currencies:    
        # Get Insturment List and which direction to align instrument
        pair_list = []
        shape_list = []
        for pair in ratios.columns:
            if currency.upper() in pair.split('_'):
                pair_list.append(pair)
                if currency.upper() == pair.split('_')[0]:
                    shape_list.append(1)
                else:
                    shape_list.append(-1)
    
        # Get Slope position for all values (instruments)
        positions = pd.DataFrame()
        deviations = pd.DataFrame()
        slopes = pd.DataFrame()
        for i in range(len(pair_list)):
            instrument = pair_list[i]
            shape = shape_list[i]
            if shape == 1:
                values = ratios.loc[start: end, instrument]
            else:
                values = (ratios.loc[start: end, instrument] * -1)\
                       + (2 * ratios.loc[start: end,  instrument].values[-1])
            pos = get_channel_mean_pos_std(values.values, window)
            positions[instrument] = pos['pos'].values.ravel()
            deviations[instrument] = pos['std'].values.ravel()
            slopes[instrument] = pos['slope'].values.ravel()
    
            # Get (again) currency position
            c_positions = pd.DataFrame()
            pos = get_channel_mean_pos_std(cu.loc[start:end, currency]\
                                           .values.astype(float), 
                                           window)
            c_positions[currency] = pos['pos'].values.ravel()

        # Standardize
        if standardize:
            positions = pd.DataFrame(positions, columns = pair_list, index=np.arange(start, end + 1)) 
            c_positions = StandardScaler().fit_transform(pos['pos'].fillna(0).values.reshape(-1, 1))
            c_positions = pd.DataFrame(c_positions.ravel(), columns = [currency], index=np.arange(start, end + 1))
        else:
            positions.index = np.arange(start, end + 1)
            c_positions.index = np.arange(start, end + 1)     
        
        # Plot parameters for only most recent values
        plt_end = cu.last_valid_index()
        plt_start = plt_end - interval
        
        # Plot most recent values
        positions = positions.loc[plt_start: plt_end]
        c_positions = c_positions.loc[plt_start: plt_end]
        positions.plot(figsize=(14, 6))
        plt.plot(positions.index.values, c_positions.values, color='black')
        plt.plot(positions.index.values, np.zeros(positions.shape[0]), color='grey')
        plt.plot(positions.index.values, np.ones(positions.shape[0]) * 2, color='grey')
        plt.plot(positions.index.values, np.ones(positions.shape[0]) * -2, color='grey')
        plt.title('Positions of ' + str(currency) + ' @ ' + str(window) + ' @ ' + str(granularity))
        plt.tight_layout()
        plt.show()
            

    























































      
###############################################################################
# Graph Position indicator on all relavent Instruments and currency
###############################################################################    
if 0:
    
    for currency in currencies:
        for window in windows:
            window = np.array([window])
        
            # Get Insturment List and which direction to align instrument
            pair_list = []
            shape_list = []
            for pair in ratios.columns:
                if currency.upper() in pair.split('_'):
                    pair_list.append(pair)
                    if currency.upper() == pair.split('_')[0]:
                        shape_list.append(1)
                    else:
                        shape_list.append(-1)
        
            # Get Slope position for all values (instruments)
            positions = pd.DataFrame()
            deviations = pd.DataFrame()
            slopes = pd.DataFrame()
            for i in range(len(pair_list)):
                instrument = pair_list[i]
                shape = shape_list[i]
                if shape == 1:
                    values = ratios.loc[start: end, instrument]
                else:
                    values = (ratios.loc[start: end, instrument] * -1)\
                           + (2 * ratios.loc[start: end,  instrument].values[-1])
                pos = get_channel_mean_pos_std(values.values, window)
                positions[instrument] = pos['pos'].values.ravel()
                deviations[instrument] = pos['std'].values.ravel()
                slopes[instrument] = pos['slope'].values.ravel()
                
                '''
                # Get (again) currency position
                c_positions = pd.DataFrame()
                pos = get_channel_mean_pos_std(cu.loc[start:end, currency]\
                                               .values.astype(float), 
                                               window)
                c_positions[currency] = pos['pos'].values.ravel()
                '''
                
            # Standardize
            if standardize:
                positions = pd.DataFrame(positions, columns = pair_list, index=np.arange(start, end + 1)) 
                c_positions = StandardScaler().fit_transform(pos['pos'].fillna(0).values.reshape(-1, 1))
                c_positions = pd.DataFrame(c_positions.ravel(), columns = [currency], index=np.arange(start, end + 1))
            else:
                positions.index = np.arange(start, end + 1)
                c_positions.index = np.arange(start, end + 1) 
                
            # Plot
            positions.plot(figsize=(14, 6))
            plt.plot(positions.index.values, c_positions.values, color='black')
            plt.plot(positions.index.values, np.zeros(positions.shape[0]), color='grey')
            plt.plot(positions.index.values, np.ones(positions.shape[0]) * 2, color='grey')
            plt.plot(positions.index.values, np.ones(positions.shape[0]) * -2, color='grey')
            plt.title(str(currency) + ' @ ' + str(window) + ' @ ' + str(granularity))
            plt.tight_layout()
            plt.show()
            
            # Plot Prameters only most recent values
            interval = 10
            plt_end = cu.last_valid_index()
            plt_start = plt_end - interval
            positions = positions.loc[plt_start: plt_end]
            #c_positions = c_positions.loc[plt_start: plt_end]
            # Plot recent values
            positions.plot(figsize=(6, 6))
            #plt.plot(positions.index.values, c_positions.values, color='black')
            plt.plot(positions.index.values, np.zeros(positions.shape[0]), color='grey')
            plt.plot(positions.index.values, np.ones(positions.shape[0]) * 2, color='grey')
            plt.plot(positions.index.values, np.ones(positions.shape[0]) * -2, color='grey')
            plt.title(str(currency) + ' @ ' + str(window) + ' @ ' + str(granularity))
            plt.tight_layout()
            plt.show()
                
                
      
        
###############################################################################
# Get slopes Indicator on Currencies
###############################################################################    
if 0:

    # Get Slopes (all currencies for each window)
    for i in range(windows.shape[0]):
       # start = cu.last_valid_index() - (windows[-1] * 3)
        c_positions = pd.DataFrame()
        deviations = pd.DataFrame()
        slopes = pd.DataFrame()
        for currency in currencies:
            pos = get_channel_mean_pos_std(cu.loc[start: end, currency]\
                                           .values.astype(float), 
                                           np.array([windows[i]]))
            c_positions[currency] = pos['pos'].values.ravel()
            deviations[currency] = pos['std'].values.ravel()
            slopes[currency] = pos['slope'].values.ravel()
            
        # Standardize
        if standardize:
            slopes = StandardScaler().fit_transform(slopes.fillna(0))
            slopes = pd.DataFrame(slopes, columns = currencies, index=np.arange(start, end + 1)) 
        else:
            slopes.index = np.arange(start, end + 1)
        
        # Plot slopes
        slopes.plot()   
        plt.plot(np.arange(start, end + 1), np.zeros(slopes.shape[0]), color='grey')
        plt.tight_layout()
        plt.title(str(windows[i]) + ' @ ' + str(granularity))
        plt.show()
    


        
            

###############################################################################
# Graph Position indicator on all relavent Instruments 
# but - jsut the most recents
###############################################################################    
if 0:


    for currency in currencies:
        for window in windows:
            window = np.array([window])
        
            # Get Insturment List and which direction to align instrument
            pair_list = []
            shape_list = []
            for pair in ratios.columns:
                if currency.upper() in pair.split('_'):
                    pair_list.append(pair)
                    if currency.upper() == pair.split('_')[0]:
                        shape_list.append(1)
                    else:
                        shape_list.append(-1)
        
            # Get Slope position for all values (instruments)
            positions = pd.DataFrame()
            deviations = pd.DataFrame()
            slopes = pd.DataFrame()
            for i in range(len(pair_list)):
                instrument = pair_list[i]
                shape = shape_list[i]
                if shape == 1:
                    values = ratios.loc[start: end, instrument]
                else:
                    values = (ratios.loc[start: end, instrument] * -1)\
                           + (2 * ratios.loc[start: end,  instrument].values[-1])
                pos = get_channel_mean_pos_std(values.values, window)
                positions[instrument] = pos['pos'].values.ravel()
                deviations[instrument] = pos['std'].values.ravel()
                slopes[instrument] = pos['slope'].values.ravel()
        
                # Get (again) currency position
                c_positions = pd.DataFrame()
                pos = get_channel_mean_pos_std(cu.loc[start:end, currency]\
                                               .values.astype(float), 
                                               window)
                c_positions[currency] = pos['pos'].values.ravel()
    
            # Standardize
            if standardize:
                positions = pd.DataFrame(positions, columns = pair_list, index=np.arange(start, end + 1)) 
                c_positions = StandardScaler().fit_transform(pos['pos'].fillna(0).values.reshape(-1, 1))
                c_positions = pd.DataFrame(c_positions.ravel(), columns = [currency], index=np.arange(start, end + 1))
            else:
                positions.index = np.arange(start, end + 1)
                c_positions.index = np.arange(start, end + 1)     
            
            # Plot parameters for only most recent values
            interval = 10
            plt_end = cu.last_valid_index()
            plt_start = plt_end - interval
            
            # Plot most recent values
            positions = positions.loc[plt_start: plt_end]
            c_positions = c_positions.loc[plt_start: plt_end]
            positions.plot(figsize=(14, 6))
            plt.plot(positions.index.values, c_positions.values, color='black')
            plt.plot(positions.index.values, np.zeros(positions.shape[0]), color='grey')
            plt.plot(positions.index.values, np.ones(positions.shape[0]) * 2, color='grey')
            plt.plot(positions.index.values, np.ones(positions.shape[0]) * -2, color='grey')
            plt.title(str(currency) + ' @ ' + str(window) + ' @ ' + str(granularity))
            plt.tight_layout()
            plt.show()
                
          

    


###############################################################################
# Graph Currency Sets.
# Align inverse positions
###############################################################################    
if 0:

    interaval = 120


    for currency in currencies:
        # Get Insturment List and which direction to align instrument
        pair_list = []
        shape_list = []
        for pair in ratios.columns:
            if currency.upper() in pair.split('_'):
                pair_list.append(pair)
                if currency.upper() == pair.split('_')[0]:
                    shape_list.append(1)
                else:
                    shape_list.append(-1)
        # Get Slope position for all values (instruments)
        # positions = pd.DataFrame()
        # deviations = pd.DataFrame()
        # slopes = pd.DataFrame()
        currency_set = pd.DataFrame()
        for i in range(len(pair_list)):
            instrument = pair_list[i]
            shape = shape_list[i]
            if shape == 1:
                values = ratios.loc[end - interval: end, instrument]
            else:
                values = (ratios.loc[end - interval: end, instrument] * -1)\
                       + (2 * ratios.loc[start: end,  instrument].values[-1])

            currency_set[pair_list[i]] = values
            
        # Standardize in weird way
        standardized_set = StandardScaler().fit_transform(currency_set.values) \
             - StandardScaler().fit_transform(currency_set.values)[0]
        std_set = pd.DataFrame(standardized_set, columns=pair_list)
                 
        # Plot Standardized Values
        std_set.plot()
        plt.title('Standardized.  Values not accurate:     '+ str(currency))
        plt.show()



###############################################################################
# Print Currencies
###############################################################################    
if 0:

    for currency in currencies:

        # Get ticks by pip for currencies
        max_ticks = cu.loc[cu.last_valid_index() - interval:,
                           currency].values.max()
        min_ticks = cu.loc[cu.last_valid_index() - interval:,
                          currency].values.min()
        if currency in  ['hkd', 'jpy']:
            ticks = np.arange(min_ticks, max_ticks, .00001).round(4)
        else:
            ticks = np.arange(min_ticks, max_ticks, .0001).round(4)

        # Plot currencies
        fig = plt.figure(figsize=(14,4))
        if currency in  ['hkd', 'jpy']:
            fig.set_facecolor('xkcd:salmon')
        cu.loc[cu.last_valid_index() - interval: ,  currency].plot()
        plt.tight_layout()
        plt.title(str(currency) + '        ' +str(cu.loc[cu.last_valid_index(),  currency].round(5)))
        plt.yticks(ticks)
        plt.grid(which='both')
        plt.show()  
        
    # Print prices and granularity
    print()
    print(granularity)
    print()
    print(cu.loc[cu.last_valid_index(), 'timestamp'])
    print()
    for each in cu.iloc[cu.last_valid_index(), 1:].astype(float).round(6).values:
        print(each)







###############################################################################
# Average position of all currency set for mulitple windows
###############################################################################    
if 0:
    
    


    windows = np.array([15, 30, 60, 90, 120])
    color_list = plt.cm.Blues(np.linspace(.4, .8, windows.shape[0]))
    
    for currency in currencies:
        slopes_mean = pd.DataFrame()
        position_mean = pd.DataFrame()
        for window in windows:
            win = np.array([window])
        
            # Get Insturment List and which direction to align instrument
            pair_list = []
            shape_list = []
            for pair in ratios.columns:
                if currency.upper() in pair.split('_'):
                    pair_list.append(pair)
                    if currency.upper() == pair.split('_')[0]:
                        shape_list.append(1)
                    else:
                        shape_list.append(-1)
        
            # Get Slope position for all values (instruments)
            positions = pd.DataFrame()
            deviations = pd.DataFrame()
            slopes = pd.DataFrame()
            for i in range(len(pair_list)):
                instrument = pair_list[i]
                shape = shape_list[i]
                if shape == 1:
                    values = ratios.loc[start: end, instrument]
                else:
                    values = (ratios.loc[start: end, instrument] * -1)\
                           + (2 * ratios.loc[start: end,  instrument].values[-1])
                pos = get_channel_mean_pos_std(values.values, win)
                positions[instrument] = pos['pos'].values.ravel()
                deviations[instrument] = pos['std'].values.ravel()
                slopes[instrument] = pos['slope'].values.ravel()
            slopes_mean[win[0]] = slopes.mean(axis=1)
            position_mean[win[0]] = positions.mean(axis=1)
            
        plt_index = np.arange(start, end + 1)
        position_mean.index = plt_index
        slopes_mean.index = plt_index          
          
        position_mean.plot(colors=color_list)
        plt.plot(plt_index, np.ones(plt_index.shape[0]) * 2, color='grey')
        plt.plot(plt_index, np.ones(plt_index.shape[0]) * -2, color='grey')
        plt.plot(plt_index, np.ones(plt_index.shape[0]) * 0, color='grey')
        plt.tight_layout()
        plt.title('Mean of Currency Set Channel Positions on Mulitple Windows')
        slopes_mean.plot(colors=color_list)
        plt.plot(plt_index, np.ones(plt_index.shape[0]) * 0, color='grey')
        plt.tight_layout()
        plt.title('Mean of Currency Set Slopes on Mulitple Windows')
        plt.show()



       
 
