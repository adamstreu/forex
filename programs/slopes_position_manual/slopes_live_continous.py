###############################################################################
# Notes
###############################################################################
if True:

    '''
    WITH NEW ANIMATION CAPABILITIES
        run a few (of smae type) with diff granularities
        Run a volume and diff chart on small granularity and timeframe
    
    
        
    TODAYS NOTES:
        
        currencies:
            crafeully to match calcualtors with graph lines
        
        
        add more note area to calculator - don;t need all the misc info.
        
        
        
        
        
        Mode GRaphs:
            
            A Full (summary)sheet:
                of shifted currencies ( and perhaps for different intervals ( close and far))
                maybe correlation
                
            Plot channels on the currencies
    

        Process per currency ( or two)
            then just adjsut while running.
            each currency si watched, etc.
            


        Remeber - we are not looking for where it's going to go next.
            We want to know where it will be in an hour or two.
            
        could try a smaller timeframe - 5 - 10 minutes.  MAYBE.\
            yeah.  5 minute could be good.  
            8:30 am and we are seeing prices move then channels change

        DIDN'T HAVE SPREAD OR PRICING IN THERE !  Print that shit
        
        Experiment with diffferne talpha bchannel break points
        
        Maybe.
            Multi processirs or maybe cron jobs
            or maybe gui with window style tabs
            or maybe saved as png then in browser with auto refresh and mutliprocessor and on small granularity ( but with larger intervals shown as well)
            
        a smaller and a larger scale shown
        
        We Do want to catch the moves BEFORE they happen.
        When they appear to be starting a new channel but before it clicks over.
        There is risk in this but high rewards as well.
        Need to learn to play it well.
        Perhaps by spreading the risk and taking small losses for the larger wins.

        
        
    Misc:
            
        Ideas still to do:
            Streaming - watch for one currency taking off
            volume
            look through old chart for other indicators - maybe.....


    
    NOTE TO SELF:

        Be Prepared for Monday - record suggested changes througout day
            don;t waste day making new ones while in session (before 11)
        
        Did a bad job of watching the market today.  Something I 
        nee to learn to do well.  Should be able to nurture a feel for 
        where the market is for every relevant instrument / currency.
        
        This is early and regular morning work.  Long term I quit after  Spread is
        too high.
        
        Need to track carefully, remember well, be focused and log shit
        and place things in correct locations, unhurried, with limits, 
        with stops and targets.
        
        Like things really begin to move 9:00 new york.
        
        




    Method 1: the real method:
        Working within a few hours.
        Wait for a currency to have a 'stable pattern'
        Wait for another currency to have a stable pattern similar but opposite.
        Bet on it.  That's it.  Using slope and channel position of two currencies.
        
        We are waiting my the pattern to be on the outskirt of the distribution 
        then betting on it returning.




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
    import matplotlib.gridspec as gridspec
    from time import sleep
    import os; os.chdir('/northbend')
    from programs.slopes_position_manual.plot_currency_universe import plot_currency_universe    
    from programs.slopes_position_manual.plot_currency_indicators import plot_currency_indicators 
    from libraries.currency_universe import backfill_with_singular
    from libraries.currency_universe import get_universe_singular
    from libraries.oanda import market
    from libraries.oanda import get_time
    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15) 
    np.warnings.filterwarnings('ignore')
    plt.rcParams['figure.figsize'] = [14, 4]
    
   

###############################################################################
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
    currencies = ['eur']
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'hkd', 'jpy', 'nzd', 'usd']     
    
    # Solo interval for graphing
    interval = 120
    
    # Windows and calculation interval
    windows = np.array([15, 30, 60, 90, 120])
    
    # Create Indexes
    indicator_index = np.arange(ratios.last_valid_index() - (interval * 2 + 10),
                                ratios.last_valid_index() + 1)
    plot_index = np.arange(ratios.last_valid_index() - (interval - 1),
                                ratios.last_valid_index() + 1)
    
    # Plot window av slopes and position
    color_list = plt.cm.Blues(np.linspace(.5, 1, windows.shape[0]))[::-1]
    


###############################################################################
# Ready Plots
###############################################################################    
if 0:     
    
    
    # Plots for each currency indicators
    for currency in currencies:
        fig = plt.figure(currency, clear=True, tight_layout=True,
                         facecolor='grey')
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0, hspace=0) 
        # Arange subplots sizing        
        ax1  = plt.subplot(gs[0, :])
        ax8  = plt.subplot(gs[1, :], sharex=ax1)
        ax9  = plt.subplot(gs[2, :], sharex=ax1)
        ax10 = plt.subplot(gs[3, :], sharex=ax1)
        ax2 = plt.subplot(gs[4, 0])
        ax3 = plt.subplot(gs[4, 1], sharey=ax2)        
        ax4 = plt.subplot(gs[4, 2], sharey=ax2)
        ax5 = plt.subplot(gs[4, 3], sharey=ax2)
        ax6 = plt.subplot(gs[4, 4], sharey=ax2)
        # Axis stuff
        plt.setp(ax3.get_yticklabels(), visible=False)
        plt.setp(ax4.get_yticklabels(), visible=False)
        plt.setp(ax5.get_yticklabels(), visible=False)
        plt.setp(ax6.get_yticklabels(), visible=False)   
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax8.get_xticklabels(), visible=False)
        plt.setp(ax9.get_xticklabels(), visible=False) 
        plt.setp(ax10.get_xticklabels(), visible=False) 
        ax1.set_facecolor('xkcd:pale grey')
        ax2.set_facecolor('xkcd:pale grey')
        ax3.set_facecolor('xkcd:pale grey')
        ax4.set_facecolor('xkcd:pale grey')
        ax5.set_facecolor('xkcd:pale grey')
        ax6.set_facecolor('xkcd:pale grey')
        ax8.set_facecolor('xkcd:pale grey')
        ax9.set_facecolor('xkcd:pale grey')
        ax10.set_facecolor('xkcd:pale grey')
    
    
    # Plots for the currency universe
    for currency in currencies:
        plt.figure(str(currency) + '_set', clear=True, tight_layout=True,
                         facecolor='grey', edgecolor='black')
        gs = gridspec.GridSpec(len(currencies)-1, 1)
        gs.update(wspace=0, hspace=0) 
        
        # First plot for x ticks
        a = plt.subplot(gs[0, :])
        a.set_facecolor('xkcd:pale grey')
        a.spines['bottom'].set_linewidth(2)
        a.spines['left'].set_linewidth(2)
        a.spines['top'].set_linewidth(2)
        a.spines['right'].set_linewidth(2)
        #plt.setp(a.get_xticklabels(), visible=True)
        # Axis stuff
        for c in range(1, len(currencies)-1):
            b = plt.subplot(gs[c, :], sharex=a)
            #plt.setp(b.get_xticklabels(), visible=False)
            b.set_facecolor('xkcd:pale grey')
            b.spines['top'].set_visible(True)
            b.spines['right'].set_visible(True)
            b.spines['bottom'].set_linewidth(2)
            b.spines['left'].set_linewidth(2)
            b.spines['top'].set_linewidth(2)
            b.spines['right'].set_linewidth(2)

              
###############################################################################
# Call Graphs for first time.  Clean up plots.
###############################################################################    
if 1:             
    
    # Call Graphs
    plot_currency_universe(cu.copy(), plot_index, currencies, ratios, interval)
    plot_currency_indicators(currencies, cu.copy(), ratios.copy(), plot_index, 
                             indicator_index, interval, windows, color_list)

    # Update Graphs
    for currency in currencies:
        ax = plt.figure(str(currency)+'_set').get_axes()
        a = ax[0]
        plt.setp(a.get_xticklabels(), visible=True)
        plt.subplots_adjust(wspace=.0010, hspace=.001)
   
        
        
###############################################################################
# Update All Info in real time
###############################################################################
if 1:
    
    # Parameters
    pause_time = 15
    timestamp =  cu.loc[cu.last_valid_index(), 'timestamp']
    
    # Look for new candle every _x_ seconds.  Update all info when found
    while True:
        # Do we have new information ?
        new_time = get_time(granularity)
        if new_time > timestamp:
            timestamp = new_time
            print('Candle Found at:\t' + str(timestamp))
            # Update ratios and cu with new data
            a, b = get_universe_singular(currencies, granularity)
            # Add line to cur
            a['timestamp'] = timestamp
            cu = cu.append(a, ignore_index=True, verify_integrity=True)
            # Add to ratio
            b['timestamp'] = timestamp
            ratios = ratios.append(b, ignore_index=True, verify_integrity=True)
            
            indicator_index = np.arange(ratios.last_valid_index() - (interval * 2 + 10),
                                ratios.last_valid_index() + 1)
            plot_index = np.arange(ratios.last_valid_index() - (interval - 1),
                                ratios.last_valid_index() + 1)

            # Regraph Indicators and CUrrency Sets
            plot_currency_universe(cu.copy(), plot_index, currencies, ratios.copy(), interval)
            plot_currency_indicators(currencies, cu.copy(), ratios.copy(), plot_index, 
                                     indicator_index, interval, windows, color_list)

            # Update Graphs
            for currency in currencies:
                ax = plt.figure(str(currency)+'_set').get_axes()
                a = ax[0]
                plt.setp(a.get_xticklabels(), visible=True)
                plt.subplots_adjust(wspace=.0010, hspace=.001)
                plt.pause(.1)
                
            for currency in currencies:
                ax = plt.figure(currency).get_axes()
                a = ax[0]
                plt.pause(.1)
        else:
            print('{}: Waiting on new candle from {}'.format(str(new_time),
                                                                    str(timestamp)))
        sleep(pause_time)
                        



        ''' NO tsure if needed          
        for currency in currencies:
            ax = plt.figure(str(currency)+'_set').get_axes()
            a = ax[0]
            plt.setp(a.get_xticklabels(), visible=True)
            plt.subplots_adjust(wspace=.0010, hspace=.001)
            fig  = plt.figure(str(currency)+'_set')
            fig.tight_layout()
        '''
 

    
    

        

        
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 



















































    






