# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from matplotlib import animation    
import os; os.chdir('/northbend')
from classes.channel import Channel 
from libraries.oanda import market
from libraries.currency_universe import backfill_with_singular



granularity = 'M1'
# Get cu file from Desktop
path = '/northbend/tmp/currencies_'
file = path + granularity
file += '.pkl'
# Solo interval for graphing
interval = 120
# Normal Test
normaltest_alpha = 1e-3
normaltest_step = 10
# Refresh
refresh_interval = 7000
hours_subtraction_from_utc = 6 # Chicago



###############################################################################
# Set Parameters
###############################################################################    
if 1:     

    # Only use these currencies for placements (high leverage)   
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 
                  'hkd', 'jpy', 'nzd', 'usd']     
    
    # General Parameters
    _from = '2019-01-01T00:00:00Z'
    _to   = '2020-01-01T00:00:00Z'
    granularities = ['M1', 'M5', 'M10', 'M15']
    update_granularity = 'M1'
    
    # Get cu file from Desktop
    path = '/northbend/tmp/currencies_'

    # Pause timer
    pause_time = 8

###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   
    

    # Get instrument List
    instrument_list = []
    for mark in market:
        if mark.split('_')[0] in [x.upper() for x in currencies]: 
            if mark.split('_')[1] in [x.upper() for x in currencies]:
                instrument_list.append(mark)    
    
    # Start with Data and Ratios Backfilled
    cu = pd.DataFrame(columns = currencies)
    cu, ratios = backfill_with_singular(currencies, granularity, 
                                        _from, _to)
    cu.index.names = ['timestamp']
    ratios.index.names = ['timestamp']
    cu.reset_index(inplace=True)
    ratios.reset_index(inplace=True)
    ratios.iloc[:, 1] = ratios.iloc[:, 1:].astype(float)
    cu.iloc[:, 1] = cu.iloc[:, 1:].astype(float)

    df = cu.copy()
    

###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 0:   

    currency = 'aud'
    
    plot_index = np.arange(df.last_valid_index() - interval, 
                           df.last_valid_index() + 1)

    # Draw figure
    df = pd.read_pickle(file)
    fig, ax = plt.subplots(df.shape[1] - 1, 1, sharex=True, num=granularity,
                           facecolor='#343837')
    plt.subplots_adjust(hspace=.1, wspace=.05, bottom=.05, top=.98, 
                        right=.9, left=.05)
   
    
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
    currency_set = pd.DataFrame()
    for i in range(len(pair_list)):
        instrument = pair_list[i]
        shape = shape_list[i]
        if shape == 1:
            values = ratios.loc[plot_index, instrument]
        else:
            values = (ratios.loc[plot_index, instrument] * -1)\
                   + (2 * ratios.loc[plot_index,  instrument].values[-1])
        currency_set[pair_list[i]] = values
        
    
    ####### Get Channels with standard devaiton distributions ######
    for c in range(currency_set.shape[1]):
    
        step = 10
        coll = []
        for i in range(10, interval + step, step):
            channel = Channel(currency_set.iloc[:, c].values)
            test = channel.flattened / channel.channel_deviation
            k2, p = normaltest(test)
            alpha = 1e-3
            #print("p = {:g}".format(p))
            if p < alpha:
                pass #print(str(i) + , "The null hypothesis can be rejected")
            else:
                coll.append(i) #print(str(i))#  + "The null hypothesis cannot be rejected")
        coll = np.array(coll)
    
        keep = []
        for i in range(coll.shape[0] - 1):
            if coll[i + 1] > coll[i] + step:
                keep.append(coll[i])
        if coll.shape[0] > 0: keep.append(coll[-1])
        keep = np.array(keep)
    
        
    
        # Plot currencies
        a = ax[c]
        currency_set.iloc[:, c].plot(ax=a, legend=False, linewidth=2.5)
        
        
        # Get ticks by pip for currencies
        max_ticks = currency_set.iloc[:, c].values.max()
        min_ticks = currency_set.iloc[:, c].values.min()
        if currency == 'jpy':
            ticks = np.arange(min_ticks, max_ticks, .1).round(6)
        elif currency == 'hkd':
            ticks = np.arange(min_ticks, max_ticks, .01).round(6)
        else:
            if 'JPY' in currency_set.columns[c].split('_'):
                ticks = np.arange(min_ticks, max_ticks, .1).round(6)
            elif 'HKD' in currency_set.columns[c].split('_'):
                ticks = np.arange(min_ticks, max_ticks, .01).round(6)
            else:
                ticks = np.arange(min_ticks, max_ticks, .001).round(4)
        
        # Plot each normal section on top with channel lines]
        if keep.shape[0] > 0:
            for k in keep[::-1]:
                a.plot(plot_index, channel.line + (np.ones(plot_index.shape[0]) * (2 * channel.channel_deviation)), color='black')
                a.plot(plot_index, channel.line + (np.ones(plot_index.shape[0]) * (-2 * channel.channel_deviation)), color='black')
       
        
        a.set_yticks(ticks)
        a.grid(which='both')
    
    ####### Finalize Plot ######
    for i in range(currency_set.shape[1]):
        ax[i].set_ylabel(currency_set.columns[i], rotation=90, size='large')
        ax[i].yaxis.tick_right()
    #plt.pause(.01)
    
        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
    '''
    for currency in currencies:
        # Set ax and figure
        ax = plt.figure(str(currency) + '_set').get_axes()
        
        # Get ticks by pip for currencies
        max_ticks = cu.loc[plot_index, currency].values.max()
        min_ticks = cu.loc[plot_index, currency].values.min()
        if currency == 'hkd':
            ticks = np.arange(min_ticks, max_ticks, .00001).round(6)
        elif currency == 'jpy':
            ticks = np.arange(min_ticks, max_ticks, .000005).round(6)
        else:
            ticks = np.arange(min_ticks, max_ticks, .0001).round(4)
    
        
        ####### Get Channels with standard devaiton distributions ######
    
        step = 10
        coll = []
        for i in range(10, interval + step, step):
            channel = Channel(cu.loc[cu.last_valid_index() - i: \
                                     cu.last_valid_index(), currency].values)
            k2, p = normaltest(channel.flattened)
            alpha = 1e-3
            #print("p = {:g}".format(p))
            if p < alpha:
                pass #print(str(i) + , "The null hypothesis can be rejected")
            else:
                coll.append(i) #print(str(i))#  + "The null hypothesis cannot be rejected")
        coll = np.array(coll)
    
        keep = []
        for i in range(coll.shape[0] - 1):
            if coll[i + 1] > coll[i] + step:
                keep.append(coll[i])
        if coll.shape[0] > 0: keep.append(coll[-1])
        keep = np.array(keep)
        
    
        # Plot currencies
        a = ax[0]
        cu.loc[plot_index, currency].plot(ax=a)
    
        # Plot each normal section on top
        for k in keep[::-1]:
            channel = Channel(cu.loc[cu.last_valid_index() - k:, currency].values)
            cu.loc[cu.last_valid_index() - k:, currency].plot(ax=a)
            a.plot(np.arange(cu.last_valid_index() - (k + 1) , cu.last_valid_index()),
                     channel.line, color='grey')
            cu.loc[cu.last_valid_index() - k:, currency].plot(ax=a, marker='.')
            
        plt.setp(a.get_xticklabels(), visible=True)
        a.set_title('Currency Price')
        a.set_yticks(ticks)
        a.grid(which='both')
    '''
        
    
    


          
