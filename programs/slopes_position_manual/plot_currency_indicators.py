import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import normaltest
import multiprocessing 
import os; os.chdir('/northbend')
from classes.channel import Channel
from libraries.indicators import get_channel_mean_pos_std

'''
Assumes figures for plotting have already been made and 
named according to currencies
'''

def plot_currency_indicators(currencies, cu, ratios, plot_index, 
                             indicator_index, interval, windows, color_list):
    
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
    
    
    ###############################################################################
    # Plot currencies.  Color by Standard Normal organizatio.  With regression line.
    ###############################################################################    
    if 1:      
        
        
        
        for currency in currencies:
            
            # Set ax and figure
            ax = plt.figure(currency).get_axes()
            
            # Get ticks by pip for currencies
            max_ticks = cu.loc[plot_index, currency].values.max()
            min_ticks = cu.loc[plot_index, currency].values.min()
            if currency == 'hkd':
                ticks = np.arange(min_ticks, max_ticks, .00001).round(6)
            elif currency == 'jpy':
                ticks = np.arange(min_ticks, max_ticks, .000005).round(6)
            else:
                ticks = np.arange(min_ticks, max_ticks, .0001).round(4)
    
            
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
            # cu.loc[plot_index, currency].plot(ax=ax[0], color='blue', marker='+')
    
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
    
    
    print(currency)
    
    
    
    ###############################################################################
    # Graph shifted Currency Sets.  Align inverse positions
    ###############################################################################    
    if 1:
        for currency in currencies:
            
            # Set ax and figure
            ax = plt.figure(currency).get_axes()
            
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
            has_jpy = pd.DataFrame()
            for i in range(len(pair_list)):
                instrument = pair_list[i]
                shape = shape_list[i]
                if shape == 1:
                    values = ratios.loc[plot_index, instrument]
                else:
                    values = (ratios.loc[plot_index, instrument] * -1)\
                           + (2 * ratios.loc[plot_index,  instrument].values[-1])
                # Don't include jpy, hkd inside shifted currency sets for others
                if currency != 'JPY' and currency != 'HKD':
                    if 'JPY' in instrument.split('_') or 'HKD' in instrument.split('_'):
                        has_jpy[pair_list[i]] = values
                    else:    
                        currency_set[pair_list[i]] = values
                else:
                        currency_set[pair_list[i]] = values
                    
            try:
                # Plot Values
                a = ax[3]
                (currency_set - currency_set.loc[currency_set.first_valid_index()]).plot(ax=a)
                a.plot(plot_index, np.ones(plot_index.shape[0]) * 0, color='grey')
                a.set_title('Shifted Currency Set - excluding JPY and HKD')
            except:
                pass
                print('had nothing for jpy, hkd - see line 118, plot_indicaotrs')
    
    
    
    
    ###############################################################################
    # Plot currency set average positions and slopes over multiple windows
    ###############################################################################    
    if 1:             
        
        for currency in currencies:
            
            # Set ax and figure
            ax = plt.figure(currency).get_axes()
            
            slopes_mean = pd.DataFrame()
            position_mean = pd.DataFrame()
            for w in range(windows.shape[0]):
                win = np.array([windows[w]])
            
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
                        values = ratios.loc[indicator_index, instrument]
                    else:
                        values = (ratios.loc[indicator_index, instrument] * -1)\
                               + (2 * ratios.loc[indicator_index,  instrument].values[-1])
                    pos = get_channel_mean_pos_std(values.values, win)
                    positions[instrument] = pos['pos'].values.ravel()
                    deviations[instrument] = pos['std'].values.ravel()
                    slopes[instrument] = pos['slope'].values.ravel()
                
                # Arange Index to match currency locations
                slopes_mean[win[0]] = slopes.mean(axis=1)
                position_mean[win[0]] = positions.mean(axis=1)
                
                # Plot positions for each window
                end_values = -15
                positions.index = indicator_index
                a = ax[w + 4]
                positions.loc[plot_index[end_values:]].plot(ax=a, legend=False)
                positions.loc[plot_index[end_values:]].mean(axis=1).plot(ax=a, color='black', legend=False)                
                a.plot(plot_index[end_values:], np.ones(plot_index.shape[0])[end_values:] * 2, color='grey')
                a.plot(plot_index[end_values:], np.ones(plot_index.shape[0])[end_values:] * -2, color='grey')
                a.plot(plot_index[end_values:], np.ones(plot_index.shape[0])[end_values:] * 0, color='grey')
                a.set_title(windows[w])
    
            # Arange Index to match currency locations
            position_mean.index = indicator_index
            slopes_mean.index = indicator_index          
            
            # Plot currency set position average over mulitple windows
            a = ax[1]
            position_mean.loc[plot_index].plot(ax=a, colors=color_list, 
                                               legend=False)
            a.plot(plot_index, np.ones(plot_index.shape[0]) * 2, color='grey')
            a.plot(plot_index, np.ones(plot_index.shape[0]) * -2, color='grey')
            a.plot(plot_index, np.ones(plot_index.shape[0]) * 0, color='grey')
            a.set_title('Mean of Currency Set Channel Positions on Mulitple Windows')
            a.grid(axis='x')
            
            # Plot currency set Slope average over mulitple windows
            a = ax[2]
            slopes_mean.loc[plot_index].plot(ax=a, colors=color_list)
            a.plot(plot_index, np.ones(plot_index.shape[0]) * 0, color='grey')
            a.set_title('Mean of Currency Set Slopes on Mulitple Windows')
            a.grid(axis='x')


    #plt.pause(.01)





###############################################################################
# Multiprocess Test
###############################################################################    
if 0:             
    
    
    
  



    # Call cpus number, processes, join, wait to complete, etc.
    cpus = multiprocessing.cpu_count()
    jobs = [] 
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    # Run Jobs for first Rotations
    for w in range(len(currencies)):
        currency = currencies[w]
        fig = plt.figure(str(currency) + '_set')
        ax = plt.figure(str(currency) + '_set').get_axes()[0]
        p = multiprocessing.Process(target=test, 
                                    args=(currency,
                                          ax,
                                          cu.copy(),
                                          plot_index,
                                          return_dict))
        jobs.append(p)
        p.start()
        
        # Pause to join at number of cpus before moving on.
        if (w % (cpus-1) == 0 and w > 0) or (w == windows.shape[0] - 1):
            for job in jobs:
                job.join()

    print(return_dict.keys())
    for k in return_dict.keys():
        print(return_dict[k])
#        ax = plt.figure(str(k) + '_set').get_axes()[0]
#        ax[0] = return_dict[k]


    
