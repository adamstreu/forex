import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import normaltest
import os; os.chdir('/northbend')
from classes.channel import Channel

'''
Assumes the graphs have already been created from other module
'''
        
# Plot Currency Universe
def plot_currency_universe(cu, plot_index, currencies, ratios, interval):
    
    ####### Redraw Plots  ######

    # Plots for the currency universe
    for currency in currencies:
        fig = plt.figure(str(currency) + '_set', clear=True, tight_layout=True,
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
        


    ####### Plot Intruments  ######
    for currency in currencies:        
        fig = plt.figure(str(currency) + '_set')
        ax = plt.figure(str(currency) + '_set').get_axes()

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
                    a.plot(plot_index, channel.line, color='lightgrey')
                    a.plot(plot_index, channel.line + (np.ones(plot_index.shape[0]) * (2 * channel.channel_deviation)), color='black')
                    a.plot(plot_index, channel.line + (np.ones(plot_index.shape[0]) * (-2 * channel.channel_deviation)), color='black')
                    a.plot(plot_index, channel.line, color='grey')
           
            
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
        
    
    


          