
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from matplotlib import animation 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os
os.chdir('/northbend')
from classes.channel import Channel 
    
'''
Which one is in front of which one
can we account for higher universe price ? 

claradfcl
Have to add diff.
Have to add to regular data feeder.


'''



# Uses interval and M1 to get corr between currencies.  We want uncorrelated.
# sns.clustermap(df.iloc[df.last_valid_index() - interval:, 1:].corr(), linecolor='black', linewidth=1.5)

# Granularities
granularity = 'M1'
granularities = ['M1', 'M5']#, 'M15', 'H1']
currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 
              'hkd', 'jpy', 'nzd', 'usd']     
currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'usd']     
currencies = ['aud', 'cad', 'nzd', 'chf', 'eur', 'usd', 'gbp']  
# Get cu file from Desktop
path = '/northbend/tmp/currencies_'
file = path + granularity
file += '.pkl'
ratios_path = '/northbend/tmp/ratios_'
ratios_file = ratios_path + granularity
ratios_file += '.pkl'
volume_file = '/northbend/tmp/volume.pkl'
cu_small_file = '/northbend/tmp/cu_small.pkl'

# Solo interval for graphing
interval = 120

# Normal Test
normaltest_alpha = 1e-2
normaltest_step = 10

# Refresh
refresh_interval = 5000
hours_subtraction_from_utc = 6 # Chicago


def supports(values, bins):
        
    # Get Supports
    hist = np.histogram(values, bins)
    hist_x = hist[1] + ((hist[1][1] - hist[1][0]) / 2)
    hist_x = hist_x[:-1]
    

    supports = []
    for i in range(1, len(hist[0][:-1])):
        if hist[0][i] > hist[0][i-1] & hist[0][i] > hist[0][i+1]:
            supports.append(hist_x[i])
    supports = np.array(supports)

    return supports



    


def volume(data, volume_file = volume_file, cu_small_file = cu_small_file,
           interval=interval,
           hours_subtraction_from_utc=hours_subtraction_from_utc): 
    
    df = pd.read_pickle(volume_file)
    df.drop('jpy', axis=1, inplace=True)
    df.drop('hkd', axis=1, inplace=True)    
    plot_index = np.arange(df.last_valid_index() - interval, 
                           df.last_valid_index() + 1)
    
    # Clear Plots
    ax.cla()

    # Plot Volume
    df.iloc[plot_index, 1:].plot(subplots=True, ax=ax)

    
    
    
def diff(data, volume_file = volume_file, cu_small_file = cu_small_file,
           interval=interval,
           hours_subtraction_from_utc=hours_subtraction_from_utc): 

    cu = pd.read_pickle(cu_small_file)
    plot_index = np.arange(cu.last_valid_index() - interval, 
                           cu.last_valid_index() + 1)
    
    # Clear Plots
    ax.cla()

    # Plot diff
    cu.drop('jpy', axis=1, inplace=True)
    cu.drop('hkd', axis=1, inplace=True)    
    diff = cu - cu.shift(1)
    diff.iloc[plot_index, 1:].plot(subplots=True, ax=ax)
    
    
    
    


def standardized(data, file=file, granularity = granularity, 
                 ratios_file=ratios_file, interval=interval, 
                 normaltest_step=normaltest_step,
                 normaltest_alpha=normaltest_alpha,
                 hours_subtraction_from_utc=hours_subtraction_from_utc): 
    
    ax.cla()
    # Import Data
    df = pd.read_pickle(str(file))
    
    plot_index = np.arange(df.last_valid_index() - interval, 
                           df.last_valid_index() + 1)
    
    ss = StandardScaler().fit_transform(df.iloc[:, 1:])
    ss = pd.DataFrame(ss, columns = df.columns[1:], index=df.index)
    ss.insert(0, 'timestamp', df.timestamp)
    # ss.drop('jpy', axis=1, inplace=True)
    ss.drop('hkd', axis=1, inplace=True)    
    
    ss.iloc[plot_index, 1:].plot(ax = ax)
    
    # Set plot x Time markers
    x_ticks_count = plt.xticks()[0].shape[0]
    x_tick_locations = np.linspace(plot_index[0], plot_index[-1], x_ticks_count).astype(int)
    x_ticks_markers = df.loc[x_tick_locations, 'timestamp']
    x_ticks_markers = x_ticks_markers.apply(lambda x: x - pd.Timedelta(hours=hours_subtraction_from_utc)).dt.time
    x_ticks_markers = x_ticks_markers.astype(str).apply(lambda x: x[:5]).values.astype(str)
    
    ax.set_xticks(x_tick_locations)
    ax.set_xticklabels(x_ticks_markers, color='white', rotation=270)
    ax.grid(which='major', axis='x', linestyle= '-',linewidth=.2,
                               color='#343837', b=True) 
              
    '''                 
    # Plot a Linear fit to each line
    for col in ss.columns[1:]:
        x = plot_index.reshape(-1,1)
        y = ss.loc[plot_index, col]
        linfit = LinearRegression().fit(x,y)
        y = linfit.predict(plot_index.reshape(-1,1))
        plt.plot(x, y, color='grey')
    ''' 


def run_all(data, file=file, currencies = currencies, ratios_file=ratios_file, interval=interval, 
            normaltest_step=normaltest_step,
            normaltest_alpha=normaltest_alpha,
            hours_subtraction_from_utc=hours_subtraction_from_utc):
    
    print(ratios_file)

    ratios = pd.read_pickle(ratios_file)    
    ratios_drop = ratios.filter(regex='JPY').columns.tolist() \
                + ratios.filter(regex='HKD').columns.tolist()
    ratios = ratios.drop(ratios_drop, axis=1)
    print(ratios.columns)
    
    plot_index = np.arange(ratios.last_valid_index() - interval, 
                           ratios.last_valid_index() + 1)   
    
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
            
        order_list = []
        for pair in pair_list:
            find = pair.split('_')
            find.remove(currency.upper())
            order_list.append(currencies.index(find[0].lower()))
            
        # Assign Column to currency
        ax_col = currencies.index(currency)
        
        # Plot all currency Pairs and Channels
        for i in range(len(order_list)):
            p = pair_list[i]
            o = order_list[i]
            ax[o, ax_col].cla()
            currency_set[p].plot(ax=ax[o, ax_col])
            
    
            # get channels for currency
            coll = []
            for s in range(20, interval + normaltest_step, normaltest_step):
                channel = Channel(currency_set[p].values)
                k2, _p = normaltest(channel.flattened / channel.channel_deviation)
                if _p < normaltest_alpha:
                    pass # "The null hypothesis can be rejected")
                else:
                    coll.append(s) # "The null hypothesis cannot be rejected")
            coll = np.array(coll)
       
            # Get Positions for channel breaks
            keep = []
            for k in range(coll.shape[0] - 1):
                if coll[k + 1] > coll[k] + normaltest_step:
                    keep.append(coll[k])
            if coll.shape[0] > 0: keep.append(coll[-1])
            keep = np.array(keep)
        
            
            # Plot chnnels by normal break
            for k in keep[::-1]:
                channel = Channel(currency_set.loc[currency_set.last_valid_index() - s: \
                                         currency_set.last_valid_index(), p].values)
                currency_set.loc[currency_set.last_valid_index() - (k + 1):, p]\
                  .plot(ax=ax[o, ax_col], marker='|', markersize=.5)
                line = channel.line
                line_x = np.arange(currency_set.last_valid_index() - (k), 
                                   currency_set.last_valid_index() + 1)
                line_y_up = line + (np.ones(line.shape[0]) * 2 \
                                    * channel.channel_deviation)
                line_y_down = line + (np.ones(line.shape[0]) * -2 \
                                      * channel.channel_deviation)
                #axr[i-1].plot(line_x, line, color='grey', linewidth=.5)
                ax[o, ax_col].plot(line_x, line_y_up, 
                                     color='black', linewidth=.75)
                ax[o, ax_col].plot(line_x, line_y_down, 
                                     color='black', linewidth=.75)
    
    # Add Row Labels to first column
    for i in range(df.columns.shape[0] - 1):
        ax[i, 0].set_ylabel(currencies[i].upper(), rotation=90, size='large', color='white')
        
    # Add Columns Labels to first Row
    for i in range(df.columns.shape[0] - 1):
        ax[0, i].set_title(currencies[i].upper(), size='large', color='white')
    
    # Set plot x Time markers
    x_ticks_count = plt.xticks()[0].shape[0]
    x_tick_locations = np.linspace(plot_index[0], plot_index[-1], x_ticks_count).astype(int)
    x_ticks_markers = ratios.loc[x_tick_locations, 'timestamp']
    x_ticks_markers = x_ticks_markers.apply(lambda x: x - pd.Timedelta(hours=hours_subtraction_from_utc)).dt.time
    x_ticks_markers = x_ticks_markers.astype(str).apply(lambda x: x[:5]).values.astype(str)
    
    # Set x Grid
    for a in ax.ravel():
        a.set_xticks(x_tick_locations)
        a.set_xticklabels(x_ticks_markers, color='white', rotation=270)
        a.grid(which='major', axis='x', linestyle= '-',linewidth=.2,
                                   color='#343837', b=True) 
            
    ## Set Y Grid
    #for i in range(1, df.columns.shape[0]):    
    #    max_ticks = df.iloc[plot_index, i].max()
    #    min_ticks = df.iloc[plot_index, i].min()
    #    # Different ticks for jpy and hkd
    #    if df.columns[i] == 'jpy':
    #        ticks = np.arange(min_ticks, max_ticks, .000001).round(6)
    #    elif df.columns[i] == 'hkd':
    #        ticks = np.arange(min_ticks, max_ticks, .00001).round(5)
    #    else:
    #        ticks = np.arange(min_ticks, max_ticks, .0001).round(4)
    #        # Create ticks 
    #    axr[i - 1].grid(linestyle= '-', axis='y', linewidth=.2, 
    #                               color='#343837', b=True)    
    #    axr[i - 1].set_yticks(ticks)
    
    # Remove Y Ticks
    for a in ax.ravel():
        a.set_yticklabels([])
    
    # Print Recent Location
        print(df.last_valid_index())

    
    
    


def run_all_for_two(data, file=file, currencies = currencies, ratios_file=ratios_file, interval=interval, 
            normaltest_step=normaltest_step,
            normaltest_alpha=normaltest_alpha,
            hours_subtraction_from_utc=hours_subtraction_from_utc):
    
    cur1 = 'chf'
    cur2 = 'eur'
    cur3 = 'usd'
    print(ratios_file)

    ratios = pd.read_pickle(ratios_file)    
    ratios_drop = ratios.filter(regex='JPY').columns.tolist() \
                + ratios.filter(regex='HKD').columns.tolist()
    ratios = ratios.drop(ratios_drop, axis=1)
    print(ratios.columns)
    
    plot_index = np.arange(ratios.last_valid_index() - interval, 
                           ratios.last_valid_index() + 1)   
    
    for currency in [cur1, cur2, cur3]: 
    
        if currency == cur1:
            ax_col = 0
        elif currency == cur2:
            ax_col = 1
        else:
            ax_col = 2
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
            
        order_list = []
        for pair in pair_list:
            find = pair.split('_')
            find.remove(currency.upper())
            order_list.append(currencies.index(find[0].lower()))
            
        
        # Plot all currency Pairs and Channels
        for i in range(len(order_list)):
            p = pair_list[i]
            o = order_list[i]
            ax[o, ax_col].cla()
            currency_set[p].plot(ax=ax[o, ax_col])
            
            '''
            # get channels for currency
            coll = []
            for s in range(20, interval + normaltest_step, normaltest_step):
                channel = Channel(currency_set[p].values)
                k2, _p = normaltest(channel.flattened / channel.channel_deviation)
                if _p < normaltest_alpha:
                    pass # "The null hypothesis can be rejected")
                else:
                    coll.append(s) # "The null hypothesis cannot be rejected")
            coll = np.array(coll)
       
            # Get Positions for channel breaks
            keep = []
            for k in range(coll.shape[0] - 1):
                if coll[k + 1] > coll[k] + normaltest_step:
                    keep.append(coll[k])
            if coll.shape[0] > 0: keep.append(coll[-1])
            keep = np.array(keep)
        
            
            # Plot chnnels by normal break
            for k in keep[::-1]:
                channel = Channel(currency_set.loc[currency_set.last_valid_index() - s: \
                                         currency_set.last_valid_index(), p].values)
                currency_set.loc[currency_set.last_valid_index() - (k + 1):, p]\
                  .plot(ax=ax[o, ax_col], marker='|', markersize=.5)
                line = channel.line
                line_x = np.arange(currency_set.last_valid_index() - (k), 
                                   currency_set.last_valid_index() + 1)
                line_y_up = line + (np.ones(line.shape[0]) * 2 \
                                    * channel.channel_deviation)
                line_y_down = line + (np.ones(line.shape[0]) * -2 \
                                      * channel.channel_deviation)
                #axr[i-1].plot(line_x, line, color='grey', linewidth=.5)
                ax[o, ax_col].plot(line_x, line_y_up, 
                                     color='black', linewidth=.75)
                ax[o, ax_col].plot(line_x, line_y_down, 
                                     color='black', linewidth=.75)
            '''
    
    # Add Row Labels to first column
    for i in range(df.columns.shape[0] - 1):
        ax[i, 0].set_ylabel(currencies[i].upper(), rotation=90, size='large', color='white')
        
    # Add Columns Labels to first Row
    ax[0, 0].set_title(cur1.upper(), size='large', color='white')
    ax[0, 1].set_title(cur2.upper(), size='large', color='white')
    ax[0, 2].set_title(cur3.upper(), size='large', color='white')
    
    # Set plot x Time markers
    x_ticks_count = plt.xticks()[0].shape[0]
    x_tick_locations = np.linspace(plot_index[0], plot_index[-1], x_ticks_count).astype(int)
    x_ticks_markers = ratios.loc[x_tick_locations, 'timestamp']
    x_ticks_markers = x_ticks_markers.apply(lambda x: x - pd.Timedelta(hours=hours_subtraction_from_utc)).dt.time
    x_ticks_markers = x_ticks_markers.astype(str).apply(lambda x: x[:5]).values.astype(str)
    
    # Set x Grid
    for a in ax.ravel():
        a.set_xticks(x_tick_locations)
        a.set_xticklabels(x_ticks_markers, color='white', rotation=270)
        a.grid(which='major', axis='x', linestyle= '-',linewidth=.2,
                                   color='#343837', b=True) 
        a.yaxis.tick_right()
            
    ## Set Y Grid
    #for i in range(1, df.columns.shape[0]):    
    #    max_ticks = df.iloc[plot_index, i].max()
    #    min_ticks = df.iloc[plot_index, i].min()
    #    # Different ticks for jpy and hkd
    #    if df.columns[i] == 'jpy':
    #        ticks = np.arange(min_ticks, max_ticks, .000001).round(6)
    #    elif df.columns[i] == 'hkd':
    #        ticks = np.arange(min_ticks, max_ticks, .00001).round(5)
    #    else:
    #        ticks = np.arange(min_ticks, max_ticks, .0001).round(4)
    #        # Create ticks 
    #    axr[i - 1].grid(linestyle= '-', axis='y', linewidth=.2, 
    #                               color='#343837', b=True)    
    #    axr[i - 1].set_yticks(ticks)
    '''
    # Remove Y Ticks
    for a in ax.ravel():
        a.set_yticklabels([])
    '''
    # Print Recent Location
    print(df.last_valid_index())

    
    




















def run_2_column(data, granularity, file=file, interval=interval, 
        normaltest_step=normaltest_step,
        normaltest_alpha=normaltest_alpha,
        hours_subtraction_from_utc=hours_subtraction_from_utc):
    
    try:
        # Import Data
        df = pd.read_pickle(str(file))
        df.drop('jpy', axis=1, inplace=True)
        df.drop('hkd', axis=1, inplace=True)
        df.drop('gbp', axis=1, inplace=True)
        plot_index = np.arange(df.last_valid_index() - interval, 
                               df.last_valid_index() + 1)
        
        axr = ax.ravel()
        
        # First and second columns
        first = np.arange(0, df.shape[1], 2)
        second = np.arange(1, df.shape[1], 2)
        
        # Plot Data
        for i in range(df.columns.shape[0] - 1):
            axr[i].cla()
            df.iloc[plot_index, 1+i].plot(ax=axr[i], linewidth=2)
        axr[-1].cla()
        
        
        # get channels for currency
        for i in range(1, df.columns.shape[0]):
            coll = []
            currency = df.columns[i]
            for s in range(20, interval + normaltest_step, normaltest_step):
                channel = Channel(df.loc[df.last_valid_index() - s: \
                                         df.last_valid_index(), currency].values)
                k2, p = normaltest(channel.flattened / channel.channel_deviation)
                #print("p = {:g}".format(p))
                if p < normaltest_alpha:
                    pass # "The null hypothesis can be rejected")
                else:
                    coll.append(s) # "The null hypothesis cannot be rejected")
            coll = np.array(coll)
       
           
            # Get Positions for channel breaks
            keep = []
            for k in range(coll.shape[0] - 1):
                if coll[k + 1] > coll[k] + normaltest_step:
                    keep.append(coll[k])
            if coll.shape[0] > 0: keep.append(coll[-1])
            keep = np.array(keep)
        
            
            # Plot chnnels by normal break
            for k in keep[::-1]:
                channel = Channel(df.loc[df.last_valid_index() - k:, 
                                         currency].values)
                df.loc[df.last_valid_index() - (k + 1):, currency]\
                  .plot(ax=axr[i-1], marker='|', markersize=.5)
                line = channel.line
                line_x = np.arange(df.last_valid_index() - (k), 
                                   df.last_valid_index() + 1)
                line_y_up = line + (np.ones(line.shape[0]) * 2 \
                                    * channel.channel_deviation)
                line_y_down = line + (np.ones(line.shape[0]) * -2 \
                                      * channel.channel_deviation)
                #axr[i-1].plot(line_x, line, color='grey', linewidth=.5)
                axr[i-1].plot(line_x, line_y_up, 
                                     color='black', linewidth=.75)
                axr[i-1].plot(line_x, line_y_down, 
                                     color='black', linewidth=.75)
        
    
        # Row labels - add to left side of first column
        for i in first[:-1]:
            axr[i].set_ylabel(df.columns[i + 1].upper(), rotation=90, 
                              size='large', color='white')
        # Row Labels - add to right side of second column
        for i in second:
            axr[i].set_ylabel(df.columns[i + 1].upper(), rotation=270, 
                              size='large', color='white')
            axr[i].yaxis.tick_right()
            axr[i].yaxis.set_label_position('right')
            
        
        # Add line at next granularity for M15 and M5
        if granularity == 'M15':
            line_break = 3
            vert = int(df.last_valid_index() - interval / line_break)
            axr[-1].plot([vert, vert], [0, 1], color='black', linewidth=1)
            line_break = 15
            vert = int(df.last_valid_index() - interval / line_break)
            axr[-1].plot([vert, vert], [0, 1], color='black', linewidth=1)
        elif granularity == 'M5':
            line_break = 5
            vert = int(df.last_valid_index() - interval / line_break)
            axr[-1].plot([vert, vert], [0, 1], color='black', linewidth=1)
            
            
        # Set plot x Time markers
        x_ticks_count = plt.xticks()[0].shape[0]
        x_tick_locations = np.linspace(plot_index[0], plot_index[-1], x_ticks_count).astype(int)
        x_ticks_markers = df.loc[x_tick_locations, 'timestamp']
        x_ticks_markers = x_ticks_markers.apply(lambda x: x - pd.Timedelta(hours=hours_subtraction_from_utc)).dt.time
        x_ticks_markers = x_ticks_markers.astype(str).apply(lambda x: x[:5]).values.astype(str)
        # Set x Grid
        for a in axr:
            a.set_xticks(x_tick_locations)
            a.set_xticklabels(x_ticks_markers, color='white')
            a.grid(which='major', axis='x', linestyle= '-',linewidth=.2, 
                                       color='#343837', b=True) 
            
        # Set Y Grid
        for i in range(1, df.columns.shape[0]):    
            max_ticks = df.iloc[plot_index, i].max()
            min_ticks = df.iloc[plot_index, i].min()
            # Different ticks for jpy and hkd
            if df.columns[i] == 'jpy':
                ticks = np.arange(min_ticks, max_ticks, .000001).round(6)
            elif df.columns[i] == 'hkd':
                ticks = np.arange(min_ticks, max_ticks, .00001).round(5)
            else:
                ticks = np.arange(min_ticks, max_ticks, .0001).round(4)
                # Create ticks 
            axr[i - 1].grid(linestyle= '-', axis='y', linewidth=.2, 
                                       color='#343837', b=True)    
            axr[i - 1].set_yticks(ticks)
         
        # Remove Y Ticks
        for a in ax.ravel():
            a.set_yticklabels([])
            
        # Print Recent Location
        print(df.last_valid_index())
    except Exception as e:
        print(e)
        
        


def eur_usd(data, granularity, file=file, interval=interval, 
        normaltest_step=normaltest_step,
        normaltest_alpha=normaltest_alpha,
        hours_subtraction_from_utc=hours_subtraction_from_utc):
    pass

    # Import Data
    df = pd.read_pickle(str(file))
    df = df.loc[:, ['timestamp', 'cad', 'eur', 'usd']].copy()
    plot_index = np.arange(df.last_valid_index() - interval, 
                           df.last_valid_index() + 1)
    
    axr = ax.ravel()
    

    
    # Plot Data
    for i in range(df.columns.shape[0] - 1):
        axr[i].cla()
        df.iloc[plot_index, 1+i].plot(ax=axr[i], linewidth=2)
    
    
    # get channels for currency
    for i in range(1, df.columns.shape[0]):
        coll = []
        currency = df.columns[i]
        for s in range(20, interval + normaltest_step, normaltest_step):
            channel = Channel(df.loc[df.last_valid_index() - s: \
                                     df.last_valid_index(), currency].values)
            k2, p = normaltest(channel.flattened / channel.channel_deviation)
            #print("p = {:g}".format(p))
            if p < normaltest_alpha:
                pass # "The null hypothesis can be rejected")
            else:
                coll.append(s) # "The null hypothesis cannot be rejected")
        coll = np.array(coll)
   
       
        # Get Positions for channel breaks
        keep = []
        for k in range(coll.shape[0] - 1):
            if coll[k + 1] > coll[k] + normaltest_step:
                keep.append(coll[k])
        if coll.shape[0] > 0: keep.append(coll[-1])
        keep = np.array(keep)
    
        
        # Plot chnnels by normal break
        for k in keep[::-1]:
            channel = Channel(df.loc[df.last_valid_index() - k:, 
                                     currency].values)
            df.loc[df.last_valid_index() - (k + 1):, currency]\
              .plot(ax=axr[i-1], marker='|', markersize=.5)
            line = channel.line
            line_x = np.arange(df.last_valid_index() - (k), 
                               df.last_valid_index() + 1)
            line_y_up = line + (np.ones(line.shape[0]) * 2 \
                                * channel.channel_deviation)
            line_y_down = line + (np.ones(line.shape[0]) * -2 \
                                  * channel.channel_deviation)
            #axr[i-1].plot(line_x, line, color='grey', linewidth=.5)
            axr[i-1].plot(line_x, line_y_up, 
                                 color='black', linewidth=.75)
            axr[i-1].plot(line_x, line_y_down, 
                                 color='black', linewidth=.75)
            
            
    # Row labels - add to left side of first column
    for i in range(axr.shape[0]):
        axr[i].set_ylabel(df.columns[i + 1].upper(), rotation=90, 
                          size='large', color='white')



        
    # Set plot x Time markers
    x_ticks_count = plt.xticks()[0].shape[0]
    x_tick_locations = np.linspace(plot_index[0], plot_index[-1], x_ticks_count).astype(int)
    x_ticks_markers = df.loc[x_tick_locations, 'timestamp']
    x_ticks_markers = x_ticks_markers.apply(lambda x: x - pd.Timedelta(hours=hours_subtraction_from_utc)).dt.time
    x_ticks_markers = x_ticks_markers.astype(str).apply(lambda x: x[:5]).values.astype(str)
    # Set x Grid
    for a in axr:
        a.set_xticks(x_tick_locations)
        a.set_xticklabels(x_ticks_markers, color='white')
        a.grid(which='major', axis='x', linestyle= '-',linewidth=.2, 
                                   color='#343837', b=True) 
        
    # Set Y Grid
    for i in range(1, df.columns.shape[0]):    
        max_ticks = df.iloc[plot_index, i].max()
        min_ticks = df.iloc[plot_index, i].min()
        # Different ticks for jpy and hkd
        if df.columns[i] == 'jpy':
            ticks = np.arange(min_ticks, max_ticks, .000001).round(6)
        elif df.columns[i] == 'hkd':
            ticks = np.arange(min_ticks, max_ticks, .00001).round(5)
        else:
            ticks = np.arange(min_ticks, max_ticks, .0001).round(4)
            # Create ticks 
        axr[i - 1].grid(linestyle= '-', axis='y', linewidth=.2, 
                                   color='#343837', b=True)    
        axr[i - 1].set_yticks(ticks)
     

    # Print Recent Location
    print(df.last_valid_index())

        

        




def run_3_column(data, granularities, path=path, interval=interval, 
        normaltest_step=normaltest_step,
        normaltest_alpha=normaltest_alpha):

    for g in range(len(granularities)):
        # Import Data
        df = pd.read_pickle(path + str(granularities[g]) + '.pkl')
        df = df[['timestamp', 'aud', 'cad', 'nzd', 'chf', 'eur', 'usd']]
#        df.drop('jpy', axis=1, inplace=True)
#        df.drop('hkd', axis=1, inplace=True)
        plot_index = np.arange(df.last_valid_index() - interval, 
                               df.last_valid_index() + 1)
        
        axr = ax[:, g]
        
        # Plot Data
        for i in range(df.columns.shape[0] - 1):
            axr[i].cla()
            df.iloc[plot_index, 1+i].plot(ax=axr[i], linewidth=2)
            axr[i].set_xticklabels([])
    
        '''
        # get channels for currency
        for i in range(1, df.columns.shape[0]):
            coll = []
            currency = df.columns[i]
            for s in range(20, interval + normaltest_step, normaltest_step):
                channel = Channel(df.loc[df.last_valid_index() - s: \
                                         df.last_valid_index(), currency].values)
                k2, p = normaltest(channel.flattened / channel.channel_deviation)
                #print("p = {:g}".format(p))
                if p < normaltest_alpha:
                    pass # "The null hypothesis can be rejected")
                else:
                    coll.append(s) # "The null hypothesis cannot be rejected")
            coll = np.array(coll)
        
            # Get Positions for channel breaks
            keep = []
            for k in range(coll.shape[0] - 1):
                if coll[k + 1] > coll[k] + normaltest_step:
                    keep.append(coll[k])
            if coll.shape[0] > 0: keep.append(coll[-1])
            keep = np.array(keep)
            
            # Plot chnnels by normal break
            for k in keep[::-1]:
                channel = Channel(df.loc[df.last_valid_index() - k:, 
                                         currency].values)
                df.loc[df.last_valid_index() - (k + 1):, currency]\
                  .plot(ax=axr[i-1], marker='|', markersize=.5)
                line = channel.line
                line_x = np.arange(df.last_valid_index() - (k), 
                                   df.last_valid_index() + 1)
                line_y_up = line + (np.ones(line.shape[0]) * 2 \
                                    * channel.channel_deviation)
                line_y_down = line + (np.ones(line.shape[0]) * -2 \
                                      * channel.channel_deviation)
                #axr[i-1].plot(line_x, line, color='grey', linewidth=.5)
                axr[i-1].plot(line_x, line_y_up, 
                                     color='black', linewidth=.75)
                axr[i-1].plot(line_x, line_y_down, 
                                     color='black', linewidth=.75)
                
               
        '''
        # Set plot x Time markers
        x_ticks_count = plt.xticks()[0].shape[0]
        x_tick_locations = np.linspace(plot_index[0], plot_index[-1], x_ticks_count).astype(int)
        x_ticks_markers = df.loc[x_tick_locations, 'timestamp']
        x_ticks_markers = x_ticks_markers.apply(lambda x: x - pd.Timedelta(hours=hours_subtraction_from_utc)).dt.time
        x_ticks_markers = x_ticks_markers.astype(str).apply(lambda x: x[:5]).values.astype(str)
        
        # Set x Grid
        for a in axr:
            a.set_xticks(x_tick_locations)
            a.set_xticklabels(x_ticks_markers, color='white', rotation=270)
            a.grid(which='both', linestyle= '-',linewidth=.2, 
                                            color='#343837', b=True) 
        
        '''
        # Get largest Y Scale
        largest_range = 0
        for i in range(df.iloc[:,1:].shape[1]):
            extent = axr[i].get_ylim()[1] - axr[i].get_ylim()[0]
            if extent > largest_range:
                largest_range = extent
        
        # Set y scale
        for i in range(df.iloc[:,1:].shape[1]):
            extent = axr[i].get_ylim()[1] - axr[i].get_ylim()[0]
            diff = largest_range - extent
            diff /= 2
            axr[i].set_ylim(axr[i].get_ylim()[0] - diff, axr[i].get_ylim()[1] + diff)
        '''
        # fSet ticks to right
        for a in axr:
            a.yaxis.tick_right()
        
        '''
        # Add line at next granularity for M15 and M5
        max_ticks = df.iloc[plot_index, i].max()
        min_ticks = df.iloc[plot_index, i].min()
        if granularities[g] == 'M15':
            print('m15')
            line_break = 3
            vert = int(df.last_valid_index() - interval / line_break)
            axr[-1].plot([vert, vert], [min_ticks, max_ticks], color='black', linewidth=1)
            line_break = 15
            vert = int(df.last_valid_index() - interval / line_break)
            axr[-1].plot([vert, vert], [min_ticks, max_ticks], color='black', linewidth=1)
        if granularities[g] == 'M5':
            print('m5')
            line_break = 5
            vert = int(df.last_valid_index() - interval / line_break)
            axr[-1].plot([vert, vert], [min_ticks, max_ticks], color='black', linewidth=1)
        '''
        
        

        
        bins=30
        for i in range(df.columns[1:].shape[0]):
            values = df.iloc[plot_index, i + 1].values
            s = supports(values, bins)
            for support in s:
                axr[i].plot(plot_index, np.ones(plot_index.shape[0]) * support, color='grey')
        
        
        
                
                
        
        
        
        print('{}:\t{}'.format(granularities[g],
                               df.loc[df.last_valid_index(), 'timestamp']))

    # Add Row Labels to first column
    for i in range(1, df.columns.shape[0]):
        ax[i-1, 0].set_ylabel(df.columns[i].upper(), rotation=90,  
                          size='large', color='white')

    # Add granularity column title
    for g in range(len(granularities)):
#        sessions = interval * int(granularities[g][1:])
#        session /= (60)
        ax[0, g].set_title(granularities[g], rotation=0,  
                          size='large', color='white')
    




def run(data, granularity, file=file, interval=interval, 
        normaltest_step=normaltest_step,
        normaltest_alpha=normaltest_alpha):

    # Get data to start plot

    # Import Data
    try:
        df = pd.read_pickle(str(file))
    except Exception as e:
        print(e)
        print(interval)
        print(str(file))
        exit
    plot_index = np.arange(df.last_valid_index() - interval, 
                           df.last_valid_index() + 1)
    
    # Plot Data
    for i in range(len(ax)):
        ax[i].cla()
        df.iloc[plot_index, 1+i].plot(ax=ax[i], linewidth=2)
        
    # Add labels
    for i in range(ax.shape[0]):
        ax[i].set_ylabel(df.columns[i + 1], rotation=90, 
                                            size='large', color='white')
        ax[i].yaxis.tick_right()
        ax[i].tick_params(axis='y', colors='white')

    # get channels for currency
    for i in range(1, df.columns.shape[0]):
        coll = []
        currency = df.columns[i]
        for s in range(20, interval + normaltest_step, normaltest_step):
            channel = Channel(df.loc[df.last_valid_index() - s: \
                                     df.last_valid_index(), currency].values)
            k2, p = normaltest(channel.flattened / channel.channel_deviation)
            #print("p = {:g}".format(p))
            if p < normaltest_alpha:
                pass # "The null hypothesis can be rejected")
            else:
                coll.append(s) # "The null hypothesis cannot be rejected")
        coll = np.array(coll)
    
        # Get Positions for channel breaks
        keep = []
        for k in range(coll.shape[0] - 1):
            if coll[k + 1] > coll[k] + normaltest_step:
                keep.append(coll[k])
        if coll.shape[0] > 0: keep.append(coll[-1])
        keep = np.array(keep)
        
        # Plot chnnels by normal break
        for k in keep[::-1]:
            channel = Channel(df.loc[df.last_valid_index() - k:, 
                                     currency].values)
            df.loc[df.last_valid_index() - (k + 1):, currency]\
              .plot(ax=ax.ravel()[i-1], marker='|', markersize=.5)
            line = channel.line
            line_x = np.arange(df.last_valid_index() - (k), 
                               df.last_valid_index() + 1)
            line_y_up = line + (np.ones(line.shape[0]) * 2 \
                                * channel.channel_deviation)
            line_y_down = line + (np.ones(line.shape[0]) * -2 \
                                  * channel.channel_deviation)
            ax.ravel()[i-1].plot(line_x, line, color='grey', linewidth=.5)
            ax.ravel()[i-1].plot(line_x, line_y_up, 
                                 color='black', linewidth=.5)
            ax.ravel()[i-1].plot(line_x, line_y_down, 
                                 color='black', linewidth=.5)
    
    # Add line at next granularity for M15 and M5
    if granularity == 'M15':
        line_break = 3
    else:
        line_break = 5
    if granularity != 'M1':
        max_ticks = df.iloc[plot_index, i].max()
        min_ticks = df.iloc[plot_index, i].min()  
        vert = int(df.last_valid_index() - interval / line_break)
        ys = ax[-1].get_ylim()
        ax.ravel()[-1].plot([vert, vert], [ys[0], ys[0] \
                          + ((ys[1] - ys[0]) / 3)], color='black', linewidth=1)
    
    
    # Add grid to subplots
    for i in range(1, df.columns.shape[0]):    
        max_ticks = df.iloc[plot_index, i].max()
        min_ticks = df.iloc[plot_index, i].min()
        # Different ticks for jpy and hkd
        if df.columns[i] == 'jpy':
            ticks = np.arange(min_ticks, max_ticks, .000001).round(6)
        elif df.columns[i] == 'hkd':
            ticks = np.arange(min_ticks, max_ticks, .00001).round(5)
        else:
            ticks = np.arange(min_ticks, max_ticks, .0001).round(4)
        # Create ticks    
        ax.ravel()[i - 1].grid(which='both', linewidth=.5, 
                               color='grey', b=True)    
        ax.ravel()[i - 1].set_yticks(ticks, '-')
        
    # Print and Return
    print(df.last_valid_index())






# EVERYTHING !!!!    
if 0:
    # Draw figure
    df = pd.read_pickle(file)
#    df.drop('jpy', axis=1, inplace=True)
#    df.drop('hkd', axis=1, inplace=True)
    ratios = pd.read_pickle(ratios_file)
    ratios_drop = ratios.filter(regex='HKD').columns.tolist() \
            + ratios.filter(regex='HKD').columns.tolist()
    ratios = ratios.drop(ratios_drop, axis=1)
    fig, ax = plt.subplots(df.shape[1] - 1, df.shape[1] - 1, 
                           sharex=True, 
                           num='Ratios: ' +  '      ' + str(granularity) \
                               + '      ' + str(interval),
                           facecolor='#343837')
    plt.subplots_adjust(hspace=.1, wspace=.05, bottom=.07, top=.97, 
                        right=.97, left=.03)
    # Call plot
    ani = animation.FuncAnimation(fig, run_all, fargs=(granularity,),
                                  blit=False, interval=30000)    
    plt.show()




# run all for two currencies 
if 0:
    # Draw figure
    df = pd.read_pickle(file)
    df.drop('jpy', axis=1, inplace=True)
    df.drop('hkd', axis=1, inplace=True)
    ratios = pd.read_pickle(ratios_file)
    ratios_drop = ratios.filter(regex='JPY').columns.tolist() \
            + ratios.filter(regex='HKD').columns.tolist()
    ratios = ratios.drop(ratios_drop, axis=1)
    fig, ax = plt.subplots(df.shape[1] - 1, 3, 
                           sharex=True, 
                           num='Ratios: ' +  '      ' + str(granularity) \
                               + '      ' + str(interval),
                           facecolor='#343837')
    plt.subplots_adjust(hspace=.1, wspace=.05, bottom=.07, top=.97, 
                        right=.97, left=.03)
    # Call plot
    ani = animation.FuncAnimation(fig, run_all_for_two, fargs=(granularity,),
                                  blit=False, interval=30000)    
    plt.show()

# Standardized    
if 0:
    
    fig, ax = plt.subplots(1, 1, sharex=False, 
                           num='Standardized     ' + str(interval) + '     ' + str(granularity),
                           facecolor='#343837')
    plt.subplots_adjust(hspace=.05, wspace=.15, bottom=.075, top=.95, 
                        right=.95, left=.025)
    
    plot_index = np.arange(df.last_valid_index() - interval, 
                           df.last_valid_index() + 1)
    
    ss = StandardScaler().fit_transform(df.iloc[:, 1:])
    ss = pd.DataFrame(ss, columns = df.columns[1:], index=df.index)
    ss.insert(0, 'timestamp', df.timestamp)
    # ss.drop('jpy', axis=1, inplace=True)
    ss.drop('hkd', axis=1, inplace=True)    
    
    ss.iloc[plot_index, 1:].plot(ax = ax)
    
    # Set plot x Time markers
    x_ticks_count = plt.xticks()[0].shape[0]
    x_tick_locations = np.linspace(plot_index[0], plot_index[-1], x_ticks_count).astype(int)
    x_ticks_markers = df.loc[x_tick_locations, 'timestamp']
    x_ticks_markers = x_ticks_markers.apply(lambda x: x - pd.Timedelta(hours=hours_subtraction_from_utc)).dt.time
    x_ticks_markers = x_ticks_markers.astype(str).apply(lambda x: x[:5]).values.astype(str)
    
    ax.set_xticks(x_tick_locations)
    ax.set_xticklabels(x_ticks_markers, color='white', rotation=270)
    ax.grid(which='major', axis='x', linestyle= '-',linewidth=.2,
                               color='#343837', b=True) 
    
    
    if False:
        # Call plot
        ani = animation.FuncAnimation(fig, standardized, blit=False, 
                                      interval=refresh_interval)    
        plt.show()



# Volume    
if 0:
    
    fig, ax = plt.subplots(1, 1, sharex=False, 
                           num='Volume',
                           facecolor='#343837')
    plt.subplots_adjust(hspace=.05, wspace=.15, bottom=.075, top=.95, 
                        right=.95, left=.025)
    # Call plot
    ani = animation.FuncAnimation(fig, volume, blit=False, 
                                  interval = 3000)    
    plt.show()



# Diff    
if 0:
    
    fig, ax = plt.subplots(1, 1, sharex=False, 
                           num='Diff',
                           facecolor='#343837')
    plt.subplots_adjust(hspace=.05, wspace=.15, bottom=.075, top=.95, 
                        right=.95, left=.025)
    # Call plot
    ani = animation.FuncAnimation(fig, volume, blit=False, 
                                  interval = 3000)    
    plt.show()



# Multiple Granularities
if 3:
    # Draw figure
    df = pd.read_pickle(path + str(granularities[0]) + '.pkl')
    df = df[['timestamp', 'aud', 'cad', 'nzd', 'chf', 'eur', 'usd']]
#    df.drop('jpy', axis=1, inplace=True)
#    df.drop('hkd', axis=1, inplace=True)
    fig, ax = plt.subplots(df.shape[1] - 1, len(granularities), sharex=False, 
                           num='Currency Universe     ' + str(interval),
                           facecolor='#343837')
    plt.subplots_adjust(hspace=.05, wspace=.15, bottom=.075, top=.95, 
                        right=.95, left=.025)
    # Call plot
    ani = animation.FuncAnimation(fig, run_3_column, fargs=(granularities,), blit=False, 
                                  interval=refresh_interval)    
    plt.show()


# 2 Colums
if 0:
    # Draw figure
    df = pd.read_pickle(file)
    df.drop('jpy', axis=1, inplace=True)
    df.drop('hkd', axis=1, inplace=True)
    df.drop('gbp', axis=1, inplace=True)
    plt.style.use(['seaborn-paper'])
    fig, ax = plt.subplots(int(np.ceil((df.shape[1] - 1) / 2)), 2, sharex=True, 
                           num=str(granularity) + '      ' + str(interval),
                           facecolor='#343837')
    plt.subplots_adjust(hspace=.2, wspace=.2, bottom=.2, top=.8, 
                        right=.8, left=.2)
    plt.subplots_adjust(hspace=.1, wspace=.03, bottom=.03, top=.97, 
                        right=.97, left=.03)    
    
    # a = Window(fig)
    # Call plot
    ani = animation.FuncAnimation(fig, run_2_column, fargs=(granularity,), blit=False, 
                                  interval=refresh_interval)    
    plt.show()


# eur_usd Colums
if 0:
    # Draw figure
    df = pd.read_pickle(file)
    plt.style.use(['seaborn-paper'])
    fig, ax = plt.subplots(3, 1, sharex=True, 
                           num=str(granularity) + '      ' + str(interval),
                           facecolor='#343837')
    plt.subplots_adjust(hspace=.2, wspace=.2, bottom=.2, top=.8, 
                        right=.8, left=.2)
    plt.subplots_adjust(hspace=.1, wspace=.03, bottom=.03, top=.97, 
                        right=.97, left=.03)    
    
    # a = Window(fig)
    # Call plot
    ani = animation.FuncAnimation(fig, eur_usd, fargs=(granularity,), blit=False, 
                                  interval=refresh_interval)    
    plt.show()


# 1 Column
if 0:
    # Draw figure
    df = pd.read_pickle(file)
    fig, ax = plt.subplots(df.shape[1] - 1, 1, sharex=True, num=granularity,
                           facecolor='#343837')
    plt.subplots_adjust(hspace=.1, wspace=.05, bottom=.05, top=.98, 
                        right=.9, left=.05)
    
    # Call plot
    ani = animation.FuncAnimation(fig, run, fargs=(granularity,), blit=False, 
                                  interval=refresh_interval)    
    plt.show()
    






# Just some looks
if False:
    cur1 = 'aud'
    cur2 = 'cad'
    r = 'AUD_CAD'
    fig, ax = plt.subplots(3, 1, sharex=True)
    df[cur1].plot(ax=ax[0], label = cur1)
    df[cur2].plot(ax=ax[1], label = cur2)
    rat[r].plot(ax=ax[2], label=r)
    for a in ax:
        a.grid(which='both')
        a.legend()
    plt.tight_layout()
     
    
    
    
# 2column
if False:
#    # Draw figure
    df = pd.read_pickle('/northbend/tmp/currencies_M5.pkl')
    df = df[['timestamp', 'aud', 'chf', 'cad', 'eur', 'nzd', 'usd']]
    rat = pd.read_pickle('/northbend/tmp/ratios_M5.pkl')
    plt.style.use(['seaborn-paper'])
    fig, ax = plt.subplots(int(np.ceil((df.shape[1] - 1) / 2)), 2, sharex=True, 
                           num=str(granularity) + '      ' + str(interval),
                           facecolor='#343837', figsize=(20, 10))
    plt.subplots_adjust(hspace=.1, wspace=.03, bottom=.03, top=.97, 
                        right=.97, left=.03)    
    for i in range(ax.ravel().shape[0]):
        col = df.columns[i+1]
        df.loc[:, col].plot(ax=ax.ravel()[i], label=col)
        ax.ravel()[i].grid(which='both')
        ax.ravel()[i].legend()
    plt.tight_layout()

    
#
#    for a in ax.ravel():
    
#        a.autoscale()
        
        
    
if False:
        # Declare and register callbacks
    fig, ax = plt.subplots(1,1)
    df.iloc[:, 1].plot(ax=ax)
    lowest = df.iloc[:, 1].index.values.min()
    highest = df.iloc[:, 1].index.values.max()
    
    def on_xlims_change(axes):
        print( "updated xlims: ", ax.get_xlim())
    
    def on_ylims_change(axes):
        print( "updated ylims: ", ax.get_ylim())
    
    def set_new_y_lim(axes):
        global lowest
        global highest
        lower = int(ax.get_xlim()[0])
        upper = int(ax.get_xlim()[1])
        if lower < lowest:
            lower = lowest
        if upper > highest:
            upper = highest
        print('lower: {}'.format(lower))
        print('higher: {}'.format(upper))
        y_low = df.iloc[lower:upper, 1].min()
        y_high = df.iloc[lower:upper, 1].max()
        ax.set_ylim(y_low, y_high)
        print('y_low: {}'.format(y_low))
        print('y_high: {}'.format(y_high))
        plt.show()        
#        lower_bound = df.iloc[lower, 1]
#        upper_bound = df.iloc[upper, 1]
#        ax.set_ylim(lower_bound, upper_bound)
    
    ax.callbacks.connect('xlim_changed', on_xlims_change)
    ax.callbacks.connect('ylim_changed', on_ylims_change)
    ax.callbacks.connect('xlim_changed', set_new_y_lim)
            
    plt.show()
    
    
    
    
    
