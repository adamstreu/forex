import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import multiprocessing









'''
===============================================================================
Multiprocessing Outcomes - up down.  No Spread
===============================================================================
'''

def up_down(close, target, t, up_dict, down_dict, max_search_interval):
    
    # Instantiate
    print('Getting outcomes on target: {}'.format(t))
    up = []
    down = []
    except_value = close.shape[0]
    # Find bars for each target    
    for i in range(close.shape[0]):
        stop = min(i + max_search_interval, except_value)
        try:
            up.append(np.where(close[i: stop] > close[i] + target)[0][0])
        except:
            up.append(max_search_interval)
        try:
            down.append(np.where(close[i: stop] < close[i] - target)[0][0])
        except:
            down.append(max_search_interval)
    # Return to Dict
    up_dict[t] = up
    down_dict[t] = down

    
def get_outcomes_multi(close, targets, max_search_interval):
   
    # Call processes, join, wait to complete, etc.
    cpus = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    up_dict = manager.dict()
    down_dict = manager.dict()
    jobs = []
    
    # Run Jobs for first Rotations
    for t in range(targets.shape[0]):
        p = multiprocessing.Process(target=up_down, 
                                    args=(close,
                                          targets[t], 
                                          t, 
                                          up_dict,
                                          down_dict,
                                          max_search_interval))
        jobs.append(p)
        p.start()
        if ((t + 1) % cpus == 0 ) or (t == targets.shape[0] - 1):
            for job in jobs:
                job.join()  
                
    # Add Results in order and return
    up = pd.DataFrame()
    down = pd.DataFrame()
    for t in range(targets.shape[0]):
        up[t]   = up_dict[t]
        down[t] = down_dict[t]    
    # columns of course
    up.columns = targets.round(8)
    down.columns = targets.round(8)
    # Return
    return {
            'up': up,
            'down': down
            }
        
    
    
def plot_outcomes(location, c, up, down, targets, minimums):
    '''
    For use with function above.  For use after function above has been called.
    To be called from main program to verify correctness, spacing, etc.,
    of calculated outcomes.
    '''
    up_range = up.iloc[location]
    down_range = down.iloc[location]
    interval = minimums.loc[location].max()
    up_list = plt.cm.Blues(np.linspace(.25, 1, targets.shape[0]))
    down_list = plt.cm.Oranges(np.linspace(.25, 1, targets.shape[0]))
    line_list = plt.cm.Greys(np.linspace(.25, .75, targets.shape[0]))
    # Plot c from location to max of up / down interval
    c.loc[location: location + interval].plot(color='grey')
    # Plot circle of c at c at up and down
    for t in range(targets.shape[0]):
        if up_range[targets[t]] < interval:
            plt.plot(location + up_range.values[t],
                     c.iloc[location + up_range.values[t]], 'o', color=up_list[t])
        if down_range[targets[t]] < interval:
            plt.plot(location + down_range.values[t],
                 c.iloc[location + down_range.values[t]], 'o', color=down_list[t])                 
    # Plot c + target at location
    for t in range(targets.shape[0]):
        plt.plot(np.arange(location, location + interval),
                 np.ones(interval) * (c.loc[location] + targets[t]),
                 color = line_list[t])
        plt.plot(np.arange(location, location + interval),
                 np.ones(interval) * (c.loc[location] - targets[t]),
                 color = line_list[t])
    # Plot line at c location
    plt.plot(np.arange(location, location + interval),
                 np.ones(interval) * (c.loc[location]),
                 color = 'black')
    
    
    
    
    
    
'''
===============================================================================
For Linear regression: how far will it go in a direction
===============================================================================
'''

def linear_outcomes(closing_values, max_steps = 30):

    try:
        vals = closing_values.values
    except:
        vals = closing_values
    max_steps = 30
    coll = []
    
    # Follow the next value up or down and see how far it gets in max_steps steps
    for i in range(vals.shape[0] - 1):    
        # If next values is positive
        if vals[i + 1] - vals[i] > 0:
            end_search = np.argmax(vals[i: i + max_steps] < vals[i])
            if end_search == 0:
                end_search = max_steps + i
            value = vals[i : i + end_search].max()
            coll.append(value - vals[i])
        # If next values is positive
        elif vals[i + 1] - vals[i] < 0:
            end_search = np.argmax(vals[i: i + max_steps] > vals[i])
            if end_search == 0:
                end_search = max_steps + i
            value = vals[i : i + end_search].min() 
            coll.append(value - vals[i])
        # If next values is nuetral        
        elif vals[i + 1] - vals[i] == 0:
            coll.append(0)
            end_search = i
            value = i
        #print(i, end_search, value, coll[-1])
    coll.append(0)
    
    return np.array(coll)
    
        

'''
===============================================================================
Multiprocessing Up Down with high and low and spread
===============================================================================
'''


    
def get_outcomes_multi_h_l_spread(close, targets, search_interval):
   
    # Call processes, join, wait to complete, etc.
    cpus = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    up_dict = manager.dict()
    down_dict = manager.dict()
    jobs = []
    
    # Run Jobs for first Rotations
    for t in range(targets.shape[0]):
        p = multiprocessing.Process(target=up_down, 
                                    args=(close,
                                          targets[t], 
                                          t, 
                                          search_interval, 
                                          up_dict,
                                          down_dict))
        jobs.append(p)
        p.start()
        if ((t + 1) % cpus == 0 ) or (t == targets.shape[0] - 1):
            for job in jobs:
                job.join()  
                
    # Add Results in order and return
    up = pd.DataFrame()
    down = pd.DataFrame()
    for t in range(targets.shape[0]):
        up[t]   = up_dict[t]
        down[t] = down_dict[t]    
    # columns of course
    up.columns = targets.round(8)
    down.columns = targets.round(8)
    # Return
    return {
            'up': up,
            'down': down
            }
        
    
    
    
    

    
    
    
'''
===============================================================================
tmp.  currency universe.
===============================================================================
'''
    
    
    
    
def high_low_open_outcomes(_open, high, low, search_interval, target, plot=False):
    
    high_coll = []
    low_coll = []
    
    stop = _open.shape[0]
    for i in range(stop):
        
        end = min(stop, i + search_interval)
        
        high_target = _open[i] + target[i] 
        low_target  = _open[i] - target[i]
        find_high   = np.where(high[i: end] > high_target)    
        find_low    = np.where(low[i: end] < low_target)
        
        try:
            high_coll.append(find_high[0][0])
        except:
            high_coll.append(search_interval)
        try:
            low_coll.append(find_low[0][0])
        except:
            low_coll.append(search_interval)    
    
    return {'high': np.array(high_coll),
            'low' : np.array(low_coll)}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
===============================================================================
Outcomes with Direction and Spread.  Complete.  No Multiprocess  Also Misc old.
===============================================================================
'''

def get_outcomes(direction, candles, location, search_interval, targets, plot=False):
    '''
    Purpose:
        Hopefully, matches outcomes exactly as found in real placement.
        I have the option of setting outcomes so that profit and loss 
            are equal at same target OR profit and loss MOVE the smae amount 
            (where then the abs(loss) will be greater than the gain)).
        I hvae choose to go with movement is the same.
        Therefore, the difference in the prices:
            the spread will be added to the loss and subtracted from the win.
            In other words - spread is alwasy subtracted.
        Long positions are bought at askclose.
        Targets are askclose +or- spread + width
        Final note: spread is added into the target.
    '''
    up = []
    down = []
    # Set direction parameters
    if direction == 'long':
        close = 'bidclose'
        high  = 'bidhigh'
        low   = 'bidlow'
    elif direction == 'short':
        close = 'askclose'
        high  = 'askhigh'
        low   = 'asklow'
    else:
        return None    
    close = candles.loc[location, close]
    spread = candles.loc[location, 'spread']
    # Conduct Search
    search_high = candles.loc[location: location + search_interval, high].values
    search_low  = candles.loc[location: location + search_interval, low].values
    
    for target in targets:
        try:
            up.append(np.where(close + target + spread < search_high)[0][0])
        except:
            up.append(search_interval)
        try:
            down.append(np.where(close - (target + spread) > search_low)[0][0])
        except:
            down.append(search_interval)
    # set value directions for return
    if direction == 'long':
        to_return = {'target': up,   'loss': down}
    elif direction == 'short':
        to_return = {'target': down, 'loss': up}
    else:
        to_return = None
    # Plot (verification)
    if plot:
        # plot the high and low search interval
        plt.figure(figsize=(14,8))
        x = np.arange(search_high.shape[0])
        plt.plot(search_high)
        plt.plot(search_low)
        plt.plot(x, np.ones(x.shape[0]) * close, '_', color='lightgrey') # plo dashed line along closing values
        for target in targets:
            plt.plot(x, np.ones(x.shape[0]) * (close + (target + spread)), 
                     color='lightgrey')
            plt.plot(x, np.ones(x.shape[0]) * (close - (target + spread)), 
                     color='lightgrey')  
        for i in range(len(to_return['target'])):
            if direction == 'long':
                plt.plot(x[to_return['target'][i]], 
                         search_high[to_return['target'][i]], 
                         'o', color='green')        
                plt.plot(x[to_return['loss'][i]], 
                         search_low[to_return['loss'][i]], 'o', color='red')
            else:
                plt.plot(x[to_return['target'][i]], 
                         search_low[to_return['target'][i]], 
                         'o', color='green')
                plt.plot(x[to_return['loss'][i]], 
                         search_high[to_return['loss'][i]], 'o', color='red')        
    return to_return





    
    def up_down_bars(candles, search_interval):        
        def outcomes(candles, location, search_interval, targets):
            '''
            Purpose:
                Hopefully, matches outcomes exactly as found in real placement.
                I have the option of setting outcomes so that profit and loss 
                    are equal at same target OR profit and loss MOVE the smae amount 
                    (where then the abs(loss) will be greater than the gain)).
                I hvae choose to go with movement is the same.
                Therefore, the difference in the prices:
                    the spread will be added to the loss and subtracted from the win.
                    In other words - spread is alwasy subtracted.
                Long positions are bought at askclose.
                Targets are askclose +or- spread + width
                Final note: spread is added into the target.
            '''
            up = []
            down = []
            close = candles.loc[location, 'midclose']
            spread = candles.loc[location, 'spread']
            # Conduct Search
            search_high = candles.loc[location: location + search_interval, 'midclose'].values
            search_low  = candles.loc[location: location + search_interval, 'midclose'].values
            for target in targets:
                try:
                    up.append(np.where(close + target + spread < search_high)[0][0])
                except:
                    up.append(search_interval)
                try:
                    down.append(np.where(close - (target + spread) > search_low)[0][0])
                except:
                    down.append(search_interval)
            # set value directions for return
            to_return = {'up': up,   'down': down}

            return to_return
        
        '''
        def calculate_bars(risk_dict, reward_dict, params, candles=candles, 
                           steps=steps, bar_limit=bar_limit, tc=tc):
            if tc == 'closing':
                high = 9
                low = 9
            else:
                high = 7
                low = 8
            for each in params:
                target = steps[each[0]]
                direction = each[1]
                ind = []
                df = candles.copy()
                # Set target values on up
                if direction == 'up':
                    df['target'] = (df.midclose) + (.0001 * target)
                    dfv = df.values
                    for i in range(dfv.shape[0]):
                        j = min(dfv.shape[0], i + bar_limit )
                        tmp_ind = np.where(dfv[i+1:j, high] >= dfv[i, 12]) 
                        if tmp_ind[0].shape[0] != 0:                    
                            ind.append(tmp_ind[0][0] + 1)
                        else:
                            ind.append(bar_limit)
                # Set target values on down
                else:
                    df['target'] = (df.midclose) - (.0001 * target)
                    dfv = df.values
                    for i in range(dfv.shape[0]):
                        j = min(dfv.shape[0], i + bar_limit)
                        tmp_ind = np.where(dfv[i+1:j, low] <= dfv[i, 12]) 
                        if tmp_ind[0].shape[0] != 0:
                            ind.append(tmp_ind[0][0] + 1)
                        else:
                            ind.append(bar_limit)
                # put results into appropriate manager_dictionary
                if direction == 'up':
                    up[each[0]] = list(ind)
                else:
                    down[each[0]] = list(ind)   
        '''
        # Create a list of jobs for each worker - pass through dictionary
        zipped = list(product(range(len(steps)), ['up', 'down']))
        random.shuffle(zipped)
        work_params = {}
        worker_count = multiprocessing.cpu_count() - 1
        zip_range = int(len(zipped) / worker_count) + 1
        last = 0
        i = 0
        while last < len(zipped):
            work_params[i] = zipped[last: last + zip_range]
            last = last + zip_range
            i += 1
        # Collect results into np arrays
        jobs = []
        manager = multiprocessing.Manager()
        up = manager.dict()
        down = manager.dict()
        up.update(jobs)
        down.update(jobs)
        for i in range(len(list(work_params.keys()))):
            p = multiprocessing.Process(target=calculate_bars, 
                                        args=(up, down, work_params[i]))
            jobs.append(p)
            p.start()
        for job in jobs: 
            job.join()
        collect_up = []
        collect_down = []
        for row in range(len(steps)):
            collect_up.append(up[row])
            collect_down.append(down[row])        
        up = np.array(collect_up)
        down = np.array(collect_down)
        # Get up and down outcomes and accessed by [up][down]  
        _up = up.reshape(up.shape[0],1,up.shape[1])
        _down = np.tile(down, (down.shape[0], 1)).reshape(down.shape[0], 
                                                          down.shape[0], 
                                                          down.shape[1])
        udou = (_up < _down)
        udod = (_down < _up)
        udo = np.stack((udod,udou),axis=2)
        # combine with minimums and candle location to get placement
        _min = np.minimum(_up, _down)
        places = np.tile(np.arange(up.shape[1]),(up.shape[0] * down.shape[0], 1))
        places = places.reshape(up.shape[0], down.shape[0], down[0].shape[0])
        ud_min_bars_next = _min + places
        return up, down, udo, ud_min_bars_next, _min
    


if __name__ == '__main__':
    pass


