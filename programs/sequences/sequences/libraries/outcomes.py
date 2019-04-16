import numpy as np
import pandas as pd
import random
from itertools import product
import multiprocessing

bar_limit = 4000
target_column = 'highlow' # 'closing' or 'highlow'


def up_down_simple(y, target_up, target_down):
    up   = np.argmax(y >= target_up)
    down = np.argmax(y <= target_down)
    if (up!=0 and down==0) or (up<down and up!=0):
        return (1, up)
    elif (up==0 and down!=0) or (down<up and down!=0):
        return (0, down)
    elif (up == 0 and down == 0):
        return (-1, up)


def get_ud_bars(candles, steps, bar_limit=bar_limit, tc=target_column):
    # collect bars to risk and reward arrays.
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


def get_outcomes(candles, position, bar_limit=bar_limit):
    print('Calculating Outcomes on {}'.format(position))
    # Calculate rwo, rwmin for the final placement
    def get_bars(candles, direction, risk_reward, bar_limit=bar_limit):
        dfv = candles.values
        ind = []
        for i in range(dfv.shape[0]):
            j = min(dfv.shape[0] - 1 , bar_limit + i)
            if direction == 'long':
                if risk_reward == 'risk':
                    tmp_ind = np.where(dfv[i+1:j, 2] <= dfv[i, 12])
                elif risk_reward == 'reward':
                    tmp_ind = np.where(dfv[i+1:j, 1] >= dfv[i,12])
            elif direction == 'short':
                if risk_reward == 'risk':
                    tmp_ind = np.where(dfv[i+1:j, 4] >= dfv[i, 12]) 
                elif risk_reward == 'reward':
                    tmp_ind = np.where(dfv[i+1:j, 5] <= dfv[i,12])
            # Append bar results to ind
            if tmp_ind[0].shape[0] != 0:
                ind.append(tmp_ind[0][0] + 1)
            else:
                ind.append(bar_limit) 
        ind = np.array(ind)
        return ind
    # For each direction and r | r get bars
    df = candles.copy()
    df['target'] = (df.askclose) + (.0001 * position[0])
    short_risk = get_bars(df, 'short', 'risk')
    df['target'] = (df.askclose) - (.0001 * position[1])
    short_reward = get_bars(df, 'short', 'reward')
    df['target'] = (df.bidclose) - (.0001 * position[1])
    long_risk = get_bars(df, 'long', 'risk')
    df['target'] = (df.bidclose) + (.0001 * position[0])
    long_reward = get_bars(df, 'long', 'reward')
    # Assemeble bars into rwo, rwmin
    long_rw = long_reward < long_risk
    short_rw = short_reward < short_risk
    rwo = np.stack((short_rw,long_rw),axis=0)
    short_min = np.minimum(short_risk, short_reward)
    long_min = np.minimum(long_risk, long_reward)
    rwmin = np.stack((short_min, long_min),axis=0)
    return rwo, rwmin
    