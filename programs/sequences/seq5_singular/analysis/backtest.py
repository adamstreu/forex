
import time
import random
from itertools import product
from dateutil.relativedelta import relativedelta
from datetime import date
import datetime
import os
import multiprocessing
import numpy as np
import pandas as pd
from scipy.stats import binom_test
import oandapyV20
from oandapyV20.contrib.factories import InstrumentsCandlesFactory

np.warnings.filterwarnings('ignore')

# Instrument 
granularity = 'M15'
daily_alignment = 17

# Time 
start_year = 2018   # Don't go lower than 5
end_year = 2019
periods = 'months' 

# Sequence
start = 50
stop = 301
step = 50
seq_len = 2
bar_limit = 500

# Filters
placements_filter = 50
ratio_filter = 1
return_filter = 0
win_perc_filter = 0

# Precalcs
steps = np.arange(start, stop, step)
binomial_filter = 1/((len(steps)**(2*seq_len)) * (2**seq_len) * 2) 







'''

Next:
    Analysis:
        
        
        Calculate bars, etc on outcomes for
        Try next: Just dod for polacement ratios of 1
        Remove all but one placemnt (50, 0)
        




TODAY:
    Start long and awesome backtest on AWS.  Maybe try a few granularities, etc.
    Write program
        Write the check for placements.
        correct logging to portfolio to analysis data.
    
Next:
    Analysis:
        filter at end on places.
        check timelines
            store outcomes as well (temporarily)
            to tableau :    ind, askclose, df.ind, position, outcome for all in df
        can we divide placements into groups, then see how well they do per group?
            for instance, if it wins two almost right next to each other, 
            perhaps that 'win' series whould just count for one.  then we might 
            get an even better idea of what does well and what doesn't.
                So, how close is a group?
    Main Program:
        correct logging to portfolio to analysis data.
        Write the check for placements.

At some point i really need to check that my predictions on the first 
algorithm matched up with the actual placements and outcomes.
If the outcome sdon't match I have a logic and computation problem

    
TO DO:
    Long Term analysis:
      -  Conduct backtests 
      -  Write analytics
      -  long term analysis
      -  portfolio building
      -  write main program 
      -  deploy (just need analysis on one pair to do (even done by hand))
      -  check backtest results by hand
        
    Short Term Analysis:
        Conduct backtests 
        Write analytics
        long term analysis
        portfolio building
        write main program 
        deploy


AFTERWARDS EXPERIMENTS:
    Can we try out all of this with close values (not high or low)
    make sure that placemnets from here jive with placements on regular ask
        or try using real placements outcome swith last one in sequence.
    Is there a 'best' time of day for setting the daily_alignment?

    
ALGORITHM NOTES:
    
    Using gan = 'H12', is there any diff in better algorithm perf when alighning
        it to different times of the day?
    
    total results before each placement = [(steps ** 2) * 2] ** (seq_len) * 2
    
    
    
ALGORITHM STEPS:
    get candles.
    compute bars, udmin, etc.  
    gather keys, groups, etc.
    main:
        work througheach sequence of (seq, out, direction)
        from that, take a top result from each of e_ret, win_perc, throughput
        remove any that don;t match binomial test
        possible remove near or total suplicat times.
    
    
Additional indicator:
    the first one: 
        volume data might be available from oanda.
        Also, whatever order book is.
            
    
OVERALL:
    This is another algorithm that tries to select winning placements 
    from a sequence of binary outcomes.  Assuming some blue coins are 
    higher probability winners than the red coins, how
    does a color blind person selet the correct sequence?
    
    My answer so far:  try all sequences and see which ones work the best.  
    This does not gaurantee anything though.  Try enough times and I will pick 
    every sequence there is.

    So we need to try to see that what we pick is better than just randomly 
    till we get a good one.  Can we analyze our resulting picks as a whole?
    If too many sequences land on exactly the same candles out steps / granularity is out of rythm.
    well, we can always keep one from bestbinom, best e_ret, best win_perc.
    
    Currently, we are taking two results from 
    unfilitered, if we take 1 from each seq, outcome, direction pair, 
        out final df should number of rows:
        sequences * outcomes  = 
        len(keys) * 2 ** seq_len = 
                [[(len(groups) * 2] ** seq_len]] * 2
                    +   whatever amount doesn't have any results
                    
        Again.  Every combination of seq set (key) and outcomes and direction.
            Direction is not counted in there.
            So, total results will be:
                (above  - what is filtered out ) *  # we keep from group
    
    
'''


def get_orig_bars(return_dict, candles, risk_reward, target, bar_limit, direction):
    #return_dict[target] = []
    # Set target values
    df = candles.copy()
    if direction == 'long':
        if risk_reward == 'risk':
            df['target'] = (df.askclose - df.spread) - (.0001 * target)
        elif risk_reward == 'reward':
            df['target'] = (df.askclose - df.spread) + (.0001 *  target) 
    elif direction == 'short':
        if risk_reward == 'risk':
            df['target'] = (df.bidclose + df.spread) + (.0001 * target)
        elif risk_reward == 'reward':
            df['target'] = (df.bidclose + df.spread) - (.0001 *  target)
    # Calculate where target is hit (sticks)
    dfv = df.values
    ind = []
    for i in range(dfv.shape[0]):
        j = min(dfv.shape[0] -1 , bar_limit + i)
        if direction == 'long':
            if risk_reward == 'risk':
                tmp_ind = np.where(dfv[i+1:j, 3] <= dfv[i, 10])
            elif risk_reward == 'reward':
                tmp_ind = np.where(dfv[i+1:j, 2] >= dfv[i,10])
        elif direction == 'short':
            if risk_reward == 'risk':
                tmp_ind = np.where(dfv[i+1:j, 6] >= dfv[i, 10]) 
            elif risk_reward == 'reward':
                tmp_ind = np.where(dfv[i+1:j, 7] <= dfv[i,10])
        # Append bar results to ind
        if tmp_ind[0].shape[0] != 0:
            ind.append(tmp_ind[0][0] + 1)
        else:
            ind.append(bar_limit) 
    return_dict[target] = list(ind)





def get_candles(_from, _to, instrument, granularity = granularity,
                da = daily_alignment):
    # Prepare Request
    code = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = code
    client = oandapyV20.API(access_token=client)
    params = {'from': _from,
              'to': _to,
              'granularity': granularity,
              'price': 'M',
              'count': 5000,
              'alignmentTimezone': 'UTC', #'America/Los_Angeles',
              'dailyAlignment': da}
    # Request Data
    coll = []
    for r in InstrumentsCandlesFactory(instrument = instrument, 
                                       params = params):
        client.request(r)
        coll.append(r.response)
    # collect Returned Data into list.  Cast to floats.
    low = []
    high = []
    close = []
    timestamp = []
    for i in range(len(coll)):
        for j in range(len(coll[i]['candles'])):
            high.append(float(coll[i]['candles'][j]['mid']['h']))
            low.append(float(coll[i]['candles'][j]['mid']['l']))
            close.append(float(coll[i]['candles'][j]['mid']['c']))               
            timestamp.append(coll[i]['candles'][j]['time'])
    # Assemble DataFrame.  Cast Values.
    df = pd.DataFrame(pd.to_datetime(timestamp))
    df.columns = ['timestamp']
    df['high'] = pd.to_numeric(high)
    df['low'] = pd.to_numeric(low)
    df['close'] = pd.to_numeric(close)
    return df


def get_bars(candles, steps, bar_limit=bar_limit):
    # collect bars to risk and reward arrays.
    def calculate_bars(risk_dict, reward_dict, params, candles=candles, 
                       steps=steps, bar_limit=bar_limit):
        for each in params:
            target = steps[each[0]]
            direction = each[1]
            ind = []
            df = candles.copy()
            print(df.shape)
            # Set target values on up
            if direction == 'up':
                df['target'] = (df.close) + (.0001 * target)
                dfv = df.values
                for i in range(dfv.shape[0]):
                    j = min(dfv.shape[0], i + bar_limit )
                    tmp_ind = np.where(dfv[i+1:j, 1] >= dfv[i, 4])
                    if tmp_ind[0].shape[0] != 0:
                        ind.append(tmp_ind[0][0] + 1)
                    else:
                        ind.append(bar_limit)
            # Set target values on down
            else:
                df['target'] = (df.close) - (.0001 * target)
                dfv = df.values
                for i in range(dfv.shape[0]):
                    j = min(dfv.shape[0], i + bar_limit)
                    tmp_ind = np.where(dfv[i+1:j, 2] <= dfv[i, 4])
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
    # return
    return up, down, udo, ud_min_bars_next, _min


def crawl_bars(candles, keys, steps, groups, udo, udmin, udnext, 
               seq_len=seq_len, pf=placements_filter, rf=ratio_filter,
               ret_filter = return_filter, bf = binomial_filter,
               wf = win_perc_filter):

    def df_from_results(arr, bf=bf, steps=steps, seq_len=seq_len):
        columns = ['sequence', 
                   'outcomes', 
                   'position', 
                   'direction', 
                   'ratio', 
                   'up', 
                   'down', 
                   'placements', 
                   'win_perc', 
                   'p_win_perc',
                   'e_ret',
                   'throughput', 
                   'places',
                   'choice']
        df = pd.DataFrame(arr, columns=columns)
        # Calculate and Filter on the binom test
        binom_test_v = np.vectorize(binom_test)
        df.insert(9, 'binom', binom_test_v(df.win_perc * df.placements, 
                                           df.placements, df.p_win_perc))
        df = df[df.binom < bf]
        # Extra Calculations and filters go here
        df['t_ret'] = df.e_ret * df.placements
        df['avg_bars'] = df.e_ret / df.throughput
        print(len(df))
        return df
    
    
    def create_ratios(steps=steps, groups=groups):
        g = []
        for each in groups:
            g.append([steps[each[1][0]], steps[each[1][1]]])
        g = np.array(g)
        up_ratio = g[:, 0] / g[:, 1]
        down_ratio = g[:, 1] / g[:, 0]
        return up_ratio, down_ratio
            
    
    def calculate_sequence(return_dict, keys, i, groups=groups, steps = steps,
                           udmin=udmin, udo=udo, udnext=udnext, 
                           seq_len = seq_len, rf=rf, ret_f = ret_filter,
                           pf=pf, wf=wf): 
        keep = []
        outcome_group = list(product([0, 1], repeat=seq_len))  
        count = 0
        count_empty = 0
        space = np.linspace(1, len(keys), 10).astype(int)
        up_ratio, down_ratio = create_ratios()
        for key in keys:
#            count += 1
#            if i == 0 and count in space:
#                msg = 'Percent Complete: {}% '
#                print(msg.format(np.argmax(space==count) * 10))
            for out in outcome_group:
                # Do first step in sequence.
                u = groups[key[0]][1][0]
                d = groups[key[0]][1][1]
                o = out[0]
                take = (udnext[u][d] * udo[u][d][o])  
                take = take[take != 0]
                take = np.unique(take)
                # Calculate remaining steps is sequence.
                for s in range(1, seq_len):
                    u = groups[key[s]][1][0]
                    d = groups[key[s]][1][1]
                    o = out[s]
                    where = (udnext[u][d] * udo[u][d][o])    
                    take = np.take(where, take)
                    take = take[take != 0]
                    take = np.unique(take)     
                    placements = len(take)
                seq_coll = []
                if placements > pf:
                    # Calculate all outcomes at final positions.
                    for direction in [1, 0]: 
                        if direction == 1:
                            ratios = up_ratio
                        else:
                            ratios = down_ratio
                        for g in range(len(groups)):
                            ratio = ratios[g]
                            if ratio == 1: #>= rf:
                                u = groups[g][1][0]
                                d = groups[g][1][1]
                                outs = np.take(udo[u][d][direction], take)
                                win_perc = outs.mean()
                                e_ret =  (win_perc * (ratio + 1 )) - 1
                                if e_ret > ret_f and win_perc >= wf:
                                    throughput = e_ret / udmin[u][d][take].mean() 
                                    seq_coll.append([key,
                                                 out, 
                                                 groups[g][0],
                                                 direction,
                                                 ratio,
                                                 steps[u], 
                                                 steps[d], 
                                                 placements, 
                                                 win_perc,
                                                 udo[u][d][direction].mean(),
                                                 e_ret,
                                                 throughput,
                                             take])
                if len(seq_coll) > 0:
                    df_coll = np.array(seq_coll)
                    e_ret_column = 10
                    win_perc_column = 8
                    throughput_column = 11
                    ret = list(df_coll[np.argmax(df_coll[:, e_ret_column])]) 
                    win = list(df_coll[np.argmax(df_coll[:, win_perc_column])]) 
                    put = list(df_coll[np.argmax(df_coll[:, throughput_column])]) 
                    keep.append(ret + ['ret'])
                    keep.append(win + ['win'])
                    keep.append(put + ['put'])
                else:
                    count_empty +=1 
        if len(keep) > 0:
            print(len(keep))
            dict_coll = df_from_results(keep)
            return_dict[i] = dict_coll 
            print(count_empty)
        else:
            print('no df returned')
        
    # Prepare various jobs for workers
    random.shuffle(keys)                
    work_params = {}
    worker_count = multiprocessing.cpu_count() - 1
    zip_range = int(len(keys) / worker_count) + 1
    last = 0
    i = 0
    while last < len(keys):
        work_params[i] = keys[last: last + zip_range]
        last = last + zip_range
        i += 1
    # Start Jobs anc collect results
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict.update(jobs)
    # Run Jobs
    for i in work_params.keys():
        p = multiprocessing.Process(target=calculate_sequence, 
                                    args=(return_dict, work_params[i], i))
        jobs.append(p)
        p.start()
    for job in jobs: 
        job.join()
    outcome = pd.DataFrame(columns = return_dict[0].columns)
    for k in return_dict.keys():
        outcome = outcome.append(return_dict[k])
    outcome.reset_index(drop=True, inplace=True)
    return outcome


def create_keys_and_groups(steps):
    ud_combos = list(product(range(len(steps)), range(len(steps))))
    groups = list(zip(np.arange(len(ud_combos)), ud_combos))
    keys_combos = list(product(np.arange(len(groups)), repeat = seq_len))
    return keys_combos, groups


def get_next_key_groups(df, steps, current_keys, pf, rf, rrf, 
                        bf, sort_by, keep):
    df = df.copy()
    df = df[df.placements > pf]
    df = df[df.e_ret > rf]                                        
    df = df[df.perc > rrf]
    # Add in Binomial test
    binom_test_v = np.vectorize(binom_test)
    df['binom'] = binom_test_v(df.win_perc * df.placements, 
                               df.placements, df.p_win_perc)
    df = df[df.binom < bf]
    # sort values and group and drop.
    if df.shape[0] > 0:
        return df.sort_values(sort_by).head(keep)[['group', 'outcomes', 
                                                   'direction']].values
    else:
        print('nothing left of df.  Returning old keys')
        return current_keys        
        
        
def extract_key_results(df, keys_groups):
    ret = pd.DataFrame(columns=df.columns)
    for k in keys_groups:
        try: 
            ret = ret.append(df[(df.group == k[0]) & (df.outcomes == k[1]) & \
                                (df.direction == k[2])])
        except Exception as e:
            print(e)        
    return ret


def get_periods(year_start, year_end, period_length, periods_max=10000):
    def time_delta(t, period_length=period_length):
        if period_length == 'months':
            t += relativedelta(months=1)
        if period_length == 'weeks':
            t += relativedelta(weeks=1)
        if period_length == 'days':
            t += relativedelta(days=1)
        return t
    coll = []
    start = datetime.datetime(year_start, 1, 1,0,0,0)
    end = datetime.datetime(year_end, 1, 1,0,0,0)  
    while start <= end:
        coll.append(str(pd.to_datetime(start)).replace(' ', 'T') + 'Z' )
        start = time_delta(start)#relativedelta(period_length=1)
    return coll[:periods_max]


def candles_to_periods(df, period):
    candles_p = df.copy()
    candles_p['timestamp'] = pd.DatetimeIndex(candles_p.timestamp).normalize()
    candles_p = candles_p.set_index('timestamp', drop=True)
    position = str(period).upper()[0]
    candles_p = candles_p.groupby(pd.TimeGrouper(position)).mean()
    return candles_p


def filter_further(df):
    '''
    Filter by places.  
    sort by binom or win perc or return.
    take top one, remove all where places are duplicated (or don;t add any,. etc)
    
    # Prepare df1
    #df1 = dfall[dfall.win_perc == 1].copy()
    #binom_test_v = np.vectorize(binom_test)
    #df1['binom'] = binom_test_v(df1.win_perc * df1.placements, 
    #                            df1.placements, df1.p_win_perc)
    #df1 = df1[df1.binom < .001]
    index_coll = []
    df1 = dfall.copy()
    df = df1.sort_values('e_ret', ascending=False).copy()
    while df.shape[0] > 0:
        index_coll.append(df.head(1).index[0])
        pl = []
        l = list(df.places.values[0])
        for i in df.index.values:
            if list(df.loc[i, 'places']) == l:
                pl.append(i)
        df = df.drop(pl, axis=0)
    '''
    return df


def prepare_for_portfolio(df, currency, gran, from_, analysis_type, 
                          steps, groups, da = daily_alignment):
    dfc = df.copy()
    dfc.insert(0, 'granularity', gran)    
    dfc.insert(0, 'pair', currency)
    dfc.insert(6, 'seq_len', seq_len)
    dfc['analysis_date'] = str(date.today())
    dfc['from'] = from_
    dfc['start'] = start
    dfc['stop'] = stop
    dfc['step'] = step
    dfc['analysis_type'] = analysis_type
    dfc['daily_alignment'] = da
    dfc.drop(['up', 'down'], axis=1, inplace=True) # 
    dfc.drop(['places'], axis=1, inplace=True)
    seq = []
    pos = []
    sequences = df.sequence.values
    positions = df.position.values
    [pos.append(tuple((steps[groups[x][1][0]], steps[groups[x][1][1]]))) for x in positions]
    [seq.append(tuple([tuple((steps[groups[x][1][0]], steps[groups[x][1][1]])) for x in y])) for y in sequences]
    dfc.position = pos
    dfc.sequence = seq
    file = 'results/long_analysis_on_{}_for_{}_{}_{}_{}_{}_{}_{}.csv'
    file = file.format(str(date.today()),
                       currency, 
                       granularity,
                       seq_len,
                       start,
                       stop,
                       step,
                       time_periods[0])
    dfc.to_csv(file)
    file = 'results/portfolio_build_long_{}.port'
    file = file.format(str(date.today()))
    with open(file, 'a') as f:
        for line in dfc.values:
            f.write('{},\n'.format(list(line)))
    return dfc


def create_results_dir():
    # Create directory to store results 
    if not os.path.exists('results'):
        os.makedirs('results')
        

if __name__ == '__maind__': 
    
    
    currencies = ['GBP_NZD', 'AUD_CAD', 'EUR_AUD', 'AUD_USD', 'EUR_CHF',
                  'EUR_GBP', 'GBP_CHF', 'GBP_USD', 'NZD_USD', 'USD_CAD', 
                  'USD_CHF', 'EUR_NZD', 'EUR_SGD', 'EUR_CAD', 'USD_SGD',
                  'GBP_AUD', 'EUR_USD']
    currencies = ['GBP_NZD']
    for currency in currencies:
        try:   
            time1 = time.time()
            msg = 'long_analysis_on_{}_for_{}_{}_{}_{}_{}.csv'
            print(msg.format(currency, granularity,seq_len, start, stop, step))
            ###################################################################
            # Long Analysis
            create_results_dir()
           
            candles = get_candles(time_periods[0], time_periods[-1], currency) 
            up, down, udo, udnext, udmin = get_bars(candles, steps) 
            print(candles.shape)
            print(up.shape)
            keys, groups = create_keys_and_groups(steps) 
            df = crawl_bars(candles, keys, steps, groups, udo, udmin, udnext)
            df_down = filter_further(df)
            df_port = prepare_for_portfolio(df, currency, granularity, 
                                       time_periods[0], 'long', steps, groups)
            ###################################################################
            print('Run time: {:.2f}'.format((time.time() - time1) / 60))

        except Exception as e:
            print(currency, e)
        
        





    '''
    # Process Timelines sequentially (adjust keys per segment)
    ###########################################################################
    # Time variables
    start_year = 2016
    end_year = 2017
    periods = 'weeks' # months, doys, years
    time_periods = get_periods(start_year, end_year, periods, 100)
    # Filter variables
    p_filter = 5
    ret_filter = 0
    r_r_filter = .75
    b_filter = .01
    keep = 50
    order = 'binom'
    # Process first time period typically.  Assess first key round
    _from = time_periods[0]
    _to = time_periods[1]
    key_coll = []
    candles = get_candles(_from, _to) 
    up, down, udo, udnext, udmin = get_bars(candles, steps) 
    keys, groups = create_keys_and_groups(steps) 
    df = crawl_bars(candles, keys, steps, groups, udo, udmin, udnext)
    key_groups = get_next_key_groups(df, steps, 0, p_filter, ret_filter, 
                                     r_r_filter, b_filter, order, keep)

    # Run simulation for all time periods
    tdf = pd.DataFrame()  
    cdf = pd.DataFrame()    
    for i in range(1, len(time_periods) - 1 ):
        # Get and compute new data for new time periods
        print(i) 
        _from = time_periods[i]
        _to = time_periods[i+1]
        key_coll += list(key_groups)
        candles = get_candles(_from, _to) 
        up, down, udo, udnext, udmin = get_bars(candles, steps) 
        # return typical analysis.  
        df = crawl_bars(candles, keys, steps, groups, udo, udmin, udnext)
        df['period'] = i
        cdf = cdf.append(df)
        # From analysis, Keep results from key group
        results = extract_key_results(df, key_groups)
        results['period'] = i
        tdf = tdf.append(results)
        # Obtain next key group from monthly analysis
        key_groups = get_next_key_groups(df, steps, key_groups, p_filter, 
                                         ret_filter, r_r_filter, 
                                         b_filter, order, keep)
    # Finished results over all time group.
    cdf.columns = df.columns
    tdf.columns = results.columns
    ###########################################################################
#     file = 'results_{}_{}_{}_{}.csv'.format(currency, 
#                                              granularity,
#                                              seq_len,
#                                              '_'.join(str(e) for e in steps),
#                                              _from)
    # results.to_csv(file)
    
    
    
    #with open('output', 'w') as file:
#    file.write('{}, {}, {}, {}, {} \n'.format(currency, granularity, 
# _from, _to, steps))

    

    '''
    
    
    
    '''
    TIMELINE NOTES:
        One hour does not provide enough data for 
        getting keys for the following week.
        
        
        
    '''
    
    
    # candles_p = candles_to_periods(candles, periods)  

    
    

    
    