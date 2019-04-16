import time
import random
from itertools import product
from datetime import date
import multiprocessing
import numpy as np
import pandas as pd
from scipy.stats import binom_test
import oandapyV20
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import seaborn as sns


# Ignore numpy warnings
np.warnings.filterwarnings('ignore')

# Run Options
run_program = 2  # remote = 0, local =  1, instance = 2
filter_places = False
filter_binomial = False

# Instrument
currency = 'EUR_AUD'
granularity = 'M30'
daily_alignment = 17

# Time
_from = '2010-04-01T00:00:00Z'
_to = '2018-05-01T00:00:00Z'

# Steps
start = 50
stop = 451
step = 50

# Sequence
seq_len = 3
target_column = 'highlow' # 'closing' or 'highlow'
within_bars = 50

# Position
position = (75, 75)

# Filters
bar_limit = 4000  
placements_filter = 0
win_perc_filter = .9
binom_filter= .001  # 1 / ((len(steps) ** (2 * seq_len)) * (2 ** seq_len) * 2)
return_filter = 0


# Currencies for run program 0, 1
currencies = ['AUD_CAD', 'EUR_AUD', 'AUD_USD', 'EUR_CHF',
              'EUR_GBP', 'GBP_CHF', 'GBP_USD', 'NZD_USD', 'USD_CAD', 
              'USD_CHF', 'EUR_NZD', 'EUR_SGD', 'EUR_CAD', 'USD_SGD',
              'GBP_AUD', 'EUR_USD'] #'GBP_NZD', 
#currencies = ['GBP_NZD']


def get_candles(instrument, granularity, _from, _to, da=daily_alignment):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'from': _from,
              'to': _to,
              'granularity': granularity,
              'price': 'BAM',
              'count': 5000,
              'alignmentTimezone': 'UTC',
              'dailyAlignment': da}
    # Request Data
    coll = []
    for r in InstrumentsCandlesFactory(instrument = instrument, 
                                       params = params):
        client.request(r)
        coll.append(r.response)
    # collect Returned Data into list.  Cast to floats.
    bidlow = []
    bidhigh = []
    bidclose = []
    asklow = []
    askhigh = []
    askclose = []
    midlow = []
    midhigh = []
    midclose = []
    timestamp = []
    volume = []
    for i in range(len(coll)):
        for j in range(len(coll[i]['candles'])):
            bidhigh.append(float(coll[i]['candles'][j]['bid']['h']))
            bidlow.append(float(coll[i]['candles'][j]['bid']['l']))
            bidclose.append(float(coll[i]['candles'][j]['bid']['c']))
            askhigh.append(float(coll[i]['candles'][j]['ask']['h']))
            asklow.append(float(coll[i]['candles'][j]['ask']['l']))
            askclose.append(float(coll[i]['candles'][j]['ask']['c']))               
            midhigh.append(float(coll[i]['candles'][j]['mid']['h']))
            midlow.append(float(coll[i]['candles'][j]['mid']['l']))
            midclose.append(float(coll[i]['candles'][j]['mid']['c']))               
            timestamp.append(coll[i]['candles'][j]['time'])
            volume.append(float(coll[i]['candles'][j]['volume']))
    # Assemble DataFrame.  Cast Values.
    df = pd.DataFrame(pd.to_datetime(timestamp))
    df.columns = ['timestamp']
    df['bidhigh'] = pd.to_numeric(bidhigh)
    df['bidlow'] = pd.to_numeric(bidlow)
    df['bidclose'] = pd.to_numeric(bidclose)
    df['askhigh'] = pd.to_numeric(askhigh)
    df['asklow'] = pd.to_numeric(asklow)
    df['askclose'] = pd.to_numeric(askclose)
    df['midhigh'] = pd.to_numeric(midhigh)
    df['midlow'] = pd.to_numeric(midlow)
    df['midclose'] = pd.to_numeric(midclose)
    df['spread'] = df.askclose - df.bidclose
    df['volume'] = pd.to_numeric(volume)
    return df


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


def get_position_bars(candles, position, bar_limit=bar_limit):
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
    

def get_positions(candles, steps, sequence, outcome, udo, udnext,
                  wb = within_bars):
    seq_len = len(sequence)
    # Do first step in sequence.
    u = np.argmax(steps == sequence[0][0])
    d = np.argmax(steps == sequence[0][1])
    o = outcome[0]
    take = (udnext[u][d] * udo[u][d][o] )  
    take = take[take != 0]
    take = np.unique(take)
    # Calculate remaining steps is sequence.
    if seq_len > 1:
        for s in range(1, seq_len):
            u = np.argmax(steps == sequence[s][0])
            d = np.argmax(steps == sequence[s][1])
            o = outcome[s]
            where = (udnext[u][d] * udo[u][d][o])   
            take = np.take(where, take)
            take = take[take != 0]
            take = np.unique(take)      
    return take


def crawl_bars(candles, sequences, steps,  position,udo, udmin, udnext, 
               rwo, rwmin, seq_len=seq_len, wf = win_perc_filter,
               filter_places=filter_places,pf=placements_filter,
               bar_limit=bar_limit, filter_binomial=filter_binomial,
               rf = return_filter):
    # Worker main crawl sequence
    def calculate_sequence(return_dict, sequences, i, rwo=rwo, rwmin=rwmin, 
                           position=position, candles = candles,  pf=pf,
                           steps = steps, udmin=udmin, udo=udo, udnext=udnext, 
                           seq_len = seq_len, wf=wf, bar_limit=bar_limit,
                           filter_places=filter_places, rf=rf): 
        # For each outcomes, sequence, combination, call placements function
        outcomes = list(product([0, 1], repeat=seq_len)) 
        count = 0
        space = np.linspace(1, len(sequences), 20).astype(int)
        seq_coll = []
        for sequence in sequences:
            count += 1
            if i == 0 and count in space:
                msg = 'Percent Complete: {} / 20 '
                print(msg.format(np.argmax(space==count)))
            for outcome in outcomes:
                take = get_positions(candles, steps, sequence, outcome,
                                     udo, udnext)
                if take.shape[0] > pf:
                    for direction in [1, 0]:
                        outs = rwo[direction][take]
                        win_perc = outs.mean()
                        if win_perc > wf:
                            placements = take.shape[0]
                            avg_bars = rwmin[direction][take][rwmin[direction][take] \
                                             < bar_limit].mean()
                            seq_coll.append([sequence,
                                             outcome, 
                                             position,
                                             direction,
                                             int(placements), 
                                             win_perc,
                                             avg_bars,
                                             take,
                                             outs])   
                        
        if len(seq_coll) > 0:   
            columns = ['sequence', 
                       'outcomes', 
                       'position', 
                       'direction', 
                       'placements', 
                       'win_perc', 
                       'avg_bars',
                       'places',
                       'wins']
            df_seq = pd.DataFrame(seq_coll, columns=columns)
            if filter_binomial:
                df_seq = binomial_filter(df_seq, rwo)
            if filter_places:
                df_seq = places_filter_by_exact(df_seq, 'win_perc')
            return_dict[i] = df_seq
                            
    # Prepare various jobs for workers
    work_params = {}
    worker_count = multiprocessing.cpu_count() - 1
    zip_range = int(len(sequences) / worker_count) + 1
    last = 0
    i = 0
    while last < len(sequences):
        work_params[i] = sequences[last: last + zip_range]
        last = last + zip_range
        i += 1
    # Start Jobs anc collect results
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict.update(jobs)
    # Run Jobs and wait until they are finished to proceed
    for i in work_params.keys():
        p = multiprocessing.Process(target=calculate_sequence, 
                                    args=(return_dict, work_params[i], i))
        jobs.append(p)
        p.start()
    for job in jobs: 
        job.join()
    # If Any results are to be returned, collect columns and append to df
    if len(return_dict.keys()) > 0:
        columns = return_dict[return_dict.keys()[0]].columns
        outcome = pd.DataFrame(columns = columns)
        for k in return_dict.keys():
            outcome = outcome.append(return_dict[k])
        outcome.reset_index(drop=True, inplace=True)
        return outcome
    

def create_sequences(start, stop, step, seq_len):
    steps = np.arange(start, stop, step)
    # steps = np.r_[np.arange(25, 251, 25), np.arange(300,451, 50)]
    step_combos = list(product(list(steps), repeat=2))
    sequences = list(product(list(step_combos), repeat=seq_len))
    return steps, sequences


def filter_further(df):
    return df


def create_portfolio_output(_df, currency, gran, _from, analysis_type, 
                            steps, groups, da = daily_alignment):
    dfc = _df.copy()
    dfc.insert(0, 'granularity', gran)    
    dfc.insert(0, 'pair', currency)
    dfc.insert(6, 'seq_len', seq_len)
    dfc['analysis_date'] = str(date.today())
    dfc['from'] = _from
    dfc['start'] = start
    dfc['stop'] = stop
    dfc['step'] = step
    dfc['analysis_type'] = analysis_type
    dfc['daily_alignment'] = da
    dfc.drop(['places'], axis=1, inplace=True) 
    file = 'results/long_analysis_on_{}_for_{}_{}_{}_{}_{}_{}_{}.csv'
    file = file.format(str(date.today()),
                       currency, 
                       granularity,
                       seq_len,
                       start,
                       stop,
                       step,
                       _from)
    file = 'results/portfolio_build_long_{}.port'
    file = file.format(str(date.today()))
    with open(file, 'a') as f:
        for line in dfc.values:
            f.write('{},\n'.format(list(line)))
    return dfc

    
def binomial_filter(df, rwo, bf=binom_filter):
    df.loc[df.direction == 0, 'p_win_perc'] = rwo[0].mean()
    df.loc[df.direction == 1, 'p_win_perc'] = rwo[1].mean()
    binom_test_v = np.vectorize(binom_test)
    df['binom'] =  binom_test_v(df.win_perc * df.placements, 
                                           df.placements, df.p_win_perc)
    df = df[df.binom < bf]
    return df


def places_filter_by_day(df, sort_column, candles):
    '''
    Should really be using winning locations, not just placements.
    '''
    # Places Filter
    if sort_column == 'binom':
        ascending=True
    else:
        ascending=False
    dfd = df.sort_values(sort_column, ascending = ascending)
    weeks = candles.copy()
    weeks['timestamp'] = pd.DatetimeIndex(weeks.timestamp).normalize()
    weeks['ind'] = weeks.index.values
    weeks = weeks.set_index('timestamp', drop=True)
    weeks = weeks.groupby('timestamp').max().ind.values 
    w = [] 
    w_coll = []
    for place in dfd.places.values:
        for each in place:
            w.append(np.argmax(weeks > each))
        w_coll.append(list(set(w)))
        w = []
    dfd['week_wins'] = w_coll 
    portfolio = []
    port_tracker = []
    for each in dfd.week_wins:
        if ((~np.isin(list(set(each)), portfolio)).sum()) >= 1:
            portfolio = list(set(portfolio + each))
            port_tracker.append(1)
        else:
            port_tracker.append(0)
    dfd['portfolio'] = port_tracker
    dfd = dfd[dfd.portfolio == 1]
    dfd = dfd.drop(['portfolio','week_wins'], axis=1)
    return dfd

def places_filter_by_exact(df, sort_column):
    if sort_column == 'binom':
        ascending=True
    else:
        ascending=False
    dfd = df.sort_values(sort_column, ascending = ascending)
    portfolio = []
    port_tracker = []
    for each in dfd.places:
        if ((~np.isin(list(set(each)), portfolio)).sum()) >= 1:
            portfolio = list(set(portfolio + list(each)))
            port_tracker.append(1)
        else:
            port_tracker.append(0)
    dfd['portfolio'] = port_tracker
    dfd = dfd[dfd.portfolio == 1]
    dfd = dfd.drop(['portfolio'], axis=1)
    return dfd


def further_process_results(df):
    df['e_ret'] = (df.win_perc * 2) - 1
    df['t_ret'] = df.e_ret * df.placements
    df['throughput'] = df.e_ret / df.avg_bars
    places = df.places
    df.drop('places', axis=1, inplace=True)
    return df, places


def main(currency=currency, granularity=granularity, 
         bar_limit=bar_limit, seq_len=seq_len, position=position, 
         placeement_filter=placements_filter, win_perc_filter=win_perc_filter,
         binom_filter=binom_filter, _from=_from, _to=_to, 
         start=start, step=step, stop=stop, daily_alignment=daily_alignment): 
    print(currency)
    time1 = time.time()
    candles = get_candles(currency, granularity, _from, _to, ) 
    steps, sequences = create_sequences(start, stop, step, seq_len) 
    up, down, udo, udnext, udmin = get_ud_bars(candles, steps)
    rwo, rwmin = get_position_bars(candles, position)
    df = crawl_bars(candles, sequences, steps, position, udo, udmin,  
                    udnext, rwo, rwmin)
    print('Run time: {:.2f}'.format((time.time() - time1) / 60))
    return df, steps, sequences, udo, rwo, candles



if __name__ == '__main__d': 
    
    if run_program == 0:
        results = pd.DataFrame()
        time1 = time.time()
        path = 'results'
        for currency in currencies:
            try:
                print(currency)
                time1 = time.time()
                candles = get_candles(currency, granularity, _from, _to, ) 
                steps, sequences = create_sequences(start, stop, step, seq_len) 
                up, down, udo, udnext, udmin = get_ud_bars(candles, steps)
                rwo, rwmin = get_position_bars(candles, position)
                df = crawl_bars(candles, sequences, steps, position, udo, udmin,  
                                udnext, rwo, rwmin)
                print('Run time: {:.2f}'.format((time.time() - time1) / 60))
                df = places_filter_by_day(df, 'binom', candles)
                df.insert(0, 'granularity', granularity)
                df.insert(0, 'pair', currency)
                df.avg_bars = df.avg_bars.round(0)
                df.win_perc = df.win_perc.round(2)
                df.to_pickle('{}_pickle'.format(currency))                
                results = results.append(df)
                df.drop('places', inplace=True, axis=1)
                with open('portfolio.py', 'a') as f:
                    for each in df.values.tolist():
                        f.write('{},\n'.format(each))
            except Exception as e:
                print('Could not append df for {}: {}'.format(currency, e))
        print('Total run time: {}'.format(time.time() - time1))
        file = 'seq_len_{}_on_{}_steps_{}_{}_{}_{}'
        results.to_pickle(file.format(seq_len,str(date.today()),start,
                                      stop, step,
                                      'filtered_on_bin_exact_then_win_perc'))

    elif run_program == 1:
        df, steps, sequences, udo, rwo, candles = main()
#        df.insert(0, 'pair', currency)
#        df.insert(0, 'granularity', granularity)
#        df.to_pickle('pickled_results')

        # df, places = further_process_results(df)
        # create_portfolio_output(df)
    elif run_program == 2:
        from indicators import stochastic_oscillator
        from matplotlib import pyplot as plt
        # Variables 
        currency = 'USD_CAD'
        granularity = 'H1'
        _from = '2006-01-01T00:00:00Z'
        _to =   '2019-01-01T00:00:00Z' 
        outcomes = (1, 0, 0, 1,  1)
        sequence = ((375, 375), (175, 275), (275, 75), (175, 375), (175, 375))
        position = (75, 75)
        direction = 0
        # Indicators
        so_k = 15
        so_d = 15
        # Normal Setup
        steps = np.array(sorted(list(set([25] + list(np.unique(sequence))))))
        seq_len = len(sequence)
        candles = get_candles(currency, granularity, _from, _to, ) 
        up, down, udo, udnext, udmin = get_ud_bars(candles, steps)
        rwo, rwmin = get_position_bars(candles, position)
        # Get Placements and calculate simple statistics
        take = get_positions(candles, steps, sequence, outcomes, udo, udnext)
        results = rwo[direction][take]
        win_perc = results.mean()
        binom = binom_test(results.sum(), results.shape[0], rwo[direction].mean())
        volume = candles.loc[take, 'volume']
        spreads = candles.loc[take, 'spread']
        # Compute indicators
        so = stochastic_oscillator(candles, so_k, so_d)  # 10 5 seems descent so far
        # Print Results
        plt.plot(so[take], results, 'o')
        if direction == 0:
            so10 = rwo[direction][take][so[take] >= 90]
            so20 = rwo[direction][take][so[take] >= 80]
            so30 = rwo[direction][take][so[take] >= 70]
        else:
            so10 = rwo[direction][take][so[take] <= 10]
            so20 = rwo[direction][take][so[take] <= 20]
            so30 = rwo[direction][take][so[take] <= 30]    
        print('timestamp, outcome, bars, stoch osc, spread, volume')
        for each in list(zip(candles.loc[take,'timestamp'], 
                             results, 
                             rwmin[direction][take], 
                             so[take],
                             spreads,
                             volume)): 
            print('{}\t{}\t{}\t{:.0f}\t{:.4f}\t{}'.format(each[0], each[1], 
                                                      each[2], each[3], 
                                                      each[4], each[5]))
        print()
        print('placements: {}'.format(take.shape[0]))
        print('win perc:   {}'.format(results.mean()))
        print('Binom:      {:.6f}'.format(binom))
        
        print('so 10: {:.2f}, {}, {:.2f}'.format(so10.mean(),
                                             so10.shape[0],
                                             binom_test(so10.sum(),
                                                        so10.shape[0],
                                                        results.mean())))
        print('so 20: {:.2f}, {}, {:.2f}'.format(so20.mean(),
                                     so20.shape[0],
                                     binom_test(so20.sum(),
                                                so20.shape[0],
                                                results.mean())))
        print('so 30: {:.2f}, {}, {:.2f},'.format(so30.mean(),
                                             so30.shape[0],
                                             binom_test(so30.sum(),
                                                        so30.shape[0],
                                                        results.mean())))
    else:
        currency = 'AUD_CAD'
        granularity = 'H1'
        _from = '2006-01-01T00:00:00Z'
        _to =   '2019-01-01T00:00:00Z' 
        # Normal Setup
        steps = np.array(sorted(list(set([25] + list(np.unique(sequence))))))
        seq_len = len(sequence)
        candles = get_candles(currency, granularity, _from, _to, ) 
        up, down, udo, udnext, udmin = get_ud_bars(candles, steps)
        rwo, rwmin = get_position_bars(candles, position)
        
            
            
            
            
            
        