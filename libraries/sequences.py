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
run_program = 1  # remote = 0, local =  1, instance = 2
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
    client = 'client=oandapyV20.API(access_token=env['client'])'
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



if __name__ == '__main__': 
    pass
            
            
            
            
            
        
