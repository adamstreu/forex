import pandas as pd
import numpy as np
import time
import yaml
from itertools import product
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
from logs import log_ping, log_eval, log_placement
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import datetime
from dateutil.relativedelta import relativedelta


daily_alignment = 17
candle_bars = 2500

'''
Notes:
    It might be silly ( and very expensive at some point), 
    to calculate all the stspe for an indicator when not all will be used
'''



def get_sequence_candles(instrument, granularity, count, da = daily_alignment):
    env = yaml.safe_load(open('/seq/seq5/configs/env.yaml','r'))
    client = oandapyV20.API(access_token=env['client'])
    params = {'count': count,
              'granularity': granularity,
              'price': 'M',
              'alignmentTimezone': 'UTC', #'America/Los_Angeles',
              'dailyAlignment': da}

    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    coll = client.request(r)
    # Remove last candle if not complete.
    if coll['candles'][-1]['complete'] == False:
        coll['candles'].pop()
    # Assemble Dataframe
    high = []
    low = []
    close = []
    timestamp = []
    for i in range(len(coll['candles'])):
        high.append(float(coll['candles'][i]['mid']['h']))
        low.append(float(coll['candles'][i]['mid']['l']))
        close.append(float(coll['candles'][i]['mid']['c']))
        timestamp.append(coll['candles'][i]['time'])
    # Assemble DataFrame.  Cast Values.
    df = pd.DataFrame(pd.to_datetime(timestamp))
    df.columns = ['timestamp']
    df['high'] = pd.to_numeric(high)
    df['low'] = pd.to_numeric(low)
    df['close'] = pd.to_numeric(close)
    return df


def get_placement_candles(instrument, granularity, count, 
                          da = daily_alignment):
    env = yaml.safe_load(open('/seq/seq5/configs/env.yaml','r'))
    client = oandapyV20.API(access_token=env['client'])
    params = {'count': count,
              'granularity': granularity,
              'price': 'AB',
              'alignmentTimezone': 'UTC', #'America/Los_Angeles',
              'dailyAlignment': da}
              
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    coll = client.request(r)
    # Remove last candle if not complete.
    if coll['candles'][-1]['complete'] == False:
        coll['candles'].pop()
    # Assemble Dataframe
    bidopen = []
    bidlow = []
    bidhigh = []
    bidclose = []
    askopen = []
    asklow = []
    askhigh = []
    askclose = []
    timestamp = []
    for i in range(len(coll['candles'])):
        bidopen.append(float(coll['candles'][i]['bid']['o']))
        bidhigh.append(float(coll['candles'][i]['bid']['h']))
        bidlow.append(float(coll['candles'][i]['bid']['l']))
        bidclose.append(float(coll['candles'][i]['bid']['c']))
        askopen.append(float(coll['candles'][i]['ask']['o']))
        askhigh.append(float(coll['candles'][i]['ask']['h']))
        asklow.append(float(coll['candles'][i]['ask']['l']))
        askclose.append(float(coll['candles'][i]['ask']['c']))
        timestamp.append(coll['candles'][i]['time'])
    # Assemble DataFrame.  Cast Values.
    df = pd.DataFrame(pd.to_datetime(timestamp))
    df.columns = ['timestamp']
    df['bidopen'] = pd.to_numeric(bidopen)
    df['bidhigh'] = pd.to_numeric(bidhigh)
    df['bidlow'] = pd.to_numeric(bidlow)
    df['bidclose'] = pd.to_numeric(bidclose)
    df['askopen'] = pd.to_numeric(askopen)
    df['askhigh'] = pd.to_numeric(askhigh)
    df['asklow'] = pd.to_numeric(asklow)
    df['askclose'] = pd.to_numeric(askclose)
    df['spread'] = df.askclose - df.bidclose
    return df


def dict_from_list():
    import sys
    sys.path.insert(0, '/seq/seq5/configs/')
    from portfolio import indicators_list
    indicators = indicators_list()
    d = {}
    for ind in indicators:
        if ind[0] not in d:
            d[ind[0]] = {}
        if ind[1] not in d[ind[0]]:
            d[ind[0]][ind[1]] = {'positions': [], 'steps': []}
        d[ind[0]][ind[1]]['positions'].append(ind)
        [d[ind[0]][ind[1]]['steps'].extend([x[0], x[1]]) for x in ind[2]]
    for k in d.keys():
        for g in d[k].keys():
            d[k][g]['steps'] = sorted(list(set(d[k][g]['steps'])))
    return d


def get_bars(candles, steps, bar_limit=candle_bars):
    zipped = list(product(range(len(steps)), ['up', 'down']))
    up = []
    down = []
    for each in zipped:
        target = steps[each[0]]
        direction = each[1]
        ind = []
        df = candles.copy()
        # Set target values on up
        if direction == 'up':
            df['target'] = (df.close) + (.0001 * target)
            dfv = df.values
            for i in range(dfv.shape[0]):
                tmp_ind = np.where(dfv[i+1:, 1] >= dfv[i, 4])
                if tmp_ind[0].shape[0] != 0:
                    ind.append(tmp_ind[0][0] + 1)
                else:
                    ind.append(bar_limit)
        # Set target values on down
        else:
            df['target'] = (df.close) - (.0001 * target)
            dfv = df.values
            for i in range(dfv.shape[0]):
                tmp_ind = np.where(dfv[i+1:, 2] <= dfv[i, 4])
                if tmp_ind[0].shape[0] != 0:
                    ind.append(tmp_ind[0][0] + 1)
                else:
                    ind.append(bar_limit)
        # put results into appropriate manager_dictionary
        if direction == 'up':
            up.append(list(ind))
        else:
            down.append(list(ind))
    up = np.array(up)
    down = np.array(down)
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


def check_granularity(granularity, timestamp, da=daily_alignment):
    env = yaml.safe_load(open('/seq/seq5/configs/env.yaml','r'))
    timestamp = timestamp - pd.Timedelta(hours=da)
    minutes_of_day = (timestamp.hour * 60 ) + timestamp.minute 
    if env['granularity'].strip().lower()[0] == 'm':
        plus_end_of_candle = int(env['granularity'][1:].strip())
    else:  
        plus_end_of_candle = int(env['granularity'][1:].strip()) * 60
    if granularity[0].strip().lower() == 'm':
        granularity_minutes = int(granularity[1:].strip())
    else:
        granularity_minutes = int(granularity[1:].strip()) * 60
    if (minutes_of_day + plus_end_of_candle) % granularity_minutes == 0:
        return True
    else:
        return False
    


def crawl_bars(candles, steps, keys, outcomes, udo, udmin, udnext):
    seq_len = len(keys)
    # Do first step in sequence.
    u = steps.index(keys[0][0])
    d = steps.index(keys[0][1])
    o = outcomes[0]
    take = (udnext[u][d] * udo[u][d][o])  
    take = take[take != 0]
    take = np.unique(take)
#    print(take)
    # Calculate remaining steps is sequence.
    if seq_len > 1:
        for s in range(1, seq_len):
            u = steps.index(keys[s][0])
            d = steps.index(keys[s][1])
            o = outcomes[s]
            where = (udnext[u][d] * udo[u][d][o])    
            take = np.take(where, take)
            take = take[take != 0]
            take = np.unique(take) 
    return take


def create_order(instrument, quantity, target, stop, direction):
    env = yaml.safe_load(open('/seq/seq5/configs/env.yaml','r'))
    if direction == 1:
        accountid=env['accountid_long']        
    else:    
        accountid=env['accountid_short']
    client=oandapyV20.API(access_token=env['client'])
    data = {'order' : {"units": quantity, 
                       "instrument": instrument, 
                       "timeInForce": "FOK", 
                       "type": "MARKET", 
                       "positionFill": "DEFAULT",
                       'takeProfitOnFill' : {'price': str(round(target, 5)), 
                                             'timeInForce' : 'GTC'},
                       'stopLossOnFill': {'price': str(round(stop, 5)), 
                                          'timeInForce' : 'GTC'}}}
    r = orders.OrderCreate(accountid, data=data)
    client.request(r)    
    return int(r.response['orderCreateTransaction']['id'])


def calculate_placement(ind, candles):
    if ind[5] == 1:
        target = candles.bidclose.values[-1] + (.0001 *  ind[4][0])
        stop = candles.bidclose.values[-1] - (.0001 * ind[4][1])
    else:
        target = candles.askclose.values[-1] - (.0001 *  ind[4][1])
        stop = candles.askclose.values[-1] + (.0001 * ind[4][0])
    return target, stop


def calculate_quantity(ind, target, stop):
    if ind[5] == 1:
        qty = 100
    else:
        qty = -100
    return qty       


def evaluate_positions(timestamp):
    d = dict_from_list()                       
    for i in d.keys(): 
        for g in d[i].keys():                   
            if check_granularity(g, timestamp):                 
                candles = get_sequence_candles(i, g, candle_bars) # had a +1 here
                steps = d[i][g]['steps']
                up, down, udo, udnext, udmin = get_bars(candles, steps) 
                for ind in d[i][g]['positions']:
                    placements = crawl_bars(candles, steps, ind[2], ind[3],
                                            udo, udmin, udnext)
                    if candle_bars - 2 in placements:
                        placement_candles = get_placement_candles(i, g, 2)
                        target, stop = calculate_placement(ind,
                                                            placement_candles)
                        qty = calculate_quantity(ind, target, stop)
                        order = create_order(i, qty, target, stop, ind[5])
                        log_placement(order, placement_candles, ind,  
                                      qty, target, stop)
                        log_eval('Placement: {}, {}\t{} \t>>>\t>>>'.format(i, g, timestamp))
                    else:
                        log_eval('Nothing: {}, {}'.format(i, g))
                        pass
            else:
                log_eval('Granularity check not passed: {}, {}\t{}'.format(i, 
                                                                    g, 
                                                                    timestamp))

            
def watch_candles():
    while True:
        # Compare latest fetched candle with last logged candle.
        env = yaml.safe_load(open('/seq/seq5/configs/env.yaml','r'))
        most_recent_candle = get_placement_candles('EUR_USD', 
                                                   env['granularity'], 2)
        # If candle is larger, call placements.  Otherwise, slee & try again.
        if most_recent_candle.iloc[-1,0] <= \
                                   pd.to_datetime(env['latest_timestamp']):
           msg = 'Waiting on new candle.\t{}'
           log_ping(msg.format(most_recent_candle.iloc[-1,0]))
        else:     
            msg = 'New Candle Found.\t{}\t\n< < < '
            log_ping(msg.format(most_recent_candle.iloc[-1,0]))
            log_eval(msg.format(most_recent_candle.iloc[-1,0]))
            evaluate_positions(most_recent_candle.iloc[-1, 0])
            env['latest_timestamp'] = str(most_recent_candle.iloc[-1, 0])
            yaml.safe_dump(env, open('/seq/seq5/configs/env.yaml', 'w'), 
                           default_flow_style=False)
        time.sleep(env['ping_oanda_candles_interval'])



if __name__ == '__main__':
    log_ping('\n\n')
    log_eval('\n\n')
    watch_candles()
   