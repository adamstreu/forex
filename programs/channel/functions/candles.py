#!python3
import pandas as pd
import numpy as np
import time
import yaml
import importlib
import pickle
import oandapyV20
import oandapyV20.endpoints.orders as orders
import os; os.chdir('/forex')
from programs.channel.functions.logs import log_ping, log_eval, log_placement
from libraries.oanda import get_candles_by_count
from programs.channel.functions.channels import channel_statistics
import programs.channel.configs.portfolio as ports


def check_granularity(granularity, timestamp, da=17):
    env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
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


def create_order(instrument, quantity, target, stop, direction):
    env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
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
    print(data)
    r = orders.OrderCreate(accountid, data=data)
    client.request(r)    
    return int(r.response['orderCreateTransaction']['id'])


def calculate_placement(candles, model):
    if model['direction'] == 1:
#        target = candles.bidclose.values[-1] + (.0001 *  ind[4][0])
#        stop = candles.bidclose.values[-1] - (.0001 * ind[4][1])
        target = candles.midclose.values[-1] + model['up_width']
        stop   = candles.midclose.values[-1] - model['down_width']
        qty    = 1000
    else:
#        target = candles.askclose.values[-1] - (.0001 *  ind[4][1])
#        stop = candles.askclose.values[-1] + (.0001 * ind[4][0])
        target = candles.midclose.values[-1] - model['down_width']
        stop   = candles.midclose.values[-1] + model['up_width']
        qty    = -1000
    return {'target': target, 'stop': stop, 'quantity': qty}


def calculate_quantity(ind, target, stop):
    if ind[5] == 1:
        qty = 1000
    else:
        qty = -1000
    return qty       


def evaluate_positions(timestamp):
    importlib.reload(ports)    
    models, currencies = ports.portfolio()    
    for model in models:
        if check_granularity('M5', timestamp):
            candles = get_candles_by_count(model['instrument'], 'M5', model['window'])
            cs = channel_statistics(candles.midclose.values, 'M5', candles)
            mod = pickle.load(open(model['file_location'], 'rb'))
            prediction = mod.predict(np.array(cs[model['df']]).reshape(1, -1))
            if prediction[0] == model['direction']:
                placements = calculate_placement(candles, model)
                order = create_order(model['instrument'],
                                     placements['quantity'],
                                     placements['target'],
                                     placements['stop'],
                                     model['direction'])
                log_placement(order, 
                              candles, 
                              model, 
                              placements['quantity'], 
                              placements['target'], 
                              placements['stop'])
                log_eval('Placement:\t{}'.format(model))
            else:
                log_eval('Nothing on {}'.format(model['instrument']))

            
def watch_candles():
    while True:
        # Compare latest fetched candle with last logged candle.
        env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
        most_recent_candle = get_candles_by_count('EUR_USD', 
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
            yaml.safe_dump(env, open('/forex/programs/channel/configs/env.yaml', 'w'), 
                           default_flow_style=False)
        time.sleep(env['ping_oanda_candles_interval'])



if __name__ == '__main__':
    log_ping('\n\n')
    log_eval('\n\n')
    watch_candles()
   