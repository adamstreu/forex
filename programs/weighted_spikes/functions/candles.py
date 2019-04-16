import pandas as pd
import numpy as np
import time
import yaml
import importlib
import os; os.chdir('/northbend')
from programs.channel_breakout.functions.logs import log_ping
from programs.channel_breakout.functions.logs import log_eval
from programs.channel_breakout.functions.logs import log_placement
from libraries.oanda import get_candles_by_count
from libraries.oanda import get_open_positions
from libraries.oanda import create_order
import programs.channel_breakout.configs.portfolio as ports
from classes.channel import Channel


def clear_logs():
    ''' Clear ping and eval logs '''
    file = '/northbend/programs/weighted_spikes/configs/env.yaml'
    env = yaml.safe_load(open(file,'r'))
    log = open(env['eval_log'], 'w')
    log.write('\nStarting channels.\n\n')
    log.close()
    log = open(env['ping_log'], 'w')
    log.write('\nStarting channels.\n\n')
    log.close()
    

def check_granularity(granularity, timestamp, da=17):
    file = '/northbend/programs/weighted_spikes/configs/env.yaml'
    env = yaml.safe_load(open(file, 'r'))
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


def calculate_placement(candles, instrument, channel, direction):
    avg_width = channel.channel_range / 6
    if direction == 'bottom': # long position
        target = candles.bidclose.values[-1] + (avg_width * instrument['bottom']['up'])
        stop   = candles.bidclose.values[-1] - (avg_width * instrument['bottom']['down']) 
        qty    = 1000
    else: # short position
        target = candles.asdclose.values[-1] - (avg_width * instrument['bottom']['down']) 
        stop   = candles.askclose.values[-1] + (avg_width * instrument['bottom']['up']) 
        qty    = -1000
    return {'target': target, 'stop': stop, 'quantity': qty}


def calculate_quantity(instrument, direction, target, stop):
    if direction == 1:
        qty = 100
    else:
        qty = -100
    return qty       


def evaluate_positions(timestamp):
    importlib.reload(ports)    
    port     = ports.portfolio()    
    accounts = port['accounts']
    print('Timestamp: {}'.format(timestamp))    
    for instrument in port['portfolio'].keys():
        print('\n---------Instrument-------------: {}'.format(instrument))
        for granularity in port['portfolio'][instrument].keys():
            print('---granularity---: {}'.format(granularity))
            print('Is granularity check passed: {}'.format(check_granularity(granularity, timestamp)))
            if check_granularity(granularity, timestamp):
                # get largest window in granularity
                largest_window = max(port['portfolio'][instrument][granularity].keys())
                print('Largest window: {}'.format(largest_window))
                print('-------')
                candles = get_candles_by_count(instrument, 
                                               granularity, 
                                               largest_window + 1)
                for window in port['portfolio'][instrument][granularity].keys():
                    print('--')
                    print('Window: {}'.format(window))
                    # Evaluate Channeal, profile and filters for window.
                    # Check for breakouts and filter passing
                    closings = candles.midclose.values[-(window):]
                    print('Closings shape: {}'.format(closings.shape))
                    channel = Channel(candles, candles.shape[0] - 1, window )
                    print('Channel position: {}'.format(channel.closing_position))
                    for direction in ['top', 'bottom']:
                        print('Direction: {}'.format(direction))
                        if direction == 'top':
                            breakout = 'short'
                            cond1 = channel.closing_position > port['portfolio'][instrument][granularity][window][direction]['position']
                        elif direction == 'bottom':
                            breakout = 'long'
                            cond1 = channel.closing_position < port['portfolio'][instrument][granularity][window][direction]['position']
                        print('Breakout Found: {}'.format(cond1))
                        cond2 = True
                        cond3 = True
                        cond4 = True
                        #move this belwo - just wanted it printed here.
                        account = accounts[granularity][window][breakout]
                        print('Account: {}'.format(account))
                        if cond1 and cond2 and cond3 and cond4:
                            # Get account for gran / wind0w / direction comb.   
                            # do not place if instrument is already an open pos.
                            print('Instrument not already  placed: {}'.format(instrument not in get_open_positions(account)))
                            if instrument not in get_open_positions(account):
                                # Get target and loss                                
                                target = port['portfolio'][instrument][granularity][window][direction]['target']
                                stop   = port['portfolio'][instrument][granularity][window][direction]['stop']
                                target *= channel.channel_range / 6                               
                                stop *= channel.channel_range / 6   
                                if direction == 'bottom': # long position
                                    target = candles.askclose.values[-1] + target
                                    stop   = candles.askclose.values[-1] - stop
                                    qty    = 100
                                else: # short position
                                    target = candles.bidclose.values[-1] - target
                                    stop   = candles.close.values[-1] + stop
                                    qty    = -100
                                # Create Order
                                order = create_order(instrument,
                                                     qty,
                                                     target,
                                                     stop,
                                                     account)
                                print('ORDER PLACED.')
                                print('target, stop, askclose: {}, {}, {}'.format(target, stop, qty, candles.askclose.values[-1]))
                                print('order number: {}'.format(order))
                                log_placement(order, 
                                              candles, 
                                              account, 
                                              instrument, 
                                              granularity, 
                                              window, 
                                              port['portfolio'][instrument][granularity][window][direction]['target'],
                                              port['portfolio'][instrument][granularity][window][direction]['stop'],               
                                              channel.channel_range, 
                                              channel.closing_position, 
                                              channel.largest_spike_5,
                                              channel.channel_slope,
                                              channel.closings_slope,
                                              qty, 
                                              target, 
                                              stop)
                            else:
                                msg =  '{} Breakout on {} found but position already open in account .'
                                log_eval(msg.format(direction, instrument, account))                      
    return
                            
                
def watch_candles():
    while True:
        # Compare latest fetched candle with last logged candle.
        env = yaml.safe_load(open('/northbend/programs/weighted_spikes/configs/env.yaml','r'))
        most_recent_candle = get_candles_by_count('EUR_USD', 
                                                   env['granularity'], 2)
        # If candle is larger, call placements.  Otherwise, slee & try again.
        if most_recent_candle.iloc[-1,0] <= \
                                   pd.to_datetime(env['latest_timestamp']):
           msg = 'Waiting on new candle set.\t{}'
           log_ping(msg.format(most_recent_candle.iloc[-1,0]))
        else:     
            msg = 'New Candles Found.\t{}\t\n< < < '
            log_ping(msg.format(most_recent_candle.iloc[-1,0]))
            evaluate_positions(most_recent_candle.iloc[-1, 0])
            env['latest_timestamp'] = str(most_recent_candle.iloc[-1, 0])
            yaml.safe_dump(env, open('/northbend/programs/weighted_spikes/configs/env.yaml', 'w'), 
                           default_flow_style=False)
        time.sleep(env['ping_oanda_candles_interval'])
    return


if __name__ == '__main__':
    clear_logs()
    watch_candles()
   