#!python3
import sqlite3
import yaml
import datetime
#import boto3

    
def send_text(msg):
    pass
#    env = yaml.safe_load(open('/forex/programs/channel_breakout/configs/env.yaml','r'))
#    if env['text_on'] == 1:
#        client = boto3.client("sns",
#                              aws_access_key_id=env['aws_access_key_id'],
#                              aws_secret_access_key=env['aws_secret_access_key'],
#                              region_name=env['aws_region'])
#        client.publish(PhoneNumber=env['phone_number'], Message=msg)
    

def log_eval(line):
    file = '/forex/programs/channel_breakout/configs/env.yaml'
    env = yaml.safe_load(open(file, 'r'))
    eval_log=env['eval_log']
    log = open(eval_log, 'a')
    log.write('\t{}\t{}\n'.format(datetime.datetime.now().replace(microsecond=0), line))
    log.close()
    
    
def log_ping(line):
    file = '/forex/programs/channel_breakout/configs/env.yaml'
    env = yaml.safe_load(open(file, 'r'))
    ping_log=env['ping_log']
    log = open(ping_log, 'a')
    log.write('\t{}\t{}\n'.format(datetime.datetime.now().replace(microsecond=0), line))
    log.close()

                            
def log_placement(order, candles, account_id, instrument, granularity, window, 
                  target_ratio, stop_ratio, channel_range, channel_position, 
                  spike, channel_slope, closings_slope, qty, target, stop):
    file = '/forex/programs/channel_breakout/configs/env.yaml'
    env = yaml.safe_load(open(file, 'r'))
    conn = sqlite3.connect(env['conn'])
    cur = conn.cursor() 
    if qty > 0:
        direction = 'long'
    else:
        direction = 'short'
    values = (order, 
              str(candles.timestamp.values[-1]),
              account_id,
              instrument,
              granularity,
              window,
              target_ratio,
              stop_ratio,
              channel_range,
              channel_position,
              spike, 
              channel_slope, 
              closings_slope,
              qty,
              float(candles.midclose.values[-1]),
              target, 
              stop)
    try:
        cur.execute('insert into {} ( '
                    'order_id, '
                    'timestamp, '
                    'account_id, '
                    'instrument, '
                    'granularity, '
                    'window, '
                    'target_ratio, '
                    'stop_ratio, '
                    'channel_range, '
                    'channel_position, '
                    'channel_largest_spike, ' 
                    'channel_slope, ' 
                    'closings_slope, '
                    'quantity, '
                    'midclose, '
                    'target, '
                    'stop) '
                    'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);'.format(direction), values)
        conn.commit()
        cur.close()
        conn.close() 
        send_text(values)
        log_eval('{} Placement for {} on account {}'.format(direction, instrument, account_id))
    except Exception as e:
        log_eval('Could not log Placement: {}, {}'.format(e, instrument))

        
def log_transactions(trans, account_id, direction):
    file = '/forex/programs/channel_breakout/configs/env.yaml'
    env = yaml.safe_load(open(file, 'r'))
    conn = sqlite3.connect(env['conn'])
    cur = conn.cursor()  
    try:
        if trans['reason'] == 'MARKET_ORDER':
            try:
                values = (int(trans['id']),
                          float(trans['price']),
                          int(trans['batchID']),
                          account_id)
                cur.execute('UPDATE {} SET '
                            'market_id = ?, '
                            'market = ? '
                            'WHERE order_id = ? and account_id = ?;'.format(direction), values)
                log_eval('saw placement but dont need to log > delete note later.')
                conn.commit()
                cur.close()
                conn.close() 
            except Exception as e:
                log_eval('Problem.  reason = market order'.format(account_id, e, trans))  
        if trans['reason'] == 'STOP_LOSS_ORDER' or trans['reason'] == 'TAKE_PROFIT_ORDER':
            try:
                if trans['reason'] == 'STOP_LOSS_ORDER':
                    outcome = 0
                else:
                    outcome = 1
                values =   (int(trans['id']),
                            outcome,
                            float(trans['price']),
                            trans['time'], 
                            float(trans['pl']),
                            int(trans['tradesClosed'][0]['tradeID']),
                            account_id)
                cur.execute('UPDATE {} SET '
                            'outcome_id = ?, '
                            'outcome = ?, '
                            'outcome_price = ?, '
                            'outcome_timestamp = ?, '
                            'outcome_pl = ? '                        
                            'WHERE market_id = ? and account_id = ?;'.format(direction), values)
                log_eval('{} on account {}'.format(trans['reason'], account_id))
                conn.commit()
                cur.close()
                conn.close() 
            except Exception as e:
                log_eval('PRoblem.  reason = stop loss or target'.format(account_id, e, trans))
        else:
            log_eval('saw another type of outcome: {}'.format(trans['reason']))
    except Exception as e:
        log_eval('how did it get here. {}, {}, {}'.format(account_id, e, trans))        


if __name__ == '__main__':
    pass


       

