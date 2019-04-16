#!python3
import sqlite3
import yaml
import datetime
#import boto3

    
def send_text(msg):
    pass
#    env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
#    if env['text_on'] == 1:
#        client = boto3.client("sns",
#                              aws_access_key_id=env['aws_access_key_id'],
#                              aws_secret_access_key=env['aws_secret_access_key'],
#                              region_name=env['aws_region'])
#        client.publish(PhoneNumber=env['phone_number'], Message=msg)
#        

def log_eval(line):
    env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
    eval_log=env['eval_log']
    log = open(eval_log, 'a')
    log.write('\t{}\t{}\n'.format(datetime.datetime.now().replace(microsecond=0), line))
    log.close()
    
    
def log_ping(line):
    env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
    ping_log=env['ping_log']
    log = open(ping_log, 'a')
    log.write('\t{}\t{}\n'.format(datetime.datetime.now().replace(microsecond=0), line))
    log.close()
    
    
def log_placement(order, candles, d, qty, target, stop):
    env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
    conn = sqlite3.connect(env['conn'])
    cur = conn.cursor() 
    values = (int(order), 
              str(candles.timestamp.values[-1]),
              str(d['instrument']),
              str(d['model']),
              str(d['df']),
              str(d['filter']),
              # str(d['granularity']),
              d['window'], 
              d['down_width'],
              d['up_width'],
              d['direction'],
              str(d['analysis_date']),
              qty,
              float(candles.midclose.values[-1]),
              target, 
              stop)
    try:
        if d['direction'] == 1:
            direct = 'long'
        else:
            direct = 'short'
        cur.execute('insert into {} ( '
                    'order_id, '
                    'timestamp, '
                    'instrument, '
                    'model, ' 
                    'df, '
                    'filter, '
                    # 'granularity, '
                    'window, '
                    'down_width, '
                    'up_width, '
                    'direction, '
                    'analysis_date, '
                    'quantity, '
                    'midclose, '
                    'target, '
                    'stop) '
                    'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);'.format(direct), values)
        conn.commit()
        cur.close()
        conn.close() 
        send_text(values)
    except Exception as e:
        send_text('Could not log Placement: {}, {}'.format(e, d))
        log_eval('Could not log Placement: {}, {}'.format(e, d))


def log_outcomes_short(trans):
    env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
    conn = sqlite3.connect(env['conn'])
    cur = conn.cursor()  
    try:
        if trans['reason'] == 'MARKET_ORDER':
            values = (int(trans['id']),
                      float(trans['price']),
                      int(trans['batchID']))
            cur.execute('UPDATE short SET '
                        'market_id = ?, '
                        'market = ? '
                        'WHERE order_id = ?;', values)
        if trans['reason'] == 'STOP_LOSS_ORDER' or trans['reason'] == 'TAKE_PROFIT_ORDER':
            if trans['reason'] == 'STOP_LOSS_ORDER':
                outcome = 0
            else:
                outcome = 1
            values = (int(trans['id']),
                      outcome,
                      float(trans['price']),
                      trans['time'], 
                      float(trans['pl']),
                      int(trans['tradesClosed'][0]['tradeID']))
            cur.execute('UPDATE short SET '
                        'outcome_id = ?, '
                        'outcome = ?, '
                        'outcome_price = ?, '
                        'outcome_timestamp = ?, '
                        'outcome_pl = ? '                        
                        'WHERE market_id = ?;', values)
        conn.commit()
        cur.close()
        conn.close() 
        send_text(values)
        log_eval('Logging outcomes short:\t{}'.format(values))
    except Exception as e:
        send_text('Could not log outcomes order on short {}, {}'.format(e, trans))
        log_eval('Could not log outcomes order on short  {}, {}'.format(e, trans))
        
        
def log_outcomes_long(trans):
    env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
    conn = sqlite3.connect(env['conn'])
    cur = conn.cursor()  
    try:
        if trans['reason'] == 'MARKET_ORDER':
            values = (int(trans['id']),
                      float(trans['price']),
                      int(trans['batchID']))
            cur.execute('UPDATE long SET '
                        'market_id = ?, '
                        'market = ? '
                        'WHERE order_id = ?;', values)
        elif trans['reason'] == 'STOP_LOSS_ORDER' or trans['reason'] == 'TAKE_PROFIT_ORDER':
            if trans['reason'] == 'STOP_LOSS_ORDER':
                outcome = 0
            else:
                outcome = 1
            values = (int(trans['id']),
                      outcome,
                      float(trans['price']),
                      trans['time'], 
                      float(trans['pl']),
                      int(trans['tradesClosed'][0]['tradeID']))
            cur.execute('UPDATE long SET '
                        'outcome_id = ?, '
                        'outcome = ?, '
                        'outcome_price = ?, '
                        'outcome_timestamp = ?, '
                        'outcome_pl = ? '                        
                        'WHERE market_id = ?;', values)
        else:
            log_eval('Outcome Recieved not market order or stop or loss:\t{}'.format(trans['reason']))            
        conn.commit()
        cur.close()
        conn.close() 
        send_text(values)
        log_eval('Logging outcomes long:\t{}'.format(values))
    except Exception as e:
        send_text('Could not log outcomes order on long {}, {}'.format(e, trans))
        log_eval('Could not log outcomes order on long  {}, {}'.format(e, trans))


if __name__ == '__main__':
    pass


       

