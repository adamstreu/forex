import sqlite3
import yaml
import datetime
import boto3

    
def send_text(msg):
    env = yaml.safe_load(open('/seq/seq3/configs/env.yaml','r'))
    if env['text_on'] == 1:
        client = boto3.client("sns",
                              aws_access_key_id=env['aws_access_key_id'],
                              aws_secret_access_key=env['aws_secret_access_key'],
                              region_name=env['aws_region'])
        client.publish(PhoneNumber=env['phone_number'], Message=msg)
        

def log_eval(line):
    env = yaml.safe_load(open('/seq/seq3/configs/env.yaml','r'))
    eval_log=env['eval_log']
    log = open(eval_log, 'a')
    log.write('\t{}\t{}\n'.format(datetime.datetime.now().replace(microsecond=0), line))
    log.close()
    
    
def log_ping(line):
    env = yaml.safe_load(open('/seq/seq3/configs/env.yaml','r'))
    ping_log=env['ping_log']
    log = open(ping_log, 'a')
    log.write('\t{}\t{}\n'.format(datetime.datetime.now().replace(microsecond=0), line))
    log.close()
    
    
def log_placement(order, candles, ind, qty, target, stop):
    env = yaml.safe_load(open('/seq/seq3/configs/env.yaml','r'))
    conn = sqlite3.connect(env['conn'])
    cur = conn.cursor() 
    values = (int(order), 
              str(candles.timestamp.values[-1]),
              str(ind[0]),
              str(ind[1]),
              str(ind[2]),
              str(ind[3]), 
              str(ind[4]),
              ind[5],
              ind[6],
              ind[7],
              ind[8],
              ind[9], 
              ind[10],
              qty,
              float(candles.askclose.values[-1]),
              target, 
              stop)
    try:
        if ind[5] == 1:
            direct = 'long'
        else:
            direct = 'short'
        cur.execute('insert into {} ( '
                    'order_id, '
                    'timestamp, '
                    'pair, '
                    'granularity, '
                    'sequence, '
                    'outcomes, '
                    'position, '
                    'direction, '
                    'placements, '
                    'win_perc, '
                    'avg_bars, '
                    'p_win_perc, '
                    'binom, '
                    'quantity, '
                    'askclose, '
                    'target, '
                    'stop) '
                    'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);'.format(direct), values)
        conn.commit()
        cur.close()
        conn.close() 
        send_text(values)
        log_eval('Logging Placement: {}'.format(values))
    except Exception as e:
        send_text('Could not log Placement: {}, {}'.format(e, ind))
        log_eval('Could not log Placement: {}, {}'.format(e, ind))


def log_outcomes_short(trans):
    env = yaml.safe_load(open('/seq/seq3/configs/env.yaml','r'))
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
    env = yaml.safe_load(open('/seq/seq3/configs/env.yaml','r'))
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


       

