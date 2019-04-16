#!python3
import sqlite3
import yaml
from threading import Thread
import os; os.chdir('/forex')
from programs.channel_breakout.functions.transactions import watch_transactions
from programs.channel_breakout.functions.candles      import watch_candles


def init_db():
    ''' Create db in env location if not exiting.  Do nothing otherwise '''
    file = '/forex/programs/channel_breakout/configs/env.yaml'
    env = yaml.safe_load(open(file,'r'))
    conn = sqlite3.connect(env['conn'])
    cur = conn.cursor()  
    for direction in ['long', 'short']:
        cur.execute('create table if not exists {} ('
                    'order_id integer primary key, '
                    'account_id text not null, '
                    'timestamp text not null, '
                    'instrument text not null, '
                    'granularity text not null, '
                    'window integer not null, '
                    'target_ratio text not null, '
                    'stop_ratio text not null, '
                    'channel_range         real not null, '
                    'channel_position      real not null, '
                    'channel_largest_spike real not null, ' 
                    'channel_slope         real not null, ' 
                    'closings_slope real not null, '
                    'quantity integer not null, '
                    'midclose real not null, '
                    'target real not null, '
                    'stop real not null, '
                    'market_id integer, '
                    'market real, '
                    'outcome_id integer, '
                    'outcome integer, '
                    'outcome_price float, '
                    'outcome_timestamp text, ' 
                    'outcome_pl real);'.format(direction))   
    conn.commit()
    cur.close()
    conn.close()

    
def clear_logs():
    ''' Clear ping and eval logs '''
    file = '/forex/programs/channel_breakout/configs/env.yaml'
    env = yaml.safe_load(open(file,'r'))
    log = open(env['eval_log'], 'w')
    log.write('\nStarting channels.\n\n')
    log.close()
    log = open(env['ping_log'], 'w')
    log.write('\nStarting channels.\n\n')
    log.close()
    
    
if __name__ == '__main__':
    init_db()
    clear_logs()
    Thread(target = watch_transactions).start()
    Thread(target = watch_candles).start()   


