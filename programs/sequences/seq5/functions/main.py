import sqlite3
import yaml
from threading import Thread
from outcomes_long import watch_outcomes_long
from outcomes_short import watch_outcomes_short
from candles import watch_candles


def init_db():
    ''' Create db in env location if not exiting.  Do nothing otherwise '''
    env = yaml.safe_load(open('/seq/seq5/configs/env.yaml','r'))
    conn = sqlite3.connect(env['conn'])
    cur = conn.cursor()  
    for direction in ['long', 'short']:
        cur.execute('create table if not exists {} ('
                    'order_id integer primary key, '
                    'timestamp text not null, '
                    'pair text not null, '
                    'granularity text not null, '
                    'sequence text not null, '
                    'outcomes text not null, '
                    'position text not null, '
                    'direction int not null, '
                    'placements int not null, '
                    'win_perc real not null, '
                    'avg_bars real not null, '
                    'p_win_perc real not null, '
                    'binom real not null, '
                    'quantity integer not null, '
                    'askclose real not null, '
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
    env = yaml.safe_load(open('/seq/seq5/configs/env.yaml','r'))
    log = open(env['eval_log'], 'w')
    log.write('\nStarting seq/seq5.\n\n')
    log.close()
    log = open(env['ping_log'], 'w')
    log.write('\nStarting seq/seq5.\n\n')
    log.close()
    
    
if __name__ == '__main__':
    init_db()
    clear_logs()
    Thread(target = watch_outcomes_long).start()
    Thread(target = watch_outcomes_short).start()
    Thread(target = watch_candles).start()   
