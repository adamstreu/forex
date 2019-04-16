#!python3
import sqlite3
import yaml
from threading import Thread
import os; os.chdir('/northbend')
from programs.channel_breakout.functions.transactions import watch_transactions
from programs.channel_breakout.functions.candles      import watch_candles

    
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
    
    
if __name__ == '__main__':
    # init_db()
    clear_logs()
    # Thread(target = watch_transactions).start()
    Thread(target = watch_candles).start()   


