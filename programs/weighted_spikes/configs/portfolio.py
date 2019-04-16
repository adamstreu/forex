def portfolio():
    

    
    port = {
        'EUR_USD': [{'spike': -0.00045, 'window': 25, 'target': .001, 'loss': .005},
                    {'spike': -0.00045, 'window': 25, 'target': .001, 'loss': .005}],
        'AUD_CAD': [{'spike': -0.00045, 'window': 25, 'target': .001, 'loss': .005},
                    {'spike': -0.00045, 'window': 25, 'target': .001, 'loss': .005}],
        'GBP_NZD': [{'spike': -0.00045, 'window': 25, 'target': .001, 'loss': .005},
                    {'spike': -0.00045, 'window': 25, 'target': .001, 'loss': .005}]
             }
    
    
    
    
    
    
    
    

    return port# {'portfolio': breakouts, 
#            'accounts':  accounts,
#            'accounts_list': accounts_list,
#            'accounts_direction':   accounts_direction}     



if __name__ == '__main__':
    pass

    """

    
    
    
    # Main portfolio
    ###########################################################################
    breakouts = {
        'EUR_USD': {      # Make sure gran/window/direction matches with account
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.31},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.31}
                }
            },
            'M5' : {
                200:  {
                    'top':    {'target': 6, 'stop': 1, 'position': 1.616},
                    'bottom': {'target': 4, 'stop': 4, 'position': -100}
                }
            }
        },
        'EUR_AUD': {
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.41},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.38}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0}}
#                }
#            }
        },             
        'AUD_CAD': {
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.136},
                    'bottom': {'target': 20, 'stop': 1, 'position': -0.065}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        },        
        'EUR_CHF': {
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.065},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.065}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        },             
        'EUR_GBP': {
            'M1' : {
                500: {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.065},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.065}
                }
            }#,
#            'M5' : {
#                500:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        },
        'GBP_CHF': {
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.065},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.065}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        }, 
        'GBP_USD': {
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.13},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.24}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        }, 
        'NZD_USD': {
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.065},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.1}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        }, 
        'USD_CAD': {
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.42},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.13}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        }, 
        'USD_CHF': {
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.1},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.27}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        }, 
        'EUR_NZD': {
            'M1' : {
                500:  {
                    'top':    {'target': 19, 'stop': 1, 'position': 1.03},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.065}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        }, 
        'EUR_SGD': {
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.03},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.042}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        }, 
        'EUR_CAD': {
            'M1' : {
                500:  {
                    'top':    {'target': 20, 'stop': 1, 'position': 1.03},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.13}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        },
        'USD_SGD': {
            'M1' : {
                500:  {
                    'top':    {'target': 15, 'stop': 1, 'position': 1.03},
                    'bottom': {'target': 15, 'stop': 1, 'position': -.1}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        },
        'GBP_AUD': {
            'M1' : {
                500:  {
                    'top':    {'target': 27, 'stop': 1, 'position': 1.065},
                    'bottom': {'target': 20, 'stop': 1, 'position': -.1}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        },
        'AUD_USD': {
            'M1' : {
                500:  {
                    'top':    {'target': 19, 'stop': 1, 'position': 1.45},
                    'bottom': {'target': 18, 'stop': 1, 'position': -.1}
                }
            }#,
#            'M5' : {
#                250:  {
#                    'top':    {'target': 4, 'stop': 4, 'position': 1, 'position_filter': 100, 'range_filter': 0},
#                    'bottom': {'target': 4, 'stop': 4, 'position': 0, 'position_filter': 100, 'range_filter': 0}
#                }
#            }
        }, 
        

    }

    
    
    # Use this for testing.  RElease to the accounts above.
    ###########################################################################
    # CURRENTLY CANNOT MAKE CHANGES TO THIS WHILE PROGRAM IS RUNNING
    accounts_testing = {
        'M1': {              
            250:  {'long':  '101-001-7518331-001', 'short': '101-001-7518331-001'},
            500:  {'long':  '101-001-7518331-014', 'short': '101-001-7518331-001'},
            1000: {'long':  '101-001-7518331-001', 'short': '101-001-7518331-001'}
        },
        'M5': {
            250:  {'long':  '101-001-7518331-001', 'short': '101-001-7518331-001'},
            500:  {'long':  '101-001-7518331-015', 'short': '101-001-7518331-001'}
        }
    }
     
    
    # Use for production.  Organize what portfolios are connected to what accounts
    ###########################################################################
    
    accounts = {
        'M1': {              
            250:  {'long':  '101-001-7518331-002', 'short': '101-001-7518331-003'},
            500:  {'long':  '101-001-7518331-004', 'short': '101-001-7518331-005'},
            1000: {'long':  '101-001-7518331-006', 'short': '101-001-7518331-007'},
        },
        'M5': {
            200:  {'long':  '101-001-7518331-012', 'short': '101-001-7518331-013'},
            250:  {'long':  '101-001-7518331-008', 'short': '101-001-7518331-009'}
        },
        'M15': {
            125:  {'long':  '101-001-7518331-010', 'short': '101-001-7518331-011'}
        }
    }
     
        
    # helper function - just lists accounts ( used for watching transactions)
    ###########################################################################
    accounts_list = [] 
    for gran in accounts.keys():
        for window in accounts[gran].keys():
            for direction in accounts[gran][window].keys():
                accounts_list.append(accounts[gran][window][direction])
    
    
    # Helper funciton - just lists accounts with direction ( watching trans.)
    ###########################################################################
    long  = []
    short = []
    for gran in accounts.keys():
        for window in accounts[gran].keys():
                long.append(accounts[gran][window]['long'])
                short.append(accounts[gran][window]['short'])
    accounts_direction = {'long': long, 'short': short}
        
    """