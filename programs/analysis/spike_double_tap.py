import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os; os.chdir('/northbend')


'''
    Research:
        Look at the autocorrelation bewteen all values.  
            either between double taps or within the tap window.
            
        Wiki on pearson corr coefeecient is wealth of data.
            test performing, analyssi,
            geometri inerpretations....
            
            
            
            
        PCA:
            sensitive to original scaling of data:
                carful for this both during and after collection and scaling
                mean centering and normalizing (or using z-scores).
                    normaize = each vecotr dot product equals one (length of each sample is one)
                        or - cause I don;t know - if each variable (column ) should have unit length
            use autocorrelation matrix insead of correlation matrix ?  ? 
            Often thought of as reveling the inernal strucutre of the data
                that best explains its variance
            Can reduce data to wo or three components for visualization
            PCA is a popular primary technique in pattern recognition. 
                It is not, however, optimized for class separability.
                The linear discriminant analysis is an alternative which is 
                    optimized for class separability. ( supervised lustering - forget it.)
                Assumes however that variables are norma l - mne ar eNOT
                    logistic regression does not assume so either




    Channels:
        To Do on channels as well:  PCA and Clustering.

Spike double tap:
    Define Well.
    Look for exceptions.
    Play only when it's right and don't expect any number of them.
    
    or
    
    define LOOSELY.  Only the Ver basics
    Collect large number from all times and currencies( defined by perc as necc)
    CLUSTER.  Groups similar ones together.
    Gather outcomes relative to tap shape
    describe local conditions for tap, L, R, C.
    Analysis and ML
    
    
    
Notes:
    espilon has a huge effect - keep loose and filter and analyze later
'''

# Import Candles and Outcomes
###############################################################################  
if False:
    folder = 'EUR_USD_M1_2014-01-11_2019-01-01'
    path   = '/Users/user/Desktop/' + folder + '/'
    long_target = pd.read_csv(path + 'long_target.csv')
    long_loss = pd.read_csv(path + 'long_loss.csv')
    short_loss = pd.read_csv(path + 'short_loss.csv')
    short_target = pd.read_csv(path + 'short_target.csv')
    candles = pd.read_csv(path + 'candles.csv')
    candles.set_index('location', drop=True, inplace=True)
    for df in [long_target, short_target, long_loss, short_loss]:
        df.set_index('location', inplace=True, drop=True)
    # Set Environment
    pd.set_option('max.columns', 15)
    pd.set_option('display.width', 1000)
    

###############################################################################
# DEFINE DOUBLE TAPS
###############################################################################
    
    
# Create Taps index
#------------------------------------------------------------------------------
# Parameters
local_max_interval = 20
# Look for rolling machtches from right and left within parameter
candles['midclose_flip'] = np.flip(candles.midclose.values, axis=0)
candles['left']          = candles.rolling(local_max_interval)\
                                           .midclose.max().values
candles['right']         = candles.rolling(local_max_interval).midclose_flip\
                                           .max().values
left_match               = candles.midclose.values == candles.left.values
right_match              = candles.midclose_flip.values == candles.right.values
candles['tap']           = left_match & np.flip(right_match, axis=0)
taps                     = candles.loc[candles.tap].index.values
# remove Taps next to each other

# Verify taps
if False:
    for i in range(100,129):
        plt.figure()
        candles.loc[taps[i] - (local_max_interval / 2 - 1): taps[i] - 1, 
                    'midclose'].plot()
        candles.loc[taps[i] - 1: taps[i] + 1,  'midclose'].plot()    
        candles.loc[taps[i]+ 1: taps[i] + local_max_interval / 2 - 1, 
                    'midclose'].plot()
        plt.show()
        print(taps[i])
   

    
# Create Double Taps index
#------------------------------------------------------------------------------  
# Parameters
epsilon       = .0001   # Thishasa hugeeffect on ourresult                             
max_distance  = 1500
seperate_min  = 10
# Match all taps together that meet the following criteria
double_taps = [] 
for tap in taps:
    # Within max_distance of each other, seperated by min
    cond1 = taps - tap <=  max_distance
    cond2 = taps - tap >   seperate_min
    conditions = cond1 & cond2
    for each in taps[conditions]:
        double_taps.append((tap, each))
double_taps = np.array(double_taps)
# Closing values of both taps Within epsilon of each other
diff      = candles.loc[double_taps[:,0],'midclose'].values \
          - candles.loc[double_taps[:,1],'midclose'].values
diff_perc = diff / candles.loc[double_taps[:,0],'midclose'].values
double_taps = double_taps[abs(diff_perc) < epsilon]
# The interval between taps contains no value higher than the lowest tap 
double_taps_coll = []
for i  in range(double_taps.shape[0]):
    minimum = np.minimum(candles.loc[double_taps[i, 0], 'midclose'], 
                         candles.loc[double_taps[i, 1], 'midclose'])
    cond = (candles.loc[double_taps[i, 0] + 1: double_taps[i, 1] - 1, 
                        'midclose'].values < minimum).mean() == 1 
    if cond:
        double_taps_coll.append(double_taps[i])
double_taps = np.array(double_taps_coll)
    
# Verify double taps
if False:
    for i in range(150, 151):
        # Verify Taps for each double:
        fig, ax = plt.subplots(3, 1, figsize=(14, 8))
        # sub 1
        ax[0].plot(candles.loc[double_taps[i, 0] - (local_max_interval / 2): \
                      double_taps[i, 0] + (local_max_interval / 2), 
                      'midclose'].values)
        ax[0].plot((local_max_interval / 2), candles.loc[double_taps[i, 0], 
                                                'midclose'], 'o' )
        # sub 2
        distance  = double_taps[i, 1] - double_taps[i,0]
        ax[1].plot(candles.loc[double_taps[i, 0] - distance: \
                   double_taps[i,1]+distance, 'midclose'].values)
        ax[1].plot(distance, candles.loc[double_taps[i,0], 'midclose'], 'o')
        ax[1].plot(2* distance , candles.loc[double_taps[i,1], 'midclose'],'o')                
        # Sub        ax[2].plot(candles.loc[double_taps[i, 1] - (local_max_interval / 2): \
                      double_taps[i, 1] + (local_max_interval / 2), 
                      'midclose'].values)
        ax[2].plot((local_max_interval / 2), candles.loc[double_taps[i, 1], 
                                                'midclose'], 'o' )
        # Plot
        plt.tight_layout()
        plt.show()
        print(double_taps[i])
        



