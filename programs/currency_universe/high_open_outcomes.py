from scipy import stats
from libraries.stats import get_distribution_boundary


'''
STD:
    eur = 0.0065
    usd = 0.0124   but this doesn't matter 

Rolling STD:
    
    EUR 
    ----------------
    60      0.000047
    120     0.000067
    240     0.000096
    480     0.000138
    960     0.000202
    1500    0.000258            so that rolling is mostly the same
    1750    0.000280
    2000    0.000301

    USD
    ----------------
    60      0.000044
    120     0.000063
    240     0.000089
    480     0.000128
    960     0.000185
    1500    0.000234
    1750    0.000254
    2000    0.000272
    
'''
    
   


'''



 
_open = cur.shift(1)  
_open = _open.fillna(method='backfill')

targets = eur_rolling_std.iloc[:, 1].fillna(method='backfill').values
eur_outcomes = high_low_open_outcomes(_open.eur.values, 
                                      high.eur.values, 
                                      low.eur.values, 
                                      1000, 
                                      targets)
e_l = eur_outcomes['high'] < eur_outcomes['low']
e_s = eur_outcomes['low'] < eur_outcomes['high']
e_l = pd.DataFrame(e_l, index = eur.index, columns = ['long'])
e_s = pd.DataFrame(e_s, index = eur.index, columns = ['short'])
e_minimums = np.minimum(e_l.values, e_s.values)
e_minimums = pd.DataFrame(e_minimums, index = eur.index, columns = ['minimums'])


targets = usd_rolling_std.iloc[:, 1].fillna(method='backfill').values
usd_outcomes = high_low_open_outcomes(_open.usd.values, 
                                      high.usd.values, 
                                      low.usd.values, 
                                      1000, 
                                      targets)
u_l = usd_outcomes['high'] < usd_outcomes['low']
u_s = usd_outcomes['low'] < usd_outcomes['high']
u_l = pd.DataFrame(u_l, index = usd.index, columns = ['long'])
u_s = pd.DataFrame(u_s, index = usd.index, columns = ['short'])
u_minimums = np.minimum(u_l.values, u_s.values)
u_minimums = pd.DataFrame(u_minimums, index = usd.index, columns = ['minimums'])


    
'''

if False:
    
    
    predictions = pd.DataFrame(predictions, columns=['predictions'], 
                               index=test_index)
    targets = eur_rolling_std.iloc[test_index, 1].fillna(method='backfill')
    
    bins = 10
    bin_edges = stats.mstats.mquantiles(targets.values, 
                                        np.linspace(0,1,bins +1))

    for i in range(1, bins):
        cond1 = targets > bin_edges[i]
        cond2 = targets < bin_edges[i + 1]
        index = cond1[cond1 & cond2].index.values
        wins = e_l.loc[index].mean()
        print('{}: \t {}\n'.format(i, wins))





# use rolling std (window ~ 120) as difference
difference = eur_rolling_std.iloc[:, 1]
index = (high.eur - eur.shift(1)) > difference
print(index.sum())
print(index.mean())
eur_index = eur.shift(1)[index].index.values
print(ratio_long.loc[index].mean())

difference = usd_rolling_std.iloc[:, 1]
index = (usd.shift(1) - low.usd) <  - difference
print(index.sum())
print(index.mean())
usd_index = usd.shift(1)[index].index.values
print(ratio_long.loc[index].mean())


combined = np.intersect1d(usd_index, eur_index)
print(combined.shape)
print(combined.shape[0] / eur.shape[0])




window = 100
location = index[window00]
eur.loc[location - window: location + window].plot()
high.loc[location - window: location + window, 'eur'].plot()
low.loc[location - window: location + window, 'eur'].plot()
_open.loc[location - window: location + window, 'eur'].plot()


plt.figure()
usd.loc[location - window: location + window].plot()
high.loc[location - window: location + window, 'usd'].plot()
low.loc[location - window: location + window, 'usd'].plot()
_open.loc[location - window: location + window, 'usd'].plot()



rat = high.eur / low.usd


plt.figure()
rat.loc[location - window: location + window].plot(label = 'calc')
ratio.loc[location - window: location + window].plot(label='given')
plt.legend()









difference = .00012 / 2 
index = (high.eur - eur.shift(1)) > difference
print(index.sum())
print(index.mean())
index = eur.shift(1)[index].index.values
print(ratio_long.loc[index].mean())

difference = .00015 / 2
index = (high.usd - usd.shift(1)) <  - difference
print(index.sum())
print(index.mean())
index = usd.shift(1)[index].index.values
print(ratio_long.loc[index].mean())


start = 10000
end = 10100

# Plot high , low , close on given and calculated instrument
plt.figure()
candles.loc[start:end, 'midhigh'].plot(label = 'h given')
(high.loc[start:end, 'eur'] / high.loc[start:end, 'usd']).shift(1).plot(label = 'h calculated')
candles.loc[start:end, 'midclose'].plot(label = 'm given')
(cur.loc[start:end, 'eur'] / cur.loc[start:end, 'usd']).shift(1).plot(label = 'm calculated')
candles.loc[start:end, 'midlow'].plot(label = 'l given')
(low.loc[start:end, 'eur'] / low.loc[start:end, 'usd']).shift(1).plot(label = 'l calculated')
plt.legend()
plt.show()


# Plot av diff of cal and given 
plt.figure()
high_diff = candles.loc[start:end, 'midhigh']\
          - (high.loc[start:end, 'eur'] / high.loc[start:end, 'usd']).shift(1)
close_diff = candles.loc[start:end, 'midclose']\
          - (cur.loc[start:end, 'eur'] / cur.loc[start:end, 'usd']).shift(1)    
low_diff = candles.loc[start:end, 'midlow']\
          - (low.loc[start:end, 'eur'] / low.loc[start:end, 'usd']    ).shift(1)  
           
high_diff.plot(label = 'high')
close_diff.plot(label = 'close')
low_diff.plot(label = 'low')

((high_diff + close_diff + low_diff) / 3).plot(label='avergage', color='black', linewidth=3)
plt.plot(np.arange(start, end + 1), np.zeros(high_diff.shape[0]), color='grey')
plt.title('Difference of given and calculated')
plt.show()




'''
Hypothesis:
    When the avergage difference between the calculated high, low and close 
    values is ___ different (signed) than the given high, low and close values,
    there will be some preinformation as to the movement of the instrument 
    price.
    
    Prediction:
        When values are at a certain low, the price will move up in very short 
        term.  Vice versa as well.
        
    
    first Analysis:

        on the next values minus values at those index - should support a sign 
        following prediction.
        
'''


high_diff  = candles.midhigh  - (high.eur / high.usd)#.shift(-1)
close_diff = candles.midclose - (cur.eur / cur.usd)  #.shift(-1)         
low_diff   = candles.midlow   - (low.eur / low.usd)  #.shift(-1)    

high_diff = StandardScaler().fit_transform(high_diff.fillna(0).values.reshape(-1,1))
close_diff = StandardScaler().fit_transform(close_diff.fillna(0).values.reshape(-1,1))
low_diff = StandardScaler().fit_transform(low_diff.fillna(0).values.reshape(-1,1))


mean_calculated_difference = (high_diff + close_diff + low_diff) / 3

boundaries = get_distribution_boundary(mean_calculated_difference.ravel(), .01)

bound_outs = ratios.loc[boundaries['upper_index'] + 1, 'eur_usd'].values\
           - ratios.loc[boundaries['upper_index'], 'eur_usd'].values
bound_outs = StandardScaler().fit_transform(bound_outs[:-1].reshape(-1, 1))

sns.distplot(bound_outs, bins = 100)

print(bound_outs.mean())
print(pd.Series(bound_outs.ravel()).skew())
print(pd.Series(bound_outs.ravel()).kurtosis())




plt.figure()
high_diff.plot(label = 'high')
close_diff.plot(label = 'close')
low_diff.plot(label = 'low')
mean_calculated_difference.plot(label='mean', linewidth=3, color='black')
plt.legend()
plt.show()


plt.plot(high_diff)
plt.plot(close_diff)
plt.plot(low_diff)
plt.plot(mean_calculated_difference, color='black')






diff = (candles.midclose.shift(1) - candles.midclose).fillna(method='backfill')
boundaries = get_distribution_boundary(mean_calculated_difference.ravel(), .005)
sns.distplot(diff.loc[boundaries['lower_index'] + 1])










