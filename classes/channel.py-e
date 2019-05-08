import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os; os.chdir('/northbend')


class Channel():

    def __init__(self, values, std_ratio=2, plot=0):
        
        # ====== Fit Linear curve to data, flatten and supply measures ========
        def curve(x, a, b):
            return (a * x) + b
        self.x = np.arange(values.shape[0])
        self.std_ratio = std_ratio
        coef, errors = curve_fit(curve, self.x, values)
        line = (coef[0] * self.x) + coef[1]
        self.slope = coef[0]
        self.intercept = coef[1]
        self.fit = errors
        self.flattened = values - line
        self.mean = 0
        self.channel_deviation = self.flattened.std()
        self.position_distance = self.flattened[-1]
        self.position_distance_standard = self.flattened[-1] \
                                        / self.flattened.std()
        self.line = line
        
        # If given std ratios, plot flattened with chanels and print out info
        if type(plot) == list:
            plt.plot(self.flattened)
            plt.plot(np.zeros(self.x.shape[0]), color='orange') 
            plt.plot(self.x[-1], self.flattened[-1], '+', 
                     color='black', markersize=10)
            for line in plot:
                plt.plot(np.zeros(self.x.shape[0]) + (self.deviation * line), 
                         color='orange')
                plt.plot(np.zeros(self.x.shape[0]) - (self.deviation * line), 
                         color='orange')
            plt.show()
            print(self.position_distance)
            print(self.position_distance_standard)    
          
            
    # ======= Channel Line Methods ============================================
    
    def c3(self):
        return self.line
    

    def c5(self):
        return self.line + self.channel_deviation * 2 #* self.std_ratio
    
    
    def c1(self):
        return self.line - self.channel_deviation * 2 #self.std_ratio
    
        
