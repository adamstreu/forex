import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler as minmaxscaler
from sklearn.preprocessing import StandardScaler as standardscaler
import os; os.chdir('/northbend')
from libraries.outcomes import outcomes


class Channel():
    
 
    def __init__(self, values):
        
        
        def flatten_and_scale(self):
            '''
            Purpose:    Transforms closing values to 'horizontal' position 
                            by subtracting all values from a linear regression line
                        First Shifts closing values to start at Zero.
            Input:      Array of (closing) values
            output:     Just the flattened line
            '''
             # Shift closing values downward.  
            y = self.closings
            x = np.arange(y.shape[0])
            # Compute linear regresssion
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            regression_line = intercept + slope * x
            # Flatten closings horizontal using regression line
            flattened = y - regression_line
            scaler = minmaxscaler().fit(flattened.reshape(-1, 1))
            scaled = scaler.transform(flattened.reshape(-1, 1)).ravel()
            d = {'regression_line' : regression_line,
                 'flattened'       : flattened,
                 'scaler'          : scaler,
                 'scaled'          : scaled,
                 'intercept'       : intercept,
                 'slope'           : slope,
                 'r_value'         : r_value,
                 'p_value'         : p_value,
                 'std_err'         : std_err}
            return d
               
            
        def create_channels(self):
            '''
            Does the main work of channel creation.
            '''
            y = self.scaled.ravel()
            x = np.arange(y.shape[0])
            # Find top and bottom channels that are most perpendicular to each other.
            # min_distance = 1000 # set min mean variable for iterations
            min_distance_perc = 0
            slope_record = 1000
            slope_keep = 0
            slope_side = 'unknown'
            for perc in np.arange(2, 25, 2):  
                
                # Get indices and solution of largest and smallest % of numbers in closings
                qty = int(perc * y.shape[0] / 100)
                top = np.argpartition(y, -qty)[-qty:]
                bottom = np.argpartition(y, qty)[:qty]
                # Calculate linear Regression on Top and Bottom Line
                slope, intercept, r_value, p_value, std_err = linregress(top, y[top])
                # t_slope = slope
                top_line = intercept + slope * x
                slope, intercept, r_value, p_value, std_err = linregress(top, y[bottom])
                bottom_line = intercept + slope * x
                # Bring lines together
                mid = int(y.shape[0] / 2)
                top_lowered = top_line - (top_line[mid] - bottom_line[mid])
                bottom_raised = (top_line[mid] - bottom_line[mid]) + bottom_line

                '''
                # New Way of forming channels.
                # We had some funky ones under the last system (narrow).
                # Teo things to notice - 
                    # Run a buncnh of plot sof diff i in ( in top group).
                        # do they in general look like beetter plotted channels
                    # How do the test results change/.
                if abs(top_line - bottom_raised).sum() < slope_record:
                    slope_keep = perc
                    slope_record = abs(top_line - bottom_raised).sum()
                '''
                
                '''
                # This is the old way of forming channels
                    # channel breakout so far used this.
                    # Values will change.....
                # Calculate the middle line and take its slope
                mid_top = (top_line + bottom_raised) / 2
                midslope = (mid_top[-1] - mid_top[0]) / mid_top.shape[0]
                if abs(midslope) < slope_record:
                    slope_keep = perc
                    slope_record = abs(midslope)
                '''

                # Keep a whatever line hit's lowest slope and when.
                # Then put line there and put other cutting across middle of
                # where the other was.
                top_slope = (top_line[-1] - top_line[0]) / top_line.shape[0]
                bottom_slope = (bottom_line[-1] - bottom_line[0]) / bottom_line.shape[0]
                if abs(bottom_slope) < slope_record:
                    slope_record = abs(bottom_slope)
                    slope_keep = perc
                    slope_side = 'bottom'
                if abs(top_slope) < slope_record:
                    slope_record = abs(top_slope)
                    slope_keep = perc
                    slope_side = 'top'
                
                
                
                '''
                # Plot channel Creation.  
                print(perc)
                print('slope record, slope keep, slope direction: {}\t{}\t{}'.format(slope_record, slope_keep, slope_side))
                plt.plot(self.scaled)
                plt.plot(top_line, color = 'blue')
                plt.plot(top_lowered, color = 'blue')
                plt.plot(bottom_line, color = 'orange')
                plt.plot(bottom_raised, color = 'orange')
                plt.show()
                '''
                
                
                
            min_distance_perc = slope_keep
            # Get indices and solution of largest and smallest % of numbers in closings
            qty = int(min_distance_perc * y.shape[0] / 100)
            top = np.argpartition(y, -qty)[-qty:]
            bottom = np.argpartition(y, qty)[:qty]
            # Calculate linear Regression on Top and Bottom Line
            slope, intercept, r_value, p_value, std_err = linregress(top, y[top])
            top_line = intercept + slope * x
            slope, intercept, r_value, p_value, std_err = linregress(top, y[bottom])
            bottom_line = intercept + slope * x
            # Bring lines together
            mid = int(y.shape[0] / 2)
            top_lowered = top_line - (top_line[mid] - bottom_line[mid])
            bottom_raised = (top_line[mid] - bottom_line[mid]) + bottom_line
            
            
            
            '''
            # Plot accepted channel lines
            print('Kept lines: {}'.format(min_distance_perc))
            plt.plot(self.scaled)
            plt.plot(top_line, color = 'blue')
            plt.plot(top_lowered, color = 'blue')
            plt.plot(bottom_line, color = 'orange')
            plt.plot(bottom_raised, color = 'orange')
            plt.show()
            '''
            


            if slope_side == 'top':
                c7 = top_line
                c1 = top_lowered
            else:
                c1 = bottom_line
                c7 = bottom_raised
                       
            '''
            print('New Lines: {}'.format(min_distance_perc))     
            plt.figure()
            plt.plot(self.scaled)
            # Plot new accepted channel lines
            plt.plot(c1, color = 'orange')
            plt.plot(c7, color = 'orange')
            print(c7[-1] - c1[-1])
            plt.show()
            '''

            
            
            '''
            # Calculate average top and bottom lines to form c2 and c6 lines
            c6       = (top_line + bottom_raised) / 2
            c2       = (bottom_line + top_lowered) / 2
            # Use lines to compute remaining
            c4       = (c6 + c2) / 2
            distance = (c4 - c2) / 2
            c1 = c2 - distance
            c3 = c4 - distance
            c5 = c6 - distance
            c7 = c6 + distance    
            # Find closing position in terms of channel (c6 - c2)
            closing_position = (y[-1] - c1[-1]) / (c7[-1] - c1[-1]) 
            # Create Dictionary and Return
            
            d = {'c1': c1,
                 'c2': c2,
                 'c3': c3,
                 'c4': c4,
                 'c5': c5,
                 'c6': c6,
                 'c7': c7,
                 'degree': min_distance_perc,
                 'slope': (c2[-1] - c2[0]) / c2.shape[0],
                 'closing_position':closing_position}
            '''
            closing_position = (y[-1] - c1[-1]) / (c7[-1] - c1[-1]) 
            d = {'c1': c1,
                 'c7': c7,
                 'degree': min_distance_perc,
                 'slope': (c1[-1] - c1[0]) / c1.shape[0],
                 'closing_position': closing_position}
            return d
            
            
#        self.candles            = candles
#        self.closings           = candles.loc[location - channel_length: location, 'midclose'].values
#        self.location           = location
        self.closings           = values
        
        fs                      = flatten_and_scale(self)
        self.regression_line    = fs['regression_line']
        self.flattened          = fs['flattened']
        self.scaler             = fs['scaler']
        self.scaled             = fs['scaled']
        self.closings_slope     = fs['slope']
        self.closings_intercept = fs['intercept']
        self.linear_p_value     = fs['p_value']
        
        ch                      = create_channels(self)
        self.channel_slope      = ch['slope']
        self.channel_degree     = ch['degree']
        self.closing_position   = ch['closing_position']
        self.c1                 = ch['c1']
        '''
        self.c2                 = ch['c2']
        self.c3                 = ch['c3']
        self.c4                 = ch['c4']
        self.c5                 = ch['c5']
        self.c6                 = ch['c6']
        '''
        self.c7                 = ch['c7']
    
        # Unscale channels
        self.closings_c1 = self.scaler.inverse_transform(self.c1.reshape(-1, 1)).ravel()
        '''
        self.closings_c2 = self.scaler.inverse_transform(self.c2.reshape(-1, 1)).ravel()
        self.closings_c3 = self.scaler.inverse_transform(self.c3.reshape(-1, 1)).ravel()
        self.closings_c4 = self.scaler.inverse_transform(self.c4.reshape(-1, 1)).ravel()
        self.closings_c5 = self.scaler.inverse_transform(self.c5.reshape(-1, 1)).ravel()
        self.closings_c6 = self.scaler.inverse_transform(self.c6.reshape(-1, 1)).ravel()
        '''
        self.closings_c7 = self.scaler.inverse_transform(self.c7.reshape(-1, 1)).ravel()
        # Unflatten Channels
        self.closings_c1 += self.regression_line
        '''
        self.closings_c2 += self.regression_line
        self.closings_c3 += self.regression_line
        self.closings_c4 += self.regression_line
        self.closings_c5 += self.regression_line
        self.closings_c6 += self.regression_line
        '''
        self.closings_c7 += self.regression_line

        self.channel_range = self.closings_c7[-1] - self.closings_c1[-1]
        self.within_range  = ((self.closings_c1 < self.closings) & (self.closings < self.closings_c7)).mean()      
        self.largest_spike = (self.closings[1:] - self.closings[:-1]).max()
        self.largest_spike_5 = np.absolute(self.closings[1:] - self.closings[:-1]).max()

    
    def __str__(self):
        return 'channel class'
    
    def outcomes(self, outcomes_window):
        # Use distance of average channel to look for outcome bars
        average_channel_distance = self.channel_range / 6
        distance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * average_channel_distance
        up_down = simple_outcomes(self.candles, self.location, outcomes_window, distance, False)
        return up_down['down'] + up_down['up']
    
    
    def get_supports(self, closings, bins=10, plot = False):
#       closings = self.candles.loc[self.location - support_window + 1: self.location,  'midclose']
        hist = np.histogram(closings, bins = bins)
        x = (hist[1] + ((hist[1][1] - hist[1][0]) / 2))[:-1]
        peaks_index = []
        # Get the middle peaks
        left = (hist[0][1:] > hist[0][:-1])  # Don't use first
        right = hist[0][:-1] > hist[0][1:]
        peaks_index = left[:-1] & right[1:]
        # get the left and right peaks - careful with this
        if hist[0][0] > hist[0][1]:
            peaks_index = np.insert( peaks_index, 0, True)
        else:
            peaks_index = np.insert( peaks_index, 0, False)
        if hist[0][-1] > hist[0][-2]:
            peaks_index = np.insert( peaks_index, peaks_index.shape[0], True)
        else:
            peaks_index = np.insert( peaks_index, peaks_index.shape[0], False)    
        # Peaks 
        supports    = x[peaks_index]
        strengths = hist[0][peaks_index] / hist[0].sum()
        # Verification Plotting
        if plot:
            plt.figure(figsize = (14, 3))
            plt.plot(x, hist[0])
            for each in peaks:
                plt.plot([each, each], [0, hist[0].max()], color='red')   
        # Return
        return {'supports': supports, 'strengths': strengths}
        
    
    
    def plot_channels(self):
        plt.figure(figsize=(14, 4))
        plt.plot(self.scaled)
        plt.plot(self.c1, color='grey')
        plt.plot(self.c2, color='orange')
        plt.plot(self.c3, color='grey')
        plt.plot(self.c4, color='orange')
        plt.plot(self.c5, color='grey')
        plt.plot(self.c6, color='orange')
        plt.plot(self.c7, color='grey')
        
        
    def plot_closings(self, peaks_window=0, outcomes_window = 0):
        # Plot closing Values
        plt.figure('closing_values', figsize=(14,8))
        # Plot closings
        plt.plot(np.arange(self.closings.shape[0]), self.closings)
        # Plot closings channels
        plt.plot(self.closings_c1, color = 'grey')
        '''
        plt.plot(self.closings_c2, color = 'orange')
        plt.plot(self.closings_c3, color = 'grey')
        plt.plot(self.closings_c4, color = 'orange')
        plt.plot(self.closings_c5, color = 'grey')
        plt.plot(self.closings_c6, color = 'orange')
        '''
        plt.plot(self.closings_c7, color = 'grey')
        # Plot Peaks horizontally on closing channel
        if peaks_window != 0:
            peaks_window = peaks_window
            peaks = self.get_supports(peaks_window)
            for peak in peaks:
                plt.plot(np.ones(self.closings.shape[0]) * peak, color='indianred')
        # Plot outcomes
        if outcomes_window != 0:
            # Calculate values and Call outcomes
            average_channel_distance = self.channel_range / 6
            distance = np.arange(1, 11) * average_channel_distance
            outcomes = simple_outcomes(self.candles, self.location, 
                                                outcomes_window, distance)
            outcomes_closings = self.candles.loc[self.location: self.location \
                                                 + outcomes_window, 'midclose'].values
            x2 = np.arange(outcomes_closings.shape[0]) + self.closings.shape[0]
            # Plot outcomes candles
            plt.plot(x2, outcomes_closings)
            # plot target distances
            midclose = self.candles.loc[self.location, 'midclose']
            for target in distance:   
                plt.plot(x2, np.ones(x2.shape[0]) * midclose + target, color='grey')
                plt.plot(x2, np.ones(x2.shape[0]) * midclose - target, color='grey')     
            # Plot intersections of midclse and target values
            [plt.plot(x2[u], outcomes_closings[u], 'o', color='green') for u in outcomes['up']]
            [plt.plot(x2[d], outcomes_closings[d],  'o', color='red')  for d in outcomes['down']]      
  
            '''
            
            
            
            
            
            shift = self.closings_c4[-1] - self.closings[-1]
            avg_channel_width = self.channel_range / 6
            x2 = np.arange(outcomes_window) + len(self.closings) - 1
            # Set the outcomes channels
            o0 = np.ones(x2.shape[0]) * (self.closings_c1[-1] - shift) - avg_channel_width
            o1 = np.ones(x2.shape[0]) * (self.closings_c1[-1] - shift)
            o2 = np.ones(x2.shape[0]) * (self.closings_c2[-1] - shift)
            o3 = np.ones(x2.shape[0]) * (self.closings_c3[-1] - shift)
            o4 = np.ones(x2.shape[0]) * (self.closings_c4[-1] - shift)
            o5 = np.ones(x2.shape[0]) * (self.closings_c5[-1] - shift)
            o6 = np.ones(x2.shape[0]) * (self.closings_c6[-1] - shift)
            o7 = np.ones(x2.shape[0]) * (self.closings_c7[-1] - shift)
            o8 = np.ones(x2.shape[0]) * (self.closings_c7[-1] - shift) + avg_channel_width
            # Plot outcomes interval
            plt.plot(x2, self.candles.loc[self.location: self.location + outcomes_window, 'midclose'])
            # plot channels shifted to last values of closings
            plt.plot(x2, o0, color='red')
            plt.plot(x2, o1, color='orange')
            plt.plot(x2, o2, color='grey')
            plt.plot(x2, o3, color='orange')
            plt.plot(x2, o4, color='grey')
            plt.plot(x2, o5, color='orange')
            plt.plot(x2, o6, color='grey')
            plt.plot(x2, o7, color='grey')
            plt.plot(x2, o8, color='red')           
            # Plot the Crossings of the outcome interval and shifted and straightened channels (+ extra on both sides)
            plt.plot(len(self.closings) + outcome_positions[3], (self.closings_c1[-1] - shift) - avg_channel_width, 'o', color = 'blue')
            plt.plot(len(self.closings) + outcome_positions[2], (self.closings_c1[-1] - shift), 'o', color = 'blue')
            plt.plot(len(self.closings) + outcome_positions[1], (self.closings_c2[-1] - shift), 'o', color = 'blue')
            plt.plot(len(self.closings) + outcome_positions[0], (self.closings_c3[-1] - shift), 'o', color = 'blue')
            plt.plot(len(self.closings) + outcome_positions[4], (self.closings_c5[-1] - shift), 'o', color = 'red')
            plt.plot(len(self.closings) + outcome_positions[5], (self.closings_c6[-1] - shift), 'o', color = 'red')
            plt.plot(len(self.closings) + outcome_positions[6], (self.closings_c7[-1] - shift), 'o', color = 'red')
            plt.plot(len(self.closings) + outcome_positions[7], (self.closings_c7[-1] - shift) + avg_channel_width, 'o', color = 'red')
            # Print closing channel positions
            plt.plot(np.ones(self.closings.shape[0]) * self.closings_c7[-1], color='lightgrey')
            plt.plot(np.ones(self.closings.shape[0]) * self.closings_c1[-1], color='lightgrey')
            
            '''
            '''
            
            
                    # plot the high and low search interval
            x0 = np.arange(plot)
            plt.plot(x0, candles.loc[i - plot + 1: i, 'midclose'])
            x = np.arange(search_high.shape[0]) + x0[-1]
            plt.plot(x, search_high, color='blue')
            plt.plot(x, search_low,  color='blue')
            # Plot the target lines
            for target in targets:
                plt.plot(x, np.ones(x.shape[0]) * midclose + target, color='grey')
                plt.plot(x, np.ones(x.shape[0]) * midclose - target, color='grey')    
            # Plot up and down crossings
            [plt.plot(x[u], search_high[u], 'o', color='green') for u in up]
            [plt.plot(x[d], search_low[d],  'o', color='red')  for d in down]       

            
            
            
            '''
            
            
            
            
            
            
            
            
            
            plt.show()
            print()
            print('closing c7 value:  {}'.format(self.closings_c7[-1]))
            print('closing c1 value:  {}'.format(self.closings_c1[-1]))
            print('bars:              {}'.format(outcomes))
            print('range:             {}'.format(self.channel_range))
            print('channel degree     {}'.format(self.channel_degree))
            print('channel slope:     {}'.format(self.channel_slope))
            print('closing slope:     {}'.format(self.closings_slope))
            print('closing position:  {}'.format(self.closing_position))
            print('Within range:      {}'.format(self.within_range))
            print('Largest spike:     {}'.format(self.largest_spike))
            print('Largest spike 5:   {}'.format(self.largest_spike_5))
            
            

"""

            


       
    # Provide some statistics.  One - percent of function between channels
    d01 =  (y < c1).mean()
    d12 = ((y >= c1) & (y < c2)).mean()
    d23 = ((y >= c2) & (y < c3)).mean()
    d34 = ((y >= c3) & (y < c4)).mean()
    d45 = ((y >= c4) & (y < c5)).mean()
    d56 = ((y >= c5) & (y < c6)).mean()
    d67 = ((y >= c6) & (y < c7)).mean()
    d78 =  (y >= c7).mean()


    def average_channel_width(self):
        return 
        
    def __str__(self):
        return (str(self.closings))


    def flat_parameters(self):
        pass
    
    
    def channel_line(self, channel_number):
        pass
    
    
    def plot_channels(self, search_outcomes, search_length):        
        '''
        # Plot outcome channels
        outcome_channels = [o1, o2, o3, o4, o5, o6, o7]
        plot_colors = ['grey', 'orange', 'grey', 'orange', 'grey', 'orange', 'grey']
        for i in range(len(outcome_channels)):
            plt.plot(x2, outcome_channels[i], color=plot_colors[i])
        '''
        pass
    
    
    
    def plot_histogram(self):
        pass
    
    
    def parameters(self):
        return
    
    
        # Return Dictionary
        d = {'closing': y,
             'scaled': scaled.ravel(),
             'linregress': line,
             'slope': slope,
             'r_value': r_value,
             'p_value': p_value,
             'std_err': std_err}
    
    
    def closing_channels(self):

    

        
    
        

"""
"""
        
  
    
    def plot_channels(df, start, window_length, outcome_widths = 0,
                  peak_interval = 1000):
        # Define closing values
        df_window = df.loc[start - window_length: start].copy()
        closing_values = df_window.midclose.values
        # Define outcome values
        outcome_values = df.loc[start: start + (window_length * 1), 'midclose'].values
        # Flatten closing_values    
        closings_flat = horizontal_transform(closing_values)
        # Rescale closing Values
        mms = minmaxscaler()
        mms.fit(closings_flat['closing'].reshape(-1, 1))
        scaled = mms.transform(closings_flat['closing'].reshape(-1, 1)).ravel()
        # Create Channels
        channels = create_channels(scaled)
        # get closing peaks
        closing_peaks = find_peaks_cwt(closing_values, np.arange(1, peak_interval))
        scaled_peaks_high = find_peaks_cwt(scaled, np.arange(1, peak_interval))
        scaled_peaks_low = find_peaks_cwt(-scaled, np.arange(1, peak_interval))
        # get histogram and it's peaks
        hist = np.histogram(scaled, bins=10)
        hist_peaks = find_peaks_cwt(hist[0], np.arange(1, 10))
        # Try my own peaks
        if hist[0][0] > hist[0][1]:
            keep = [True]
        else:
            keep = [False]
        for i in range(1, hist[0].shape[0]-1):
            if hist[0][i] > hist[0][i+1] and hist[0][i] > hist[0][i-1]:
                keep.append(True)
            else:
                keep.append(False)
        if hist[0][-1] > hist[0][-2]:
            keep.append(True)
        else:
            keep.append(False)   
        keep = np.array(keep)
        
        # Unscale and unflatten channels
        c1 = mms.inverse_transform(channels['c1'].reshape(-1, 1)).ravel()
        c2 = mms.inverse_transform(channels['c2'].reshape(-1, 1)).ravel()
        c3 = mms.inverse_transform(channels['c3'].reshape(-1, 1)).ravel()
        c4 = mms.inverse_transform(channels['c4'].reshape(-1, 1)).ravel()
        c5 = mms.inverse_transform(channels['c5'].reshape(-1, 1)).ravel()
        c6 = mms.inverse_transform(channels['c6'].reshape(-1, 1)).ravel()
        c7 = mms.inverse_transform(channels['c7'].reshape(-1, 1)).ravel()
    
        # Plot regular closing values with linregression line
        plt.figure('closing_values', figsize=(14,4))
        plt.plot(np.arange(closing_values.shape[0]), closing_values)
        plt.plot((c1 + closings_flat['linregress'] ), color = 'grey')
        plt.plot((c2 + closings_flat['linregress'] ), color = 'orange')
        plt.plot((c3 + closings_flat['linregress'] ), color = 'grey')
        plt.plot((c4 + closings_flat['linregress'] ), color = 'orange')
        plt.plot((c5 + closings_flat['linregress'] ) ,color = 'grey')
        plt.plot((c6 + closings_flat['linregress'] ), color = 'orange')
        plt.plot((c7 + closings_flat['linregress'] ), color = 'grey')
        plt.plot(np.arange(outcome_values.shape[0]) + np.arange(closing_values.shape[0])[-1], outcome_values)
        plt.plot(np.arange(closing_values.shape[0])[closing_peaks], 
                 closing_values[closing_peaks], 'o', color='orange')
        
        # Plot outcome Width.  Might move this to seperate graph
        if type(outcome_widths) != 'int':
            for ow in outcome_widths:
                plt.plot(np.arange(outcome_values.shape[0]) + np.arange(closing_values.shape[0])[-1], np.ones(outcome_values.shape[0]) * (outcome_values[0] + ow))
                plt.plot(np.arange(outcome_values.shape[0]) + np.arange(closing_values.shape[0])[-1], np.ones(outcome_values.shape[0]) * (outcome_values[0] - ow))
    
        
        
        
        
        
        # Plot channel with flattened and scaled closing values
        colors = ['grey', 'grey', 'black', 'grey', 'black', 'grey', 'black', 'grey']
        plt.plot(closings_flat['linregress'])
        plt.figure('channels', figsize=(14,4))
        plt.plot(closings_flat['scaled'])
        for i in range(1, 8):
            plt.plot(channels['c'+str(i)], color=colors[i])
        plt.plot(np.arange(scaled.shape[0])[scaled_peaks_high], 
                 scaled[scaled_peaks_high], 'o', color='orange')
    #    plt.plot(np.arange(scaled.shape[0])[scaled_peaks_low], 
    #             scaled[scaled_peaks_low], 'o', color='yellow')
        
        # Print the histogram
        plt.figure('histogram', figsize=(14,4))
        x = (hist[1] + ((hist[1][1] - hist[1][0]) / 2))[:-1]
        plt.plot(x, hist[0])
    #    plt.plot(x[hist_peaks], hist[0][hist_peaks], 'o', markersize=8)
        max_hist = hist[0].max()
        
        # Plot the closing position
        plt.plot([channels['closing_position'], channels['closing_position']], 
                 [0, max_hist / 2], color = 'black')
        for i in range(1, 8):
            plt.plot([channels['c'+str(i)].mean(), channels['c'+str(i)].mean()], 
                     [0, max_hist],
                     color='orange')
    #    plt.plot(scaled[scaled_peaks_high], np.ones(scaled_peaks_high.shape[0]) * max_hist * .5,'o', color='grey')
    #    plt.plot(scaled[scaled_peaks_low], np.ones(scaled_peaks_low.shape[0]) * max_hist * .25,'o', color='grey')
        plt.plot(x[keep], hist[0][keep], 'o', color = 'red')
    
        # Plot outcome widths
    #    plt.figure()
    #    if type(outcome_widths) != 'int':
    #        plt.plot(outcome_values)
    #        for ow in outcome_widths:
    #            plt.plot(np.ones(outcome_values.shape[0]) * (outcome_values[0] + ow))
    #            plt.plot(np.ones(outcome_values.shape[0]) * (outcome_values[0] - ow))
    #        print(ow)
    #    print(outcome_values.shape[0])
        
        
        
    #    # Plot the volume
    #    plt.figure('volume_spread', figsize=(14,4))
    #    plt.plot(standardscaler().fit_transform(df[['volume', 'spread']])[:, 0])
    #    plt.plot(standardscaler().fit_transform(df[['volume', 'spread']])[:, 1])
    #    
    #    # Plot polar distribution of flattened and scaled
    #    plt.figure('polar', figsize=(14, 4))
    #    ax = plt.subplot(111, projection='polar')
    #    r = np.linspace(0, 1, window_length)
    #    theta = 2 * np.pi * r
    #    ax.plot(theta, scaled[:-1])
    #
    #    plt.figure('hist_polar_channel', figsize=(14, 4))
    #    ax = plt.subplot(111, projection='polar')
    #    r = np.linspace(0, 1, hist[1][1:].shape[0])
    #    theta = 2 * np.pi * r
    #    ax.plot(theta, hist[0])   
    
    
    
    
        # Plot and print and Retrun
        plt.show()
        print('closing position: {}'.format(channels['closing_position']))
        print('Peaks:            {}'.format(x[keep]))
        print('Amplitudes:       {}'.format(hist[0][keep]))
        print('Channels Range:   {}'.format(channels['range']))

    

"""




class Channel_old():
    
 
    def __init__(self, candles, location, channel_length, search_interval=500):
        
        
        def flatten_and_scale(self):
            '''
            Purpose:    Transforms closing values to 'horizontal' position 
                            by subtracting all values from a linear regression line
                        First Shifts closing values to start at Zero.
            Input:      Array of (closing) values
            output:     Just the flattened line
            '''
             # Shift closing values downward.  
            y = self.closings
            x = np.arange(y.shape[0])
            # Compute linear regresssion
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            regression_line = intercept + slope * x
            # Flatten closings horizontal using regression line
            flattened = y - regression_line
            scaler = minmaxscaler().fit(flattened.reshape(-1, 1))
            scaled = scaler.transform(flattened.reshape(-1, 1))
            d = {'regression_line' : regression_line,
                 'flattened'       : flattened,
                 'scaler'          : scaler,
                 'scaled'          : scaled,
                 'intercept'       : intercept,
                 'slope'           : slope,
                 'r_value'         : r_value,
                 'p_value'         : p_value,
                 'std_err'         : std_err}
            return d
               
            
        def create_channels(self):
            '''
            Does the main work of channel creation.
            '''
            y = self.scaled.ravel()
            x = np.arange(y.shape[0])
            # Find top and bottom channels that are most perpendicular to each other.
            # min_distance = 1000 # set min mean variable for iterations
            min_distance_perc = 0
            slope_record = 1000
            slope_keep = 0
            slope_side = 'unknown'
            for perc in np.arange(2, 25, 2):  
                
                # Get indices and solution of largest and smallest % of numbers in closings
                qty = int(perc * y.shape[0] / 100)
                top = np.argpartition(y, -qty)[-qty:]
                bottom = np.argpartition(y, qty)[:qty]
                # Calculate linear Regression on Top and Bottom Line
                slope, intercept, r_value, p_value, std_err = linregress(top, y[top])
                # t_slope = slope
                top_line = intercept + slope * x
                slope, intercept, r_value, p_value, std_err = linregress(top, y[bottom])
                bottom_line = intercept + slope * x
                # Bring lines together
                mid = int(y.shape[0] / 2)
                top_lowered = top_line - (top_line[mid] - bottom_line[mid])
                bottom_raised = (top_line[mid] - bottom_line[mid]) + bottom_line

                '''
                # New Way of forming channels.
                # We had some funky ones under the last system (narrow).
                # Teo things to notice - 
                    # Run a buncnh of plot sof diff i in ( in top group).
                        # do they in general look like beetter plotted channels
                    # How do the test results change/.
                if abs(top_line - bottom_raised).sum() < slope_record:
                    slope_keep = perc
                    slope_record = abs(top_line - bottom_raised).sum()
                '''
                
                '''
                # This is the old way of forming channels
                    # channel breakout so far used this.
                    # Values will change.....
                # Calculate the middle line and take its slope
                mid_top = (top_line + bottom_raised) / 2
                midslope = (mid_top[-1] - mid_top[0]) / mid_top.shape[0]
                if abs(midslope) < slope_record:
                    slope_keep = perc
                    slope_record = abs(midslope)
                '''

                # Keep a whatever line hit's lowest slope and when.
                # Then put line there and put other cutting across middle of
                # where the other was.
                top_slope = (top_line[-1] - top_line[0]) / top_line.shape[0]
                bottom_slope = (bottom_line[-1] - bottom_line[0]) / bottom_line.shape[0]
                if abs(bottom_slope) < slope_record:
                    slope_record = abs(bottom_slope)
                    slope_keep = perc
                    slope_side = 'bottom'
                if abs(top_slope) < slope_record:
                    slope_record = abs(top_slope)
                    slope_keep = perc
                    slope_side = 'top'
                
                
                
                '''
                # Plot channel Creation.  
                print(perc)
                print('slope record, slope keep, slope direction: {}\t{}\t{}'.format(slope_record, slope_keep, slope_side))
                plt.plot(self.scaled)
                plt.plot(top_line, color = 'blue')
                plt.plot(top_lowered, color = 'blue')
                plt.plot(bottom_line, color = 'orange')
                plt.plot(bottom_raised, color = 'orange')
                plt.show()
                '''
                
                
                
            min_distance_perc = slope_keep
            # Get indices and solution of largest and smallest % of numbers in closings
            qty = int(min_distance_perc * y.shape[0] / 100)
            top = np.argpartition(y, -qty)[-qty:]
            bottom = np.argpartition(y, qty)[:qty]
            # Calculate linear Regression on Top and Bottom Line
            slope, intercept, r_value, p_value, std_err = linregress(top, y[top])
            top_line = intercept + slope * x
            slope, intercept, r_value, p_value, std_err = linregress(top, y[bottom])
            bottom_line = intercept + slope * x
            # Bring lines together
            mid = int(y.shape[0] / 2)
            top_lowered = top_line - (top_line[mid] - bottom_line[mid])
            bottom_raised = (top_line[mid] - bottom_line[mid]) + bottom_line
            
            
            
            '''
            # Plot accepted channel lines
            print('Kept lines: {}'.format(min_distance_perc))
            plt.plot(self.scaled)
            plt.plot(top_line, color = 'blue')
            plt.plot(top_lowered, color = 'blue')
            plt.plot(bottom_line, color = 'orange')
            plt.plot(bottom_raised, color = 'orange')
            plt.show()
            '''
            


            if slope_side == 'top':
                c7 = top_line
                c1 = top_lowered
            else:
                c1 = bottom_line
                c7 = bottom_raised
                       
            '''
            print('New Lines: {}'.format(min_distance_perc))     
            plt.figure()
            plt.plot(self.scaled)
            # Plot new accepted channel lines
            plt.plot(c1, color = 'orange')
            plt.plot(c7, color = 'orange')
            print(c7[-1] - c1[-1])
            plt.show()
            '''

            
            
            '''
            # Calculate average top and bottom lines to form c2 and c6 lines
            c6       = (top_line + bottom_raised) / 2
            c2       = (bottom_line + top_lowered) / 2
            # Use lines to compute remaining
            c4       = (c6 + c2) / 2
            distance = (c4 - c2) / 2
            c1 = c2 - distance
            c3 = c4 - distance
            c5 = c6 - distance
            c7 = c6 + distance    
            # Find closing position in terms of channel (c6 - c2)
            closing_position = (y[-1] - c1[-1]) / (c7[-1] - c1[-1]) 
            # Create Dictionary and Return
            
            d = {'c1': c1,
                 'c2': c2,
                 'c3': c3,
                 'c4': c4,
                 'c5': c5,
                 'c6': c6,
                 'c7': c7,
                 'degree': min_distance_perc,
                 'slope': (c2[-1] - c2[0]) / c2.shape[0],
                 'closing_position':closing_position}
            '''
            closing_position = (y[-1] - c1[-1]) / (c7[-1] - c1[-1]) 
            d = {'c1': c1,
                 'c7': c7,
                 'degree': min_distance_perc,
                 'slope': (c1[-1] - c1[0]) / c1.shape[0],
                 'closing_position': closing_position}
            return d
            
            
        self.candles            = candles
        self.closings           = candles.loc[location - channel_length: location, 'midclose'].values
        self.location           = location
        
        fs                      = flatten_and_scale(self)
        self.regression_line    = fs['regression_line']
        self.flattened          = fs['flattened']
        self.scaler             = fs['scaler']
        self.scaled             = fs['scaled']
        self.closings_slope     = fs['slope']
        self.closings_intercept = fs['intercept']
        
        ch                      = create_channels(self)
        self.channel_slope      = ch['slope']
        self.channel_degree     = ch['degree']
        self.closing_position   = ch['closing_position']
        self.c1                 = ch['c1']
        '''
        self.c2                 = ch['c2']
        self.c3                 = ch['c3']
        self.c4                 = ch['c4']
        self.c5                 = ch['c5']
        self.c6                 = ch['c6']
        '''
        self.c7                 = ch['c7']
    
        # Unscale channels
        self.closings_c1 = self.scaler.inverse_transform(self.c1.reshape(-1, 1)).ravel()
        '''
        self.closings_c2 = self.scaler.inverse_transform(self.c2.reshape(-1, 1)).ravel()
        self.closings_c3 = self.scaler.inverse_transform(self.c3.reshape(-1, 1)).ravel()
        self.closings_c4 = self.scaler.inverse_transform(self.c4.reshape(-1, 1)).ravel()
        self.closings_c5 = self.scaler.inverse_transform(self.c5.reshape(-1, 1)).ravel()
        self.closings_c6 = self.scaler.inverse_transform(self.c6.reshape(-1, 1)).ravel()
        '''
        self.closings_c7 = self.scaler.inverse_transform(self.c7.reshape(-1, 1)).ravel()
        # Unflatten Channels
        self.closings_c1 += self.regression_line
        '''
        self.closings_c2 += self.regression_line
        self.closings_c3 += self.regression_line
        self.closings_c4 += self.regression_line
        self.closings_c5 += self.regression_line
        self.closings_c6 += self.regression_line
        '''
        self.closings_c7 += self.regression_line

        self.channel_range = self.closings_c7[-1] - self.closings_c1[-1]
        self.within_range  = ((self.closings_c1 < self.closings) & (self.closings < self.closings_c7)).mean()      
        self.largest_spike = (self.closings[1:] - self.closings[:-1]).max()
        self.largest_spike_5 = np.absolute(self.closings[1:] - self.closings[:-1]).max()

    
    def __str__(self):
        return 'channel class'
    
    def outcomes(self, outcomes_window):
        # Use distance of average channel to look for outcome bars
        average_channel_distance = self.channel_range / 6
        distance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * average_channel_distance
        up_down = simple_outcomes(self.candles, self.location, outcomes_window, distance, False)
        return up_down['down'] + up_down['up']
    
    
    def get_supports(self, closings, values, bins=10, plot = False):
#        closings = self.candles.loc[self.location - support_window + 1: self.location,  'midclose']
        
        hist = np.histogram(closings, bins = bins)
        x = (hist[1] + ((hist[1][1] - hist[1][0]) / 2))[:-1]
        peaks_index = []
        # Get the middle peaks
        left = (hist[0][1:] > hist[0][:-1])  # Don't use first
        right = hist[0][:-1] > hist[0][1:]
        peaks_index = left[:-1] & right[1:]
        # get the left and right peaks - careful with this
        if hist[0][0] > hist[0][1]:
            peaks_index = np.insert( peaks_index, 0, True)
        else:
            peaks_index = np.insert( peaks_index, 0, False)
        if hist[0][-1] > hist[0][-2]:
            peaks_index = np.insert( peaks_index, peaks_index.shape[0], True)
        else:
            peaks_index = np.insert( peaks_index, peaks_index.shape[0], False)    
        # Peaks (Support)
        peaks = x[peaks_index]
        # Verification Plotting
        if plot:
            plt.figure(figsize = (14, 3))
            plt.plot(x, hist[0])
            for each in peaks:
                plt.plot([each, each], [0, hist[0].max()], color='red')   
        # Return
        return peaks
        
    
    
    def plot_channels(self):
        plt.figure(figsize=(14, 4))
        plt.plot(self.scaled)
        plt.plot(self.c1, color='grey')
        plt.plot(self.c2, color='orange')
        plt.plot(self.c3, color='grey')
        plt.plot(self.c4, color='orange')
        plt.plot(self.c5, color='grey')
        plt.plot(self.c6, color='orange')
        plt.plot(self.c7, color='grey')
        
        
    def plot_closings(self, peaks_window=0, outcomes_window = 0):
        # Plot closing Values
        plt.figure('closing_values', figsize=(14,8))
        # Plot closings
        plt.plot(np.arange(self.closings.shape[0]), self.closings)
        # Plot closings channels
        plt.plot(self.closings_c1, color = 'grey')
        '''
        plt.plot(self.closings_c2, color = 'orange')
        plt.plot(self.closings_c3, color = 'grey')
        plt.plot(self.closings_c4, color = 'orange')
        plt.plot(self.closings_c5, color = 'grey')
        plt.plot(self.closings_c6, color = 'orange')
        '''
        plt.plot(self.closings_c7, color = 'grey')
        # Plot Peaks horizontally on closing channel
        if peaks_window != 0:
            peaks_window = peaks_window
            peaks = self.get_supports(peaks_window)
            for peak in peaks:
                plt.plot(np.ones(self.closings.shape[0]) * peak, color='indianred')
        # Plot outcomes
        if outcomes_window != 0:
            # Calculate values and Call outcomes
            average_channel_distance = self.channel_range / 6
            distance = np.arange(1, 11) * average_channel_distance
            outcomes = simple_outcomes(self.candles, self.location, 
                                                outcomes_window, distance)
            outcomes_closings = self.candles.loc[self.location: self.location \
                                                 + outcomes_window, 'midclose'].values
            x2 = np.arange(outcomes_closings.shape[0]) + self.closings.shape[0]
            # Plot outcomes candles
            plt.plot(x2, outcomes_closings)
            # plot target distances
            midclose = self.candles.loc[self.location, 'midclose']
            for target in distance:   
                plt.plot(x2, np.ones(x2.shape[0]) * midclose + target, color='grey')
                plt.plot(x2, np.ones(x2.shape[0]) * midclose - target, color='grey')     
            # Plot intersections of midclse and target values
            [plt.plot(x2[u], outcomes_closings[u], 'o', color='green') for u in outcomes['up']]
            [plt.plot(x2[d], outcomes_closings[d],  'o', color='red')  for d in outcomes['down']]      
  
            '''
            
            
            
            
            
            shift = self.closings_c4[-1] - self.closings[-1]
            avg_channel_width = self.channel_range / 6
            x2 = np.arange(outcomes_window) + len(self.closings) - 1
            # Set the outcomes channels
            o0 = np.ones(x2.shape[0]) * (self.closings_c1[-1] - shift) - avg_channel_width
            o1 = np.ones(x2.shape[0]) * (self.closings_c1[-1] - shift)
            o2 = np.ones(x2.shape[0]) * (self.closings_c2[-1] - shift)
            o3 = np.ones(x2.shape[0]) * (self.closings_c3[-1] - shift)
            o4 = np.ones(x2.shape[0]) * (self.closings_c4[-1] - shift)
            o5 = np.ones(x2.shape[0]) * (self.closings_c5[-1] - shift)
            o6 = np.ones(x2.shape[0]) * (self.closings_c6[-1] - shift)
            o7 = np.ones(x2.shape[0]) * (self.closings_c7[-1] - shift)
            o8 = np.ones(x2.shape[0]) * (self.closings_c7[-1] - shift) + avg_channel_width
            # Plot outcomes interval
            plt.plot(x2, self.candles.loc[self.location: self.location + outcomes_window, 'midclose'])
            # plot channels shifted to last values of closings
            plt.plot(x2, o0, color='red')
            plt.plot(x2, o1, color='orange')
            plt.plot(x2, o2, color='grey')
            plt.plot(x2, o3, color='orange')
            plt.plot(x2, o4, color='grey')
            plt.plot(x2, o5, color='orange')
            plt.plot(x2, o6, color='grey')
            plt.plot(x2, o7, color='grey')
            plt.plot(x2, o8, color='red')           
            # Plot the Crossings of the outcome interval and shifted and straightened channels (+ extra on both sides)
            plt.plot(len(self.closings) + outcome_positions[3], (self.closings_c1[-1] - shift) - avg_channel_width, 'o', color = 'blue')
            plt.plot(len(self.closings) + outcome_positions[2], (self.closings_c1[-1] - shift), 'o', color = 'blue')
            plt.plot(len(self.closings) + outcome_positions[1], (self.closings_c2[-1] - shift), 'o', color = 'blue')
            plt.plot(len(self.closings) + outcome_positions[0], (self.closings_c3[-1] - shift), 'o', color = 'blue')
            plt.plot(len(self.closings) + outcome_positions[4], (self.closings_c5[-1] - shift), 'o', color = 'red')
            plt.plot(len(self.closings) + outcome_positions[5], (self.closings_c6[-1] - shift), 'o', color = 'red')
            plt.plot(len(self.closings) + outcome_positions[6], (self.closings_c7[-1] - shift), 'o', color = 'red')
            plt.plot(len(self.closings) + outcome_positions[7], (self.closings_c7[-1] - shift) + avg_channel_width, 'o', color = 'red')
            # Print closing channel positions
            plt.plot(np.ones(self.closings.shape[0]) * self.closings_c7[-1], color='lightgrey')
            plt.plot(np.ones(self.closings.shape[0]) * self.closings_c1[-1], color='lightgrey')
            
            '''
            '''
            
            
                    # plot the high and low search interval
            x0 = np.arange(plot)
            plt.plot(x0, candles.loc[i - plot + 1: i, 'midclose'])
            x = np.arange(search_high.shape[0]) + x0[-1]
            plt.plot(x, search_high, color='blue')
            plt.plot(x, search_low,  color='blue')
            # Plot the target lines
            for target in targets:
                plt.plot(x, np.ones(x.shape[0]) * midclose + target, color='grey')
                plt.plot(x, np.ones(x.shape[0]) * midclose - target, color='grey')    
            # Plot up and down crossings
            [plt.plot(x[u], search_high[u], 'o', color='green') for u in up]
            [plt.plot(x[d], search_low[d],  'o', color='red')  for d in down]       

            
            
            
            '''
            
            
            
            
            
            
            
            
            
            plt.show()
            print()
            print('closing c7 value:  {}'.format(self.closings_c7[-1]))
            print('closing c1 value:  {}'.format(self.closings_c1[-1]))
            print('bars:              {}'.format(outcomes))
            print('range:             {}'.format(self.channel_range))
            print('channel degree     {}'.format(self.channel_degree))
            print('channel slope:     {}'.format(self.channel_slope))
            print('closing slope:     {}'.format(self.closings_slope))
            print('closing position:  {}'.format(self.closing_position))
            print('Within range:      {}'.format(self.within_range))
            print('Largest spike:     {}'.format(self.largest_spike))
            print('Largest spike 5:   {}'.format(self.largest_spike_5))
            
            

"""

            


       
    # Provide some statistics.  One - percent of function between channels
    d01 =  (y < c1).mean()
    d12 = ((y >= c1) & (y < c2)).mean()
    d23 = ((y >= c2) & (y < c3)).mean()
    d34 = ((y >= c3) & (y < c4)).mean()
    d45 = ((y >= c4) & (y < c5)).mean()
    d56 = ((y >= c5) & (y < c6)).mean()
    d67 = ((y >= c6) & (y < c7)).mean()
    d78 =  (y >= c7).mean()


    def average_channel_width(self):
        return 
        
    def __str__(self):
        return (str(self.closings))


    def flat_parameters(self):
        pass
    
    
    def channel_line(self, channel_number):
        pass
    
    
    def plot_channels(self, search_outcomes, search_length):        
        '''
        # Plot outcome channels
        outcome_channels = [o1, o2, o3, o4, o5, o6, o7]
        plot_colors = ['grey', 'orange', 'grey', 'orange', 'grey', 'orange', 'grey']
        for i in range(len(outcome_channels)):
            plt.plot(x2, outcome_channels[i], color=plot_colors[i])
        '''
        pass
    
    
    
    def plot_histogram(self):
        pass
    
    
    def parameters(self):
        return
    
    
        # Return Dictionary
        d = {'closing': y,
             'scaled': scaled.ravel(),
             'linregress': line,
             'slope': slope,
             'r_value': r_value,
             'p_value': p_value,
             'std_err': std_err}
    
    
    def closing_channels(self):

    

        
    
        

"""
"""
        
  
    
    def plot_channels(df, start, window_length, outcome_widths = 0,
                  peak_interval = 1000):
        # Define closing values
        df_window = df.loc[start - window_length: start].copy()
        closing_values = df_window.midclose.values
        # Define outcome values
        outcome_values = df.loc[start: start + (window_length * 1), 'midclose'].values
        # Flatten closing_values    
        closings_flat = horizontal_transform(closing_values)
        # Rescale closing Values
        mms = minmaxscaler()
        mms.fit(closings_flat['closing'].reshape(-1, 1))
        scaled = mms.transform(closings_flat['closing'].reshape(-1, 1)).ravel()
        # Create Channels
        channels = create_channels(scaled)
        # get closing peaks
        closing_peaks = find_peaks_cwt(closing_values, np.arange(1, peak_interval))
        scaled_peaks_high = find_peaks_cwt(scaled, np.arange(1, peak_interval))
        scaled_peaks_low = find_peaks_cwt(-scaled, np.arange(1, peak_interval))
        # get histogram and it's peaks
        hist = np.histogram(scaled, bins=10)
        hist_peaks = find_peaks_cwt(hist[0], np.arange(1, 10))
        # Try my own peaks
        if hist[0][0] > hist[0][1]:
            keep = [True]
        else:
            keep = [False]
        for i in range(1, hist[0].shape[0]-1):
            if hist[0][i] > hist[0][i+1] and hist[0][i] > hist[0][i-1]:
                keep.append(True)
            else:
                keep.append(False)
        if hist[0][-1] > hist[0][-2]:
            keep.append(True)
        else:
            keep.append(False)   
        keep = np.array(keep)
        
        # Unscale and unflatten channels
        c1 = mms.inverse_transform(channels['c1'].reshape(-1, 1)).ravel()
        c2 = mms.inverse_transform(channels['c2'].reshape(-1, 1)).ravel()
        c3 = mms.inverse_transform(channels['c3'].reshape(-1, 1)).ravel()
        c4 = mms.inverse_transform(channels['c4'].reshape(-1, 1)).ravel()
        c5 = mms.inverse_transform(channels['c5'].reshape(-1, 1)).ravel()
        c6 = mms.inverse_transform(channels['c6'].reshape(-1, 1)).ravel()
        c7 = mms.inverse_transform(channels['c7'].reshape(-1, 1)).ravel()
    
        # Plot regular closing values with linregression line
        plt.figure('closing_values', figsize=(14,4))
        plt.plot(np.arange(closing_values.shape[0]), closing_values)
        plt.plot((c1 + closings_flat['linregress'] ), color = 'grey')
        plt.plot((c2 + closings_flat['linregress'] ), color = 'orange')
        plt.plot((c3 + closings_flat['linregress'] ), color = 'grey')
        plt.plot((c4 + closings_flat['linregress'] ), color = 'orange')
        plt.plot((c5 + closings_flat['linregress'] ) ,color = 'grey')
        plt.plot((c6 + closings_flat['linregress'] ), color = 'orange')
        plt.plot((c7 + closings_flat['linregress'] ), color = 'grey')
        plt.plot(np.arange(outcome_values.shape[0]) + np.arange(closing_values.shape[0])[-1], outcome_values)
        plt.plot(np.arange(closing_values.shape[0])[closing_peaks], 
                 closing_values[closing_peaks], 'o', color='orange')
        
        # Plot outcome Width.  Might move this to seperate graph
        if type(outcome_widths) != 'int':
            for ow in outcome_widths:
                plt.plot(np.arange(outcome_values.shape[0]) + np.arange(closing_values.shape[0])[-1], np.ones(outcome_values.shape[0]) * (outcome_values[0] + ow))
                plt.plot(np.arange(outcome_values.shape[0]) + np.arange(closing_values.shape[0])[-1], np.ones(outcome_values.shape[0]) * (outcome_values[0] - ow))
    
        
        
        
        
        
        # Plot channel with flattened and scaled closing values
        colors = ['grey', 'grey', 'black', 'grey', 'black', 'grey', 'black', 'grey']
        plt.plot(closings_flat['linregress'])
        plt.figure('channels', figsize=(14,4))
        plt.plot(closings_flat['scaled'])
        for i in range(1, 8):
            plt.plot(channels['c'+str(i)], color=colors[i])
        plt.plot(np.arange(scaled.shape[0])[scaled_peaks_high], 
                 scaled[scaled_peaks_high], 'o', color='orange')
    #    plt.plot(np.arange(scaled.shape[0])[scaled_peaks_low], 
    #             scaled[scaled_peaks_low], 'o', color='yellow')
        
        # Print the histogram
        plt.figure('histogram', figsize=(14,4))
        x = (hist[1] + ((hist[1][1] - hist[1][0]) / 2))[:-1]
        plt.plot(x, hist[0])
    #    plt.plot(x[hist_peaks], hist[0][hist_peaks], 'o', markersize=8)
        max_hist = hist[0].max()
        
        # Plot the closing position
        plt.plot([channels['closing_position'], channels['closing_position']], 
                 [0, max_hist / 2], color = 'black')
        for i in range(1, 8):
            plt.plot([channels['c'+str(i)].mean(), channels['c'+str(i)].mean()], 
                     [0, max_hist],
                     color='orange')
    #    plt.plot(scaled[scaled_peaks_high], np.ones(scaled_peaks_high.shape[0]) * max_hist * .5,'o', color='grey')
    #    plt.plot(scaled[scaled_peaks_low], np.ones(scaled_peaks_low.shape[0]) * max_hist * .25,'o', color='grey')
        plt.plot(x[keep], hist[0][keep], 'o', color = 'red')
    
        # Plot outcome widths
    #    plt.figure()
    #    if type(outcome_widths) != 'int':
    #        plt.plot(outcome_values)
    #        for ow in outcome_widths:
    #            plt.plot(np.ones(outcome_values.shape[0]) * (outcome_values[0] + ow))
    #            plt.plot(np.ones(outcome_values.shape[0]) * (outcome_values[0] - ow))
    #        print(ow)
    #    print(outcome_values.shape[0])
        
        
        
    #    # Plot the volume
    #    plt.figure('volume_spread', figsize=(14,4))
    #    plt.plot(standardscaler().fit_transform(df[['volume', 'spread']])[:, 0])
    #    plt.plot(standardscaler().fit_transform(df[['volume', 'spread']])[:, 1])
    #    
    #    # Plot polar distribution of flattened and scaled
    #    plt.figure('polar', figsize=(14, 4))
    #    ax = plt.subplot(111, projection='polar')
    #    r = np.linspace(0, 1, window_length)
    #    theta = 2 * np.pi * r
    #    ax.plot(theta, scaled[:-1])
    #
    #    plt.figure('hist_polar_channel', figsize=(14, 4))
    #    ax = plt.subplot(111, projection='polar')
    #    r = np.linspace(0, 1, hist[1][1:].shape[0])
    #    theta = 2 * np.pi * r
    #    ax.plot(theta, hist[0])   
    
    
    
    
        # Plot and print and Retrun
        plt.show()
        print('closing position: {}'.format(channels['closing_position']))
        print('Peaks:            {}'.format(x[keep]))
        print('Amplitudes:       {}'.format(hist[0][keep]))
        print('Channels Range:   {}'.format(channels['range']))

    

"""



if __name__ == '__main__':
    pass




