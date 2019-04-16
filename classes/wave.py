import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import mean_squared_error
from scipy.optimize import leastsq
from scipy.stats import spearmanr
import os; os.chdir('/northbend')
from classes.channel import Channel
from libraries.correlation import get_autocorrelation



class Wave():    
    
    
    def mse(self, x, y):
        mse = mean_squared_error(x, y)
        # print(mse)
        mse /= (x.std() * y.std())
        return mse  
    
    def __init__(self, values, channel_std=2):
            
        '''
        Do i even need to flatten it?
        '''
    
        channel = Channel(values)
        autocorr = get_autocorrelation(channel.flattened)['autocor']
        margin               = int(values.shape[0] * .10) 
        corr                 = autocorr[margin:-margin]
        maximum              = corr.argmax()#[:, 1].argmax()
        minimum              = corr.argmin()#[:, 1].argmin()
        corr_period          = min(int(values.shape[0] * .75), 2 * abs(corr[maximum] - corr[minimum])) #abs(corr[maximum, 0] - corr[minimum, 0]))
        
        # Get corr peaks ( maybe good indicator of 'smoothness'
        smoothness = int(values.shape[0]  * .25)
        corr_smoothed        = corr#[:, 1]
        corr_smoothed        = pd.DataFrame(corr_smoothed).rolling(smoothness).mean().values.ravel()
        corr_smoothed        = corr_smoothed[smoothness:]
        left = (corr_smoothed[1:] > corr_smoothed[:-1])
        right = (corr_smoothed[1:] < corr_smoothed[:-1])
        auto_peaks    = (left[:-1] & right[1:]).sum()
        if corr_smoothed[0] > corr_smoothed[1]:
            auto_peaks += 1
        if corr_smoothed[-1] > corr_smoothed[-2]:
            auto_peaks += 1
            
        # Assign first guesses for wave
        c0 = channel.flattened[0] - channel.channel_deviation * 2
        amplitude            = channel.channel_deviation * 2
        frequency_guess      = values.shape[0] / corr_period  
        phase_shift_guess    = - np.argmax(channel.flattened <  c0)
        vertical_shift_guess = amplitude + c0
        # Get Real Wave
        t = np.linspace(0, 2*np.pi, channel.flattened.shape[0])
        optimize_func = lambda x: amplitude * np.sin(x[0] * t + x[1]) + x[2]  - channel.flattened
        est_frequency, est_phase_shift, est_vertical_shift = \
                leastsq(optimize_func, [frequency_guess, phase_shift_guess, vertical_shift_guess], full_output=True)[0]
        # assess fit
        wave = amplitude * np.sin(est_frequency * t + est_phase_shift) + est_vertical_shift
        
        # Provide for the tangent
        cosine = amplitude * np.cos(est_frequency * t + est_phase_shift) + est_vertical_shift
        
        #if desired, leastsq get me some info on how well each parm fts
        # wave_parameter_fits = leastsq(optimize_func, [frequency_guess, phase_shift_guess, vertical_shift_guess], full_output=True)[2]['qtf']

        self.channel = Channel(values)
        self.amplitude = amplitude
        self.frequency = est_frequency
        self.phase_shift = est_phase_shift
        self.vertical_shift = est_vertical_shift
        self.channel_wave = wave
        self.phase_position = 0 # where in phase (%?) was last position
        self.cosine = cosine
        self.tangent = cosine[-1] / self.channel.channel_deviation / 2
        self.basis = values
    
        x = np.arange(values.shape[0])
        self.linregress = self.channel.slope * x + self.channel.intercept
        self.wave = self.channel_wave + self.linregress
        self.fit = self.mse(self.wave, values)
        
        # Shit.  Wanted to collect the error terms as well


    def extension(self, prediction_values):
        prediction_range = prediction_values.shape[0]
        channel_range = self.wave.shape[0]
        pred_channel_ratio = (prediction_range + channel_range) / channel_range
        t = np.linspace(0, pred_channel_ratio * 2 *np.pi, channel_range + prediction_range)
        x = np.arange(self.channel_wave.shape[0] + prediction_values.shape[0])
        line = self.channel.slope * x + self.channel.intercept
        prediction_wave =  self.amplitude * np.sin(self.frequency * t + self.phase_shift) + self.vertical_shift
        self.channel_wave_extension = prediction_wave[channel_range:]
        self.wave_extension = (prediction_wave + line)[channel_range:]
        self.extension_r2 = 0
        self.extension_fit = self.mse(self.wave_extension, prediction_values)
        return None        
    
    
    def plot(self, prediction_values):
        self.extension(prediction_values)

        # Instantiate Plot and Create x values
        plt.figure(figsize=(8, 3))
        x1 = np.arange(self.basis.shape[0])
        x2 = np.arange(prediction_values.shape[0]) + x1[-1] + 1
        # Plt fit values and wave
        plt.plot(self.basis)    
        plt.plot(x1, self.wave)
        # PLot prediction wave and values
        plt.plot(x2, prediction_values, color='black')
        plt.plot(x2, self.wave_extension)        
        # Plot lin line and channel
        plt.plot(x1, self.linregress, color='yellow')
        plt.plot(x1, (self.linregress + self.channel.channel_deviation * 2), color='yellow')
        plt.plot(x1, (self.linregress - self.channel.channel_deviation * 2), color='yellow')
        plt.plot(x1, (self.linregress + self.channel.channel_deviation * 1), color='yellow')
        plt.plot(x1, (self.linregress - self.channel.channel_deviation * 1), color='yellow')
        # Show plot
        plt.tight_layout()
        plt.show()
        
        # Pritn plot info
        print('Wave Fit:      {}'.format(self.fit))
        print('Pred Fit:      {}'.format(self.extension_fit))
        print('determ:        {}'.format(self.extension_r2))
        print('Frequency:     {}'.format(self.frequency))
        print('Tangent:       {}'.format(self.tangent))
        print('Phase shift:   {}'.format(self.phase_shift))
        print('Phase pos:     {}'.format(self.phase_position))    


    
    
    
    
    def cosine(self, values):
        # prediction_cosine = wave.amplitude * np.cos(wave.frequency * t + wave.phase_shift) + wave.vertical_shift
        pass
    
    
 
        
    def get_error(as_type, wave, values, cosine, std_dev):
        # All erros start the same
        errors = (values - wave) * (values - wave)
        errors /= std_dev
        if as_type != 'msse':
            # If values above wave when wave goin up, et error to 0
            errors[(cosine > 0) & (values > wave)] = 0
            # Same but opposte going down
            errors[(cosine < 0) & (values < wave)] = 0
        return errors
    