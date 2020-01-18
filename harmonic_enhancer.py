# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:12:54 2019

@authors: Katarzyna Lenard - 2218524L & Mustafa Biyikli - 2190523B
"""

import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt

Fs, x_t = wavfile.read("original.wav")      # Load the audio sample
x_t = x_t[:, 1] / 2**15                     # Reduce to single-channel and normalise the signal (16-bit recording)
x_t = x_t[50000 : len(x_t)-35000]           # Remove muted sections [beginning & end]
resolution = Fs/len(x_t)
margin = -75                                # Margin line on amplitude[dB]/frequency[Hz] graph
y = np.linspace(margin, margin, len(x_t))   # This is to plot the margin line later on
figManager = plt.get_current_fig_manager()
figManager.resize(*figManager.window.maxsize())	    # Load the plots in a maximised window (changed for Linux)
SM58_mimic = []                             # This is used to plot the enhancer frequency response by appending ramp values to it

# Plot the audio signal against time, add title and name x & y axes
time = np.linspace(0, len(x_t)/Fs, num=len(x_t))    # Create an array of time[s] using number of samples
plt.subplot(321, facecolor="white") and plt.plot(time, x_t, "black", linewidth=0.5)
plt.title("Original Audio Signal [Time Domain]", loc="center")
plt.ylabel("Amplitude") and plt.xlabel("Time [s]")
plt.xlim(0, len(x_t)/Fs) and plt.ylim(-0.5, 0.5)
plt.grid()

# Plot the audio signal against frequency, add title and name x & y axes
x_f = np.fft.fft(x_t)  # Forward FT: Time Domain -> Frequency Domain; Make data sample size independent
x_f_plot = 20*np.log10(np.fft.fft(x_t)/len(x_t))    # Discrete FT for the plot
x_f_dB = 20*np.log10(x_f)   # Convert to dB
frequency = np.linspace(0, Fs, len(x_t))    # Create an array of frequency[Hz] using Nyquist Theorem of Fmax <= Fs/2
plt.subplot(322, facecolor="white") and plt.plot(frequency, np.real(x_f_plot), "black", linewidth=0.5)
plt.xscale("log")
plt.title("Original Audio Signal [Frequency Domain]", loc="center")
plt.ylabel("Amplitude [dB]") and plt.xlabel("Frequency [Hz]")
plt.ylim(-200, -40)
plt.xlim(1, Fs/2) # Exclude the mirrored part from the log scale start is 1 as 10**0 = 1
plt.grid()
plt.plot(frequency, y, "k--", linewidth=0.5)    # This is the margin line

#Apply a High Pass Filter (50Hz)
h1 = int(len(x_f_dB)/Fs*0)
h2 = int(len(x_f_dB)/Fs*50)
x_f_dB[h1 : h2+1] = 0
x_f_dB[len(x_f_dB)-h2 : len(x_f_dB)-h1+1] = 0

# Apply a Low Pass Filter (15kHz)
l1 = int(len(x_f_dB)/Fs*1.5e4)
l2 = int(len(x_f_dB)/Fs*Fs/2)
x_f_dB[l1 : l2+1] = 0
x_f_dB[len(x_f_dB)-l2 : len(x_f_dB)-l1+1] = 0

# 50Hz to 100Hz Region, SHURE SM-58 frequency response manipulation (Remove +8dB from 50Hz and 0dB from 100: linear correlation)
for harmonics in np.arange(50, 1e2, resolution):
    ramp = harmonics*8/50 - 16  
    k1 = int(len(x_f)/Fs*(harmonics))
    x_f_dB[k1] = x_f_dB[k1] + ramp
    x_f_dB[len(x_f_dB)-k1] = x_f_dB[len(x_f_dB)-k1] + ramp
    SM58_mimic.append(ramp)

# Leave 100Hz to 1kHz region untouched
for harmonics in np.arange(1e2, 1e3, resolution):
    ramp = 0 
    # Uncomment the following if fundemetal region is to be modified; change ramp value as desired
    '''
    k1 = int(len(x_f)/Fs*(harmonics))
    x_f_dB[k1] = x_f_dB[k1] + ramp
    x_f_dB[len(x_f_dB)-k1] = x_f_dB[len(x_f_dB)-k1] + ramp
    '''
    SM58_mimic.append(ramp)

# 1kHz to 5.5kHz Region, SHURE SM-58 frequency response manipulation (Add 0dB to 1kHz and +5dB to 5.5kHz: linear correlation)
for harmonics in np.arange(1e3, 5.5e3, resolution):
    ramp = harmonics/900 - 10/9  
    k1 = int(len(x_f)/Fs*(harmonics))
    x_f_dB[k1] = x_f_dB[k1] + ramp
    x_f_dB[len(x_f_dB)-k1] = x_f_dB[len(x_f_dB)-k1] + ramp
    SM58_mimic.append(ramp)
    
# 5.5kHz to 6kHz Region, SHURE SM-58 frequency response manipulation (Add +5dB from 5kHz to 6kHz: linear correlation)
for harmonics in np.arange(5.5e3, 6e3, resolution):
    ramp = 5  
    k1 = int(len(x_f)/Fs*(harmonics))
    x_f_dB[k1] = x_f_dB[k1] + ramp
    x_f_dB[len(x_f_dB)-k1] = x_f_dB[len(x_f_dB)-k1] + ramp
    SM58_mimic.append(ramp)
    
# 6kHz to 8kHz Region, SHURE SM-58 frequency response manipulation (Add +5dB to 6kHz and +1dB to 8kHz: linear correlation)
for harmonics in np.arange(6e3, 8e3, resolution):
    ramp = 2.4e4*4/harmonics - 11
    k1 = int(len(x_f)/Fs*(harmonics))
    x_f_dB[k1] = x_f_dB[k1] + ramp
    x_f_dB[len(x_f_dB)-k1] = x_f_dB[len(x_f_dB)-k1] + ramp
    SM58_mimic.append(ramp)
      
# 8kHz to 10kHz Region, SHURE SM-58 frequency response manipulation (Add +1dB to 8kHz and +4dB to 10kHz: linear correlation)
for harmonics in np.arange(8e3, 1e4, resolution):
    ramp = harmonics*3/2000 - 11
    k1 = int(len(x_f)/Fs*(harmonics))
    x_f_dB[k1] = x_f_dB[k1] + ramp
    x_f_dB[len(x_f_dB)-k1] = x_f_dB[len(x_f_dB)-k1] + ramp
    SM58_mimic.append(ramp)

# 10kHz to 15kHz Region, SHURE SM-58 frequency response manipulation (Add +4dB to 10kHz and remove +7dB from 15kHz: linear correlation)
for harmonics in np.arange(1e4, 1.5e4, resolution):
    ramp = 1.5e4*11/harmonics*2 - 29
    k1 = int(len(x_f)/Fs*(harmonics))
    x_f_dB[k1] = x_f_dB[k1] + ramp
    x_f_dB[len(x_f_dB)-k1] = x_f_dB[len(x_f_dB)-k1] + ramp
    SM58_mimic.append(ramp)    

# Plot the improved audio signal against frequency, add title and name x & y axes
x_f_plot = 10**(x_f_dB/20)
plt.subplot(324, facecolor="white") and plt.plot(frequency, np.real(20*np.log10(x_f_plot/len(x_t))), "black", linewidth=0.5)
plt.xscale("log")
plt.title("Improved Audio Signal [Frequency Domain]", loc="center")
plt.ylabel("Amplitude [dB]") and plt.xlabel("Frequency [Hz]")
plt.ylim(-200, -40)
plt.xlim(1, Fs/2) # Remove the mirrored part from the log scale start is 1 as 10**0 = 1
plt.grid()
plt.plot(frequency, y, "--k", linewidth=0.5)

# Transform the improved audio back to time domain
sound_clean = np.fft.ifft((10**(x_f_dB/20))*(2**15)) # Convert back to x[t]
sound_clean = np.real(sound_clean)
sound_clean = np.asarray(sound_clean, dtype=np.int16)

# Plot the improved audio signal against time, add title and name x & y axes
plt.subplot(323, facecolor="white") and plt.plot(time, sound_clean/2**15, "black", linewidth=0.5)
plt.title("Improved Auido Signal [Time Domain]")
plt.ylabel("Amplitude") and plt.xlabel("Time [s]")
plt.xlim(0, len(x_t)/Fs) and plt.ylim(-0.5, 0.5)
plt.grid()

# Not necessary, plotted just to justify 
plt.subplot(325)
plt.specgram(x_t, Fs=Fs)
plt.title("Original Audio Spectrogram")
plt.ylabel("Frequency [Hz]") and plt.xlabel("Time [s]")

# Plot the enhancer frequency response which shows how the signal is manipulated
plt.subplot(326, facecolor="white") and plt.plot(np.linspace(50, 15000, num=len(SM58_mimic)), SM58_mimic, "black", linewidth=2)
plt.xscale("log")
plt.title("Enhancer Frequency Response")
plt.ylabel("Manipulation [dB]") and plt.xlabel("Frequency [Hz]")
plt.xlim(10**0, Fs/2)
xmajor_ticks = [20, 50, 100, 1000, 10000]
ymajor_ticks = np.arange(-10, 11, 5)
plt.xticks(xmajor_ticks, xmajor_ticks)
plt.yticks(ymajor_ticks, ymajor_ticks)
plt.tick_params(axis='both', which='major')
plt.grid(which='both', alpha=1)

# Titles and axes labels overlap, this solves them; could also use plt.tight_layout()
plt.subplots_adjust(wspace=0.25, hspace=0.6)
plt.show()

wavfile.write("improved.wav", Fs, sound_clean)  # Save the improved audio in the same directory
