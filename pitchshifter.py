import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
from scipy.signal.windows import *
from itertools import product
import librosa
import IPython.display as idisplay
import fileinput
import sys

contents = fileinput.input()
print( ''.join(contents) )
def pitch_shifter(audio,scaling,n_fft=4096,win_length=4096,hop_length=1024,window_function=hann):
    stft_data = stft(audio, n_fft, win_length, hop_length, window_function=hann)
    channels, n_stft_frequencies, n_anls_frames = stft_data.shape

    #set our scaling factor
    stft_frequencies = np.arange(n_stft_frequencies)

    # determine the frequency bins of the new signal. Even if our scaling factor is greater than 1 this doesn't mean that we will be able to represent frequencies beyond the nyquist treshold 
    n_synth_frequencies = int(min(n_stft_frequencies, n_stft_frequencies * scaling))
    synth_frequencies = np.arange(n_synth_frequencies)

    # we create the "original" set of frequency bin indices, scaled according to the scaling factor. 
    og_idxs = synth_frequencies / scaling

    # calculate the phase difference alignment factor. hop len/win len gives us (in radians after multiplying for 2pi) the change of phase from one frame to the other for sinusoid that makes one cycle over the window lenght. For example, if hop_len=win_len then the pase change is of 2pi, i.e. a full circle. This is needed to "align" the phase changes that are due to the moving window (rather than to frequency changes)
    aligned_phase_diff = np.pi * 2 * hop_len / win_len

    #obtain the magnitude and phase vectors from the STFT
    magnitudes = np.abs(stft_data)
    phases = np.angle(stft_data)

    #obtain the phase difference array. Concatenate all the phases except the last one in an array along the n_stft_frequencies dimension.
    phase_differences = phases - np.concatenate((np.zeros((channels, n_stft_frequencies, 1)), phases[:, :, :-1]), axis=2)

    #We subtract from the phase difference the phase variation that is due to the hopping window. This will prevent spectral leakage caused by the algorithm misinterpreting the phase progression as variation in frequency (and consequently introducing artifacts).
    phase_differences -= (stft_frequencies * aligned_phase_diff)[None, :, None]

    #wrap around 2pi
    phase_differences = np.mod(phase_differences + np.pi, np.pi * 2) - np.pi

    #interpolation procedures

    #we need to create the new framework to "fit" our shifted windows.
    shifted_magnitudes = interpolate_freq(og_idxs, magnitudes)
    shifted_phase_differences = interpolate_freq(og_idxs, phase_differences) * scaling

    #again, adjust the phase differences removing the natural variation of phase given by the hopping window.
    shifted_phase_differences += (synth_frequencies * aligned_phase_diff)[None, :, None]

    #cumulatively sum all the phase differences to obtain the new phase values!
    shifted_phases = np.cumsum(shifted_phase_differences, axis=2)

    #create the synthesis signal by combining the magnitudes with their phases in a complex numbers
    synth_stft = shifted_magnitudes * np.exp(shifted_phases * 1j)
    #fill with zeroes any possible "holes" given by discarded frequnecies
    synth_stft = np.concatenate((synth_stft, np.zeros((channels, n_stft_frequencies - n_synth_frequencies, n_anls_frames))), axis=1)

    #perform the inverse fourier transform and display the pitch shifted audio
    res=new_waveform = librosa.istft(synth_stft, hop_length=hop_len, win_length=win_len, n_fft=n_fft)
    idisplay.display(idisplay.Audio(new_waveform, rate=sr))
    
    return res