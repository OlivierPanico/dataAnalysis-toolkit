from __future__ import division

''' General imports '''
from numpy.lib.stride_tricks import as_strided as ast   #Used to chunk an array into overlapping segments

import numpy as np
from numpy import fft
from math import floor
import matplotlib.pyplot as plt
from scipy.signal import coherence, filtfilt, detrend
import pandas as pd 

''' local imports '''
from dataAnalysis.utils import normalize_signal, split_1d_array


# import global_params
# from read_profiles_pandas import get_clean_dyn_characteristic, find_nearest
# from filter import butter_bandpass
# from read_profiles import path_to_data_dyn


def chunk_data(data,window_size,overlap_size=0,flatten_inside_window=True):
    '''
    !!! CAREFUL I THINK IT IS NOT FUNCTIONAL  => use np.split instead !!!
    
    This function is used to chunk an array into windows of size window_size with an overlapping size of overlap_size.
    If the last window is not full, it will pad the last empty spaces with zeros
    
    
    Parameters
    ----------
    data : array
        The array we want to slice into pieces.
    window_size : int
        size of the windows (number of points).
    overlap_size : int, optional
        size of the overlap (number of points). The default is 0.
    flatten_inside_window : bool, optional
        Reduit le nombre de dimension du tableau final. The default is True.

    Returns
    -------
    narray
        Sliced array.

    '''
    
    
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1,1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows*window_size - (num_windows-1)*overlap_size,data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(
            data,
            shape=(num_windows,window_size*data.shape[1]),
            strides=((window_size-overlap_size)*data.shape[1]*sz,sz),
            writeable=False
            )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows,-1,data.shape[1]))
    
# =============================================================================
#  Main function    
# =============================================================================
    
def compute_spectral_data(x, y, nfft=512,noverlap=256, dt=1E-5, norm=True, window=None,coherence=False,nan_treatment=False, **kwargs):
    '''
    Compute spectrum and correlation of signal x and y.
    If x=y, the result is a self-spectrum while for x different from y the cross-spectrum is computed.
    If y is delayed w.r.t x, the correlation is maxima for positive times t_corr > 0, or,
    equivalently, the angle is negative for positive frequencies.
    
    i.e. Angle < 0 or correlation time > 0 => y delayed w.r.t x
    
    Parameters
    ----------
    x : array
        First input data.
    y : array
        Second input data.
    nfft : int, optional
        Number of frequency bins. The default is 512.
    noverlap : int, optional
        Number of overlapping points. The default is 256.
    dt : float, optional
        Sampling time. The default is 1E-5.
    norm : bool, optional
        If we want to normalize the data. i.e. detrending constant values per segment. The default is True.
    window : str, optional
        Name of the corresponding window: 'hanning', 'hamming', 'bartlett', 'kaiser', 'blackman'. Careful : the use of windowing will lower spectrum frequency and reduce the correlation. The default is None.
    coherence : bool, optional
        NOT WORKING
        To compute the coherence instead of spectrum and correlation. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    frq
        frequency array.
    spec
        autospectrum or interspectrum array.
    corr
        correlation array.
    t_corr
        correlation time array.
    '''


    signals = [x, y]
    signals_ft = [None, None]
    
    for i, sig in enumerate(signals):
            
        ''' Calculation of the number of segment according to nfft and len(x) / len(y)'''
        nseg = floor(len(x) / nfft)
        
        ''' Slicing the original array into nseg segments'''
        if noverlap is None:
            sig = split_1d_array(sig, nperseg=nfft)
            print('shape splitted sig: ', np.shape(sig))
            # sig = np.reshape(sig[:nseg*nfft], (nseg, -1))  #Old method, still here to compare with the function chunk data
        else: 
            # print('chunking data')
            sig = chunk_data(np.array(sig), nfft, overlap_size=noverlap)    #An overlap of 0 will give the same result as noverlap=None
            print('shape chunked sig: ', np.shape(sig))
        
        if nan_treatment:
            sig = pd.DataFrame(sig)
            sig_list = [] #temporary array to fill the 'correct slices'
        
            ''' If there are too many NaN in a nseg will remove the whole list
            If the number of NaN is acceptable, will interpolate them with first ok value by linear method'''
            for j in range(nseg): 
                current_sig = sig.iloc[j,:] #Take only a segment
                nancount = current_sig.isna().sum() #Count the number of NaN
                if nancount > len(current_sig)/2:
                    print(j, 'too many NaN values')
                    continue
                else: 
                    sig_list.append(current_sig)  
            
            '''Change back to a DataFrame to perform interpolation (faster)'''
            sig_df = pd.DataFrame(sig_list)
            sig = sig_df.interpolate(axis=1, limit_direction='both')
            ''' Turn back a numpy array to perform the FFT'''
            sig = np.array(sig)

        # print(sig.shape)
        '''Normalisation'''
        if norm:
            sig = normalize_signal(sig, perline=True)        
                
        
        if window == 'hanning':
            W = np.hanning(nfft)
            sig *= W
        elif window == 'hamming':
            W = np.hamming(nfft)
            sig *= W
        elif window == 'bartlett':
            W = np.bartlett(nfft)
            sig *= W        
        elif window == 'blackman':
            W = np.blackman(nfft)
            sig *= W        
        elif window == 'kaiser':
            W = np.kaiser(nfft)
            sig *= W
        
        '''FFT'''
        signals_ft[i] = fft.fft(sig)
        signals[i] = sig
        
    
    x,y = signals
    x_ft, y_ft = signals_ft
    
    ''' compute self-spectrum of x or cross-spectrum of x and y if y is different from x:
        The result is normalized by the effective noise bandwidth to get the rms power density '''
    if window is None:
        spec = np.mean(y_ft * np.conjugate(x_ft), axis=0) / nfft * dt
        
    else: 
        sommation = sum(W**2)
        spec = np.mean(y_ft * np.conjugate(x_ft), axis=0) / (sommation) * dt
        
        
    frq = fft.fftfreq(spec.size, dt)
    corr = fft.ifft(spec) / dt
    t_corr = np.arange(-nfft/2, nfft/2) * dt


    if (np.isreal(x) & np.isreal(y)).all():
        corr = np.real(corr)

    frq = fft.fftshift(frq)
    spec = fft.fftshift(spec)
    corr = fft.fftshift(corr)
    
    if coherence:
         product = y_ft*np.conjugate(x_ft)
         print(np.shape(product))
         y_ft_mean = np.mean(abs(y_ft), axis=0)
         x_ft_mean = np.mean(abs(x_ft), axis=0)
         coh = np.mean(product, axis=0) / (y_ft_mean * x_ft_mean)
         
         return frq, coh
     
    # remove unphysical spikes appearing sometimes at f=-50kHz
    # spec[0] = spec[1]

    return spec, frq, corr, t_corr 


def get_coherence(probe,position,method=1,characteristic='ne',nfft = 512, plot=True, ax=None ):
    '''
    to get the coherence of two signals
    inputs:
        probe : list of two elements, [1,2] is a good choice
        position : radial position at which to compute the signals
        method = 1 : uses the coherence function from scipy.signal
        method = 2 : compute the formula from the cross and auto spectra
        method = 0 : do both and plot the results
    return:
        frequency array
        coherence array
        return the 4 arrays in the case of method = 0
    '''
    
    # Choose automatically the correct signals
    params = [probe, position, characteristic]
    for i, param in enumerate(params):
        if type(param)==list:
            if len(param)==1: params[i] = 2*param
            elif len(param)==2: pass
            else: print('Invalid parameter', str(param),'\n must be a python list of two elements max.')
        else: params[i] = [param, param]
    probe, position, characteristic = params

    signals = [None, None]
    signals2 = [None, None]
    for i in range(2):
        # Old method 
        # fname = path_to_data_dyn(probe[i], position[i])
        # mat_contents = sio.loadmat(fname)
        # signals[i] = mat_contents[characteristic[i]].flatten()
        
        ''' New method with pandas '''
        df_characteristics, df_errors=get_clean_dyn_characteristic(probe[i],position[i], recalibrate_3=True, info=False)
    
        signals[i] = df_characteristics[characteristic[i]]
        signals[i] = list(signals[i])
        
        if method==0 or method==1:
            # df_characteristics2 = df_characteristics.interpolate(axis=1,limit_direction='both')
            signals2[i] = df_characteristics[characteristic[i]]
            signals2[i].interpolate(axis=1, limit_direction='both', inplace=True)
            signals2[i] = list(signals2[i])
        
    # method 1 uses the coherence function from scipy.signal
    if method==1:
        frq_scipy, coh_scipy = coherence(signals2[0],signals2[1],nperseg=None,noverlap= None, fs = 100000, nfft = nfft,detrend='constant')
        
        
    # method 2 calculate the coherence function from the spectrum and the formula
    elif method==2:
         # interprobe spectrum 
        spec_interprobes, frq, _,_ = compute_spectral_data(signals[0], signals[1], nfft)
        # Auto-spectrum of each probe 
        spec_auto1, _, _,_ = compute_spectral_data(signals[0], signals[0], nfft)
        spec_auto2, _, _, _ = compute_spectral_data(signals[1], signals[1], nfft)
        
        coh = np.abs(spec_interprobes)**2 / (spec_auto1 * spec_auto2) 
        coh = coh.real
    # method 0 use both methods to compare
    elif method==0:
        # frq_scipy, coh_scipy = coherence(signals[0],signals[1], fs = 100000, nfft = nfft)
        frq_scipy, coh_scipy = coherence(signals2[0],signals2[1],nperseg=None,noverlap= 200, fs = 100000, nfft = None,detrend='constant')
        # interprobe spectrum 
        spec_interprobes, frq, _,_ = compute_spectral_data(signals[0], signals[1], nfft)
        # Auto-spectrum of each probe 
        spec_auto1, _, _,_ = compute_spectral_data(signals[0], signals[0], nfft)
        spec_auto2, _, _, _ = compute_spectral_data(signals[1], signals[1], nfft)
        
        coh = np.abs(spec_interprobes)**2 / (spec_auto1 * spec_auto2) 
        
        fig , ax = plt.subplots()
        ax.plot(frq[int(nfft/2):],coh[int(nfft/2):], label = 'Coherence home made')
        ax.plot(frq_scipy,coh_scipy,label='Coherence from scipy module')
        ax.legend()
        ax.set_xlabel(r'f [Hz]')
        ax.set_ylabel(r'coherence')
        
    else:
        print('Please choose method = 1 for using scipy.signal, method = 2 for using home made code or mehtod = 3 for both')
    
    if method==1:
        return frq_scipy, coh_scipy
    elif method==2:
        return frq,coh
    elif method==0:
        return frq_scipy,coh_scipy,frq,coh 
    else:
        return


  
# if __name__ == '__main__':
