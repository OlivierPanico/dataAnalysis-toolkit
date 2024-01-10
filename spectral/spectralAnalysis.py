from __future__ import division

''' General imports '''




import numpy as np
from numpy import fft
from math import floor
import matplotlib.pyplot as plt
from scipy.signal import coherence, filtfilt, detrend, correlate, correlation_lags
import pandas as pd 

''' local imports '''
from dataAnalysis.utils.utils import get_closest_ind, normalize_signal
from dataAnalysis.utils.array_splitting import custom_split_1d, split_array_1d, chunk_data

    
# =============================================================================
#  Main function    
# =============================================================================
    
def custom_csd(x, y, nperseg=512,noverlap=256, dt=1E-5, norm=False, window=None,remove_mean=False,nan_treatment=False, **kwargs):
    '''
    Compute cross spectrum of signal x and y. 
    If x=y, the result is a self-spectrum while for x different from y the cross-spectrum is computed.

    
    Parameters
    ----------
    x : array
        First input data.
    y : array
        Second input data.
    nperseg : int, optional
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
    spec
        autospectrum or interspectrum array.
    frq
        frequency array.
    corr
        correlation array.
    t_corr
        correlation time array.
    '''


    signals = [x, y]
    signals_ft = [None, None]
    
    for i, sig in enumerate(signals):
            
        ''' Calculation of the number of segment according to nperseg and len(x) / len(y)'''
        nseg = floor(len(x) / nperseg)
        
        ''' Slicing the original array into nseg segments'''
        if noverlap is None:
            sig = split_array_1d(sig, nperseg=nperseg)
            print('shape splitted sig: ', np.shape(sig))
            # sig = np.reshape(sig[:nseg*nperseg], (nseg, -1))  #Old method should work exactly as split_1d_array => gives weird result if nperseg*nseg = nbpts (ie if the decomposition is exact)
        else: 
            # print('chunking data')
            #sig = chunk_data(np.array(sig), nperseg, overlap_size=noverlap)    # if overlap=0 => should give the same array as split and reshape
            #print('shape chunked sig: ', np.shape(sig))
            sig = custom_split_1d(sig, nperseg=nperseg, noverlap = noverlap)
            
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
        
        if remove_mean:
            sig= detrend(sig, axis=0, type='constant')
        
        if window == 'hanning':
            W = np.hanning(nperseg)
            sig *= W
        elif window == 'hamming':
            W = np.hamming(nperseg)
            sig *= W
        elif window == 'bartlett':
            W = np.bartlett(nperseg)
            sig *= W        
        elif window == 'blackman':
            W = np.blackman(nperseg)
            sig *= W        
        elif window == 'kaiser':
            W = np.kaiser(nperseg)
            sig *= W
        
        
        '''FFT'''
        signals_ft[i] = fft.fft(sig)
        signals[i] = sig
        
    
    x,y = signals
    x_ft, y_ft = signals_ft
    
    ''' compute self-spectrum of x or cross-spectrum of x and y if y is different from x:
        The result is normalized by the effective noise bandwidth to get the rms power density '''
    if window is None:
        spec = np.mean(y_ft * np.conjugate(x_ft), axis=0) / nperseg * dt
        
    else: 
        sommation = sum(W**2)
        spec = np.mean(y_ft * np.conjugate(x_ft), axis=0) / (sommation) * dt
        
    #create a frequency array        
    frq = fft.fftfreq(spec.size, dt)

    #shift both array and spectrum to center around 0
    frq = fft.fftshift(frq)
    spec = fft.fftshift(spec)
    

    return frq, spec



def custom_coherence(x, y, nperseg=512,noverlap=256, dt=1E-5, norm=False, window=None,remove_mean=False,nan_treatment=False, **kwargs):
    '''
        If y is delayed w.r.t x, the correlation is maxima for positive times t_corr > 0, or,
    equivalently, the angle is negative for positive frequencies.
    
    i.e. Angle < 0 or correlation time > 0 => y delayed w.r.t x
    ''' 
    f, pxy = custom_csd(x,y, nperseg=nperseg,noverlap=noverlap, dt=dt, norm=norm, window=window,remove_mean=remove_mean,nan_treatment=nan_treatment)#, **kwargs)
    f, pxx = custom_csd(x,x, nperseg=nperseg,noverlap=noverlap, dt=dt, norm=norm, window=window,remove_mean=remove_mean,nan_treatment=nan_treatment)#, **kwargs)
    f, pyy = custom_csd(y,y, nperseg=nperseg,noverlap=noverlap, dt=dt, norm=norm, window=window,remove_mean=remove_mean,nan_treatment=nan_treatment)#, **kwargs)
    
    coh = abs(pxy)**2/(pxx*pyy)
    
    # corr = fft.ifft(pxy)
    
    # pxy_real = fft.ifft(pxy)/dt
    # pxx_real = fft.ifft(pxx)/dt
    # pyy_real = fft.ifft(pyy)/dt
    
    # coh = abs(pxy_real)**2/pxx_real*pyy_real
    
    return f, coh

 

def custom_time_coherence_std(x, y, nperseg=512, noverlap=256):
    '''
    Compute the normalized Pearson correlation from both array together with the lag array
    '''
    signals = [x, y]
    signals_ft = [None, None]
    
    for i, sig in enumerate(signals):
            
        ''' Calculation of the number of segment according to nperseg and len(x) / len(y)'''
        nseg = floor(len(x) / nperseg)
        
        ''' Slicing the original array into nseg segments'''
        if noverlap is None:
            sig = split_array_1d(sig, nperseg=nperseg)
            print('shape splitted sig: ', np.shape(sig))
            # sig = np.reshape(sig[:nseg*nperseg], (nseg, -1))  #Old method should work exactly as split_1d_array => gives weird result if nperseg*nseg = nbpts (ie if the decomposition is exact)
        else: 
            # print('chunking data')
            #sig = chunk_data(np.array(sig), nperseg, overlap_size=noverlap)    # if overlap=0 => should give the same array as split and reshape
            #print('shape chunked sig: ', np.shape(sig))
            sig = custom_split_1d(sig, nperseg=nperseg, noverlap = noverlap)
            
            signals[i] = sig    

    x_split = signals[0]
    y_split = signals[1]
    corr=0
    for i in range(len(x_split[:,0])):
        cloc = correlate(x_split[i,:]/np.std(x_split[i,:]), y_split[i,:]/np.std(y_split[i,:]), mode='same')/nperseg
        corr += cloc/len(x_split[:,0])

        
    
    tcorr = correlation_lags(nperseg, nperseg, mode='same')    
    
    return tcorr, corr
    

def custom_time_coherence(x, y, nperseg=512, noverlap=256):
    '''
    Compute the normalized Pearson correlation from both array together with the lag array
    '''
    signals = [x, y]
    signals_ft = [None, None]
    
    for i, sig in enumerate(signals):
            
        ''' Calculation of the number of segment according to nperseg and len(x) / len(y)'''
        nseg = floor(len(x) / nperseg)
        
        ''' Slicing the original array into nseg segments'''
        if noverlap is None:
            sig = split_array_1d(sig, nperseg=nperseg)
            print('shape splitted sig: ', np.shape(sig))
            # sig = np.reshape(sig[:nseg*nperseg], (nseg, -1))  #Old method should work exactly as split_1d_array => gives weird result if nperseg*nseg = nbpts (ie if the decomposition is exact)
        else: 
            # print('chunking data')
            #sig = chunk_data(np.array(sig), nperseg, overlap_size=noverlap)    # if overlap=0 => should give the same array as split and reshape
            #print('shape chunked sig: ', np.shape(sig))
            sig = custom_split_1d(sig, nperseg=nperseg, noverlap = noverlap)
            
            signals[i] = sig    

    x_split = signals[0]
    y_split = signals[1]
    corr=0
    
    nxsplit = len(x_split[:,0])
    print(nxsplit)
    for i in range(nxsplit):
        cloc = correlate(x_split[i,:], y_split[i,:], mode='same')/nperseg
        autocorr_x = correlate(x_split[i,:], x_split[i,:], mode='same')/nperseg
        autocorr_y = correlate(y_split[i,:], y_split[i,:], mode='same')/nperseg
        
        print(nperseg)
        
        print(autocorr_x[nperseg//2], autocorr_y[nperseg//2])
        
        corr += (cloc/np.sqrt(autocorr_x[nperseg//2] * autocorr_y[nperseg//2]))/len(x_split[:,0])

    
    
    tcorr = correlation_lags(nperseg, nperseg, mode='same')    
    
    return tcorr, corr


### TO BE DEPRECATED => COMPARE DIFFERENT COHERENCE METHODS
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



### ===== ###
### TESTS ###
### ===== ###

#Imports
from scipy.signal import unit_impulse, welch


class TestSignal():
    
    def __init__(self, f1=50, f2=70, phase1=0.3, phase2=0.7, nbpts=20000, fs=1e3, noise_amp = 0.5, phase1_noise_amp = 0.2):
        
        #Signal parameters
        self.f1 = f1
        self.f2 = f2
        self.phase1 = phase1
        self.phase2 = phase2
        self.nbpts = nbpts
        self.fs = fs
        self.noise_amp = noise_amp
        self.phase1_noise_amp = phase1_noise_amp
        
        t = np.linspace(-nbpts/(2*fs), nbpts/(2*fs)-1/fs, nbpts)
        dt = t[1]-t[0]
        noise = noise_amp*np.random.normal(0, 1, nbpts)
        phase1_noise = phase1_noise_amp * np.random.normal(0, 1, nbpts)
        
        self.t = t
        
        self.dt = dt
        self.noise = noise
        self.phase1_noise = phase1_noise
        
        #Specific noises
        n1 = 0*np.random.normal(0, 1, nbpts)
        n2 = 0*np.random.normal(0, 1, nbpts)
        # n3 = 0*np.random.normal(0, 1, nbpts)
        
        s1 = n1 + np.cos(2*np.pi*f1*t + phase1 + phase1_noise)
        s2 = n2 + np.cos(2*np.pi*f2*t + phase2)
        s3 = s1 * s2
        signal = s1 + s2 + s3 + noise
        
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.signal = signal
   
        
    def get_analytical_fft(self):
        
        f1 = self.f1
        f2 = self.f2
        phase1 = self.phase1
        phase2 = self.phase2
        
        fs = self.fs
        nbpts = self.nbpts
        
        frq = np.fft.fftfreq(nbpts,1/fs)
        frq = np.fft.fftshift(frq)
        
        f1_idxNeg = get_closest_ind(frq,-f1)
        f1_idxPos = get_closest_ind(frq,f1)
        
        f2_idxNeg = get_closest_ind(frq,-f2)
        f2_idxPos = get_closest_ind(frq,f2)
        
        f3_idxNeg = get_closest_ind(frq,-(f1 +f2))
        f3_idxPos = get_closest_ind(frq, f1 + f2)
        
        f4_idxNeg = get_closest_ind(frq,-(f1 - f2))
        f4_idxPos = get_closest_ind(frq,(f1-f2))
        
        dir1Neg = unit_impulse(nbpts,f1_idxNeg)
        dir1Pos = unit_impulse(nbpts,f1_idxPos)
        dir2Neg = unit_impulse(nbpts,f2_idxNeg)
        dir2Pos = unit_impulse(nbpts,f2_idxPos)
        dir3Neg = unit_impulse(nbpts,f3_idxNeg)
        dir3Pos = unit_impulse(nbpts,f3_idxPos)
        dir4Neg = unit_impulse(nbpts,f4_idxNeg)
        dir4Pos = unit_impulse(nbpts,f4_idxPos)
        
        spec1_analytic = 0.5* ( np.exp(1J*phase1)*dir1Pos + np.exp(-1J*phase1)*dir1Neg)
        spec2_analytic = 0.5* ( np.exp(1J*phase2)*dir2Pos + np.exp(-1J*phase2)*dir2Neg)
        spec3_analytic = 0.25* ( np.exp(1J*(phase1+phase2)) * dir3Pos + np.exp(-1J*(phase1+phase2)) * dir3Neg
                            + np.exp(1J*(phase1-phase2)) * dir4Pos + np.exp(-1J*(phase1-phase2)) * dir4Neg)
        analytic_spec = spec1_analytic + spec2_analytic + spec3_analytic
    
        return frq, analytic_spec
    
    def get_numerical_fft(self, window=None):
        
        nbpts = self.nbpts
        fs = self.fs
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        signal = self.signal
        
        #FFT
        if window is not None:
            print('window not functional yet')
            W = np.hanning(nbpts)
            sommation = sum(W**2)
            
        frq = np.fft.fftfreq(nbpts,1/fs)
        frq = np.fft.fftshift(frq)
        
        spec1_fft = np.fft.fft((s1))/(nbpts)
        spec1_fft = np.fft.fftshift(spec1_fft)
        
        spec2_fft = np.fft.fft(s2)/(nbpts)
        spec2_fft = np.fft.fftshift(spec2_fft)
        
        spec3_fft = np.fft.fft(s3)/(nbpts)
        spec3_fft = np.fft.fftshift(spec3_fft)
        
        numerical_spec = np.fft.fft(signal)/nbpts
        numerical_spec = np.fft.fftshift(numerical_spec)
        
        return frq, numerical_spec
    
    
    
    def get_welch(self, nperseg=512, window='hann', scaling='density'):
        '''
        Careful: welch returns either the power spectral density or the spectrum => in either case it gives a positive real valued array => careful with the phase ?
        '''
        signal = self.signal
        fs = self.fs 
        nbpts=self.nbpts
        
        frq, welch_spec = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2,window=window, scaling=scaling, return_onesided=False)
        
        return frq, welch_spec
    
    
    def plot_analytic_numerical_spectrum(self):
        
        frq, analytic_spec = self.get_analytical_fft()
        _, numerical_spec = self.get_numerical_fft()
        f_welch, welch_spec = self.get_welch()

        #Comparison
        plt.figure()
        plt.title('Comparison signal fft and analytic')
        
        plt.subplot(3,3,1)
        plt.title('spec fft real')
        plt.plot(frq,numerical_spec.real)
        plt.subplot(3,3,4)
        plt.title('spec fft imag')
        plt.plot(frq,numerical_spec.imag)
        plt.subplot(3,3,7)
        plt.title('spec fft amp')
        plt.plot(frq,np.sqrt(numerical_spec.real**2 + numerical_spec.imag**2))
        
        plt.subplot(3,3,2)
        plt.title('spec analytic real')
        plt.plot(frq,analytic_spec.real)
        plt.subplot(3,3,5)
        plt.title('spec analytic imag')
        plt.plot(frq,analytic_spec.imag)
        plt.subplot(3,3,8)
        plt.title('spec analytic amp')
        plt.plot(frq,np.sqrt(analytic_spec.real**2 + analytic_spec.imag**2))
        
        plt.subplot(3,3,3)
        plt.title('spec welch real')
        plt.plot(f_welch, welch_spec.real)
        plt.xlim(frq[0], - frq[0])
        plt.subplot(3,3,6)
        plt.title('spec welch imag')
        plt.plot(f_welch, welch_spec.imag)
        plt.xlim(frq[0], - frq[0])
        plt.subplot(3,3,9)
        plt.title('spec welch amp')
        plt.plot(f_welch,np.sqrt(welch_spec.real**2 + welch_spec.imag**2))
        plt.xlim(frq[0], - frq[0])
        
        plt.tight_layout()



  
if __name__ == '__main__':
    #Test correlation
    sig = TestSignal(f1=2000, f2 = 700, phase1=0, phase2=0, noise_amp=1, phase1_noise_amp=0)
    dt = 1/sig.fs
    x = sig.signal[0:18000]
    y = sig.signal[100:18100]
    f, pxy = custom_csd(y,x, 2048, 1024, window='hanning', norm=True, remove_mean=True, dt=1/sig.fs )
    W=np.hanning(1024)
    sommation=sum(W**2)
    test = (fft.ifft(pxy))/dt
    test = (fft.ifftshift(test))
    plt.figure()
    plt.plot(test)
    plt.title('Correlation function from fourier')
    # tcorr, corr = custom_time_coherence(x/np.std(x),y/np.std(y), 4096,2048)
    tcorr, corr = custom_time_coherence(x,y, 2048,1024)
    plt.figure()
    plt.plot(tcorr, corr)
    plt.title('Correlation from np correlate')
