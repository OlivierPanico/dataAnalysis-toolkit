# %% Full coherence analysis function 
import numpy as np 
import matplotlib.pyplot as plt
from dataAnalysis.utils.utils import get_closest_ind, normalize_array_1d, my_linearRegression
from dataAnalysis.utils.array_splitting import custom_split_1d
from dataAnalysis.utils.plot_utils import plot_1d, my_text, my_legend
from dataAnalysis.spectral.spectralAnalysis import custom_csd, custom_time_coherence
import scipy
import scipy.signal as signal

'''
Input: zref, zhop, dt
Output: corr, spectral_coh
'''


def normalize_complex_1d(a):
    a_r = normalize_array_1d(a.real)
    a_i = normalize_array_1d(a.imag)
    return a_r + 1j*a_i

# Step 0: normalization of the signals
def normalize_signals(zref, zhop):
    zref_norm = normalize_complex_1d(zref)
    zhop_norm = normalize_complex_1d(zhop)
    return zref_norm, zhop_norm

# Step 0bis: choose mode
def choose_mode(zref_norm, zhop_norm, mode='full'):
    '''
    mode='full' => compute coherence on the full complex signal
    mode='amp' => compute coherence on the amplitude signal
    mode='real' => compute coherence on the real signal
    mode='imag' => compute coherence on the imag signal
    mode='phase' => compute coherence on the phase signal
    '''
    if mode == 'amp':
        zref_norm = np.sqrt(zref_norm.real**2 + zref_norm.imag**2)
        zhop_norm = np.sqrt(zhop_norm.real**2 + zhop_norm.imag**2)
    elif mode=='real':
        zref_norm = zref_norm.real
        zhop_norm = zhop_norm.real
    elif mode=='imag':
        zref_norm = zref_norm.imag
        zhop_norm = zhop_norm.imag
    elif mode=='phase':
        zref_norm = np.angle(zref_norm)
        zhop_norm = np.angle(zhop_norm)
    else:
        pass
    return zref_norm, zhop_norm

# Step 1: compute psd and csd
def compute_psd_csd(zref_norm, zhop_norm, dt, nperseg=1024, noverlap=512, window=None, remove_mean=True):
    fref, psd_ref = custom_csd(zref_norm, zref_norm, dt=dt, nperseg=nperseg, noverlap=noverlap, window=window,norm=True, remove_mean=remove_mean)
    fhop, psd_hop = custom_csd(zhop_norm, zhop_norm,dt=dt, nperseg=nperseg, noverlap=noverlap, window=window, norm=True, remove_mean=remove_mean)
    fcsd, csd = custom_csd(zref_norm, zhop_norm, dt=dt, nperseg=nperseg, noverlap=noverlap, window=window, norm=True, remove_mean=remove_mean)
    return fref, psd_ref, fhop, psd_hop, fcsd, csd

# Step 2: normalization of csd by psd to obtain spectral coherence
def compute_spectral_coherence(psd_ref, psd_hop, csd):
    spectral_coh = abs(csd)**2/(psd_ref*psd_hop)
    return spectral_coh

# Step 3: ifftshift + inverse Fourier transform to obtain the correlation function
def compute_correlation_function(csd, dt, nperseg = 1024, mode='full'):
    csd_ifftshift = np.fft.ifftshift(csd)
    corr_from_csd = np.fft.ifft(csd_ifftshift)/dt       
    tcorr_spec = scipy.signal.correlation_lags(nperseg, nperseg, mode='same')*dt
    return corr_from_csd, tcorr_spec

# Step 4: fftshift on the correlation function to replace peak at the center
def shift_correlation_function(corr_from_csd):
    corr = np.fft.fftshift(corr_from_csd)
    return corr


# second possibility: use the scipy.signal.correlate function
def scipy_correlation_function(zref_norm, zhop_norm, dt, nperseg=1024, noverlap=512, window=None, remove_mean=True):
    
    zref_split = custom_split_1d(zref_norm, nperseg, noverlap=noverlap, zero_padding=False)
    zhop_split = custom_split_1d(zhop_norm, nperseg, noverlap=noverlap, zero_padding=False)
    nbseg = len(zref_split[:,0])
    corr = np.zeros((nperseg), dtype=zref_norm.dtype)
    
    for i in range(nbseg):
        if remove_mean:
            zref_seg = zref_split[i,:]-np.mean(zref_split[i,:])
            zhop_seg = zhop_split[i,:]-np.mean(zhop_split[i,:])
        else:
            zref_seg = zref_split[i,:]
            zhop_seg = zhop_split[i,:]
        
        corr_seg=signal.correlate(zref_seg, zhop_seg, mode='same') / np.sqrt(signal.correlate(zref_seg,zref_seg, mode='same')[int(nperseg/2)] * signal.correlate(zhop_seg,zhop_seg, mode='same')[int(nperseg/2)])
        corr+=corr_seg/nbseg
    
    delay_arr = signal.correlation_lags(nperseg,nperseg, mode='same')*dt
               
    return delay_arr, corr


# Full coherence analysis function
def full_coherence_analysis(zref, zhop, dt, nperseg=1024, noverlap=512, window=None, remove_mean=True, plot=False,ax=None, mode='full', verbose=False):

    # prepare signals
    zref_norm, zhop_norm = normalize_signals(zref, zhop)
    zref_norm, zhop_norm = choose_mode(zref_norm, zhop_norm, mode=mode)

    # fourier space
    fref, psd_ref, fhop, psd_hop, fcsd, csd = compute_psd_csd(zref_norm, zhop_norm, dt, nperseg=nperseg, noverlap=noverlap, window=window, remove_mean=remove_mean)
    spectral_coh = compute_spectral_coherence(psd_ref, psd_hop, csd)
    
    # from fourier to real space
    corr_from_csd, tcorr_spec = compute_correlation_function(csd, dt, nperseg=nperseg)
    corr = shift_correlation_function(corr_from_csd)
    
    
    # for tests (correlation directly in real space using scipy.signal.correlate)
    # tcorr_scipy, corr = scipy_correlation_function(zref_norm, zhop_norm, dt, nperseg=nperseg, noverlap=noverlap, window=window, remove_mean=remove_mean)
    
    if plot:
        # Plotting the results
        if ax is None:
            fig, ax = plot_1d([], [], grid=True)
        # ax.plot(tcorr_spec*1e6,(corr), color='darkorchid', label='correlation function', markersize=2)
        ax.plot(tcorr_spec*1e6,np.sqrt(corr.real**2+corr.imag**2), label='amplitude', marker='')
        ax.set_xlabel(r'delay $[\mu_s]$')
        ax.set_ylabel('correlation')
        # plt.legend()
        plt.title('Spectral correlation function')
        plt.xlim(-20, 20)
        plt.ylim(-0.4, 1)
    
    if verbose:
        print('Full coherence analysis done')
    
    return tcorr_spec, corr, fcsd, spectral_coh

# step 5: get maximum of correlation & associated delay 
def get_max_corr_delay(tcorr, corr):
    max_corr = np.max(abs(corr))
    max_corr_ind = get_closest_ind(abs(corr), max_corr)
    max_corr_delay = tcorr[max_corr_ind]
    return max_corr_delay, max_corr



def plot_correlation_slopes(xdata, ydata, rho_s=None, ind_turb=None, ind_aval=None, ind_turb_pos=None, caption=True, xunit='rho_s', ax=None, color='teal', ylog=True):
    '''
    xunit: 'rho_s' or 'cm'
    '''
    if xunit=='cm':
        xdata = xdata*rho_s
        
    if ax is None:
        fig, ax = plot_1d([], [], grid=True)
    
    ax.plot(xdata, (ydata), 'o', color=color)
    if caption:
        my_text(ax, 0.4, 0.9, r'$\rho_s$ = {:.2f} cm'.format(rho_s[0]), fontsize=12, color='k')

    if ind_turb is not None:
        ind_turb_min=ind_turb[0]    
        ind_turb_max=ind_turb[1]
        popt, perr, rsquared = my_linearRegression(xdata[ind_turb_min:ind_turb_max], np.log(ydata[ind_turb_min:ind_turb_max]), mode='affine')
        ax.plot(xdata[ind_turb_min:ind_turb_max], np.exp(popt[0]*xdata[ind_turb_min:ind_turb_max]+ popt[1]), 'r', marker='')
        if caption:
            if xunit=='rho_s':
                my_text(ax, 0.15, 0.9, r'$L_c \approx$ {:.2f} $\rho_s$'.format(1/popt[0]), fontsize=12, color='r')
            elif xunit=='cm':
                my_text(ax, 0.15, 0.9, r'$L_c \approx$ {:.2f} cm'.format(1/popt[0]), fontsize=12, color='r')
            
    if ind_aval is not None:
        ind_aval_min=ind_aval[0]
        ind_aval_max=ind_aval[1]
        popt, perr, rsquared = my_linearRegression(xdata[ind_aval_min:ind_aval_max], np.log(ydata[ind_aval_min:ind_aval_max]), mode='affine')
        ax.plot(xdata[ind_aval_min:ind_aval_max], np.exp(popt[0]*xdata[ind_aval_min:ind_aval_max] + popt[1]), 'g', marker='')
        if caption:
            if xunit=='rho_s':
                my_text(ax, 0.15, 0.75, r'$L_a \approx$ {:.2f} $\rho_s$'.format(1/popt[0]), fontsize=12, color='g')
            elif xunit=='cm':
                my_text(ax, 0.15, 0.75, r'$L_a \approx$ {:.2f} cm'.format(1/popt[0]), fontsize=12, color='g')

    if ind_turb_pos is not None:
        ind_turb_min=ind_turb_pos[0]
        ind_turb_max=ind_turb_pos[1]
        popt, perr, rsquared = my_linearRegression(xdata[ind_turb_min:ind_turb_max], np.log(ydata[ind_turb_min:ind_turb_max]), mode='affine')
        ax.plot(xdata[ind_turb_min:ind_turb_max], np.exp(popt[0]*xdata[ind_turb_min:ind_turb_max]+ popt[1]), 'b', marker='')
        if caption:
            if xunit=='rho_s':
                my_text(ax, 0.15, 0.6, r'$L_c \approx$ {:.2f} $\rho_s$'.format(1/popt[0]), fontsize=12, color='b')
            elif xunit=='cm':
                my_text(ax, 0.15, 0.6, r'$L_c \approx$ {:.2f} cm'.format(1/popt[0]), fontsize=12, color='b')

    if ylog:
        plt.yscale('log')
    
    plt.ylim(0.1,1.1)
    plt.ylabel('correlation')
    plt.axhline(1, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')
    
    if xunit=='rho_s':
        plt.xlabel(r'$\Delta$ $[\rho_s]$')
    elif xunit=='cm':
        plt.xlabel(r'$\Delta$ $[cm]$')
    
    return ax



# %%
