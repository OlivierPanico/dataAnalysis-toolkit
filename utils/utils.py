# =============================================================================
#  Authors
# =============================================================================
# Name: Olivier PANICO
# corresponding author: olivier.panico@free.fr

# =============================================================================
#  Imports
# =============================================================================
#General imports
import os
from os.path import isdir, join

import numpy as np
import numpy.ma as ma

from scipy.ndimage import uniform_filter #Coarse graining
from scipy.optimize import curve_fit   


def get_closest_ind(L,val):
    ires=0
    diff_min=abs(L[0]-val)
    for i in range(len(L)):
        diff=abs(L[i]-val)
        if diff<diff_min:
            ires = i
            diff_min = diff
    return ires


def coarse_grain(array, t_avg, x_avg, mode='constant'):
    return uniform_filter(array, size=(t_avg, x_avg), mode=mode)
    
    
def my_linearRegression(xdata, ydata, mode='linear'):
    
    if mode=='linear':
        def fit_func(x,a):
            return a*x
    elif mode=='affine':
        def fit_func(x,a,b):
            return b + a*x
    
       
    if xdata.ndim != 1:
        xdata = np.ndarray.flatten(xdata)
    if ydata.ndim !=1:
        ydata = np.ndarray.flatten(ydata)
    
    popt, pcov = curve_fit(fit_func,xdata,ydata)
    #Error calculation
    perr = np.sqrt(np.diag(pcov))
    #R-squared coefficient calculation https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
    residuals = ydata-fit_func(xdata,*popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared= 1-(ss_res/ss_tot)
    
    return popt, perr, r_squared
    
    
def normalize_signal(x, perline =False):
    '''
    Normalization of an array: removing mean and dividing by the deviation. 
    Will either normalize the whole array or treat it line by line.

    Parameters
    ----------
    x : array
        Input data.
    perline : bool, optional
        use perline to normalize line by line in the array. The default is False.

    Returns
    -------
    xnorm : array
        Normalized array.

    '''
    xnorm = np.zeros_like(x)
    
    if perline:
        for i in range(len(x[:,0])):
            mean= np.mean(x[i,:])
            deviation = np.std(x[i,:])
            xnorm[i,:] = (x[i,:] - mean) / deviation
            
    else: 
        xnorm = (x - np.mean(x)) / np.std(x)
        
    return xnorm




def remove_outliers(array, q_inf, q_sup):
    '''
    Take and array and remove the values below the quantile q_inf and above quantile q_sup
    input: ndarray
    q_inf: inferior quantile (float between 0 and 1)
    q_sup: superior quantile (float between 0 and 1)
    
    return: the masked array (careful: properties of masked arrays are special, do not use it in calculations)
    '''
    threshold_inf = np.quantile(array, q_inf)
    threshold_sup = np.quantile(array, q_sup)
    
    #Remove the values below threshold inf
    mask_inf = array<threshold_inf
    filtered_array = ma.masked_array(array, mask_inf)

    #Remove the values below threshold sup
    mask_sup = filtered_array>threshold_sup
    masked_array = ma.masked_array(filtered_array, mask_sup)
    
    return masked_array








