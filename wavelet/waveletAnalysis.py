# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:18:09 2023

@author: OP269158
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt


### ============= ###
### MAIN FUNCTION ###
### ============= ###
def myCWT(signal, Nscales, wavelet_name='morl', sampling_period=1):
    coefs = np.zeros((Nscales, len(signal)), dtype=complex)
    f = np.zeros(Nscales)
    coefs[:,:], f = pywt.cwt(signal, np.arange(1, Nscales+1), wavelet_name, sampling_period)
    return coefs, f

### ============== ###
### GET PROPERTIES ###
### ============== ###
'''You can create a wavelet with : wavelet = pywt.Wavelet(name) 
Then you can obtain properties on the wavelet using print(wavelet)
You can also draw it using: y,x = wavelet.wavefun(length(int))
'''

def get_wavelet_properties(x, scales, wavelet_name):
    if wavelet_name=='morl':
        wavelet_support = np.sqrt(2)
    if wavelet_name=='mexh':
        wavelet_support = 1/np.sqrt(2)
    e_folding_time = wavelet_support * scales
    return wavelet_support, e_folding_time


def get_COI_mask(x, scales, wavelet_name):
    wavelet_support,e_folding_time = get_wavelet_properties(x, scales, wavelet_name)
    ls = len(scales)
    lx = len(x)
    COI_mask = np.zeros((ls,lx))
    for i in range(lx):
        for j in range(ls):
            if (scales[j]<=x[i]/wavelet_support) and (scales[j]<=(np.max(x)-x[i])/wavelet_support):
                COI_mask[j,i] = 1
    return COI_mask

def get_COI_lines(x, scales, wavelet_name):
    wavelet_support, e_folding_time = get_wavelet_properties(x,scales,wavelet_name)
    COI_line1 = wavelet_support/x
    COI_line2 = wavelet_support/(np.max(x)-x)
    return COI_line1, COI_line2

### ==== ###
### PLOT ###
### ==== ###
def plot_COI_lines(ax, x, line1, line2):
    ax.plot(x, line1, color='black', linewidth=2)
    ax.plot(x, line2, color='black', linewidth = 2)
    return ax
    
def plot_COI_mask(ax, x, f, coefs, COI):
    masked_coef = np.ma.masked_where(COI!=1, coefs)
    im = ax.pcolormesh(x,f,masked_coef, cmap='jet', shading='auto')
    return ax, im


def plot_wavelet_power_spectrum(x, f, coefs, wavelet_name, ax=None, plotCOI=True, ylog=True):
    if ax==None:
        fig, ax = plt.subplots()
    
    coefs = abs(coefs)**2
    
    im=ax.pcolormesh(x,f,coefs**2, cmap='jet', alpha=0.3, shading='auto')
    
    if plotCOI:
        wavelet_support,_ = get_wavelet_properties(x,1/f,wavelet_name)
        COI_mask = get_COI_mask(x,1/f,wavelet_name)
        COI_line1,COI_line2 = get_COI_lines(x,1/f,wavelet_name)
        ax,im = plot_COI_mask(ax,x,f,coefs,COI_mask)
        plot_COI_lines(ax,x,COI_line1,COI_line2)
    
    if ylog:
        ax.set_yscale('log')
    ax.set(ylim=(np.min(f), np.max(f)))
    # fig.colorbar(im, ax = ax)
    return im, ax


    

def plot_scalogram(x, f, coefs, wavelet_name, amp = True, ax=None, plotCOI=True, ylog=True):
    if ax==None:
        fig, ax = plt.subplots()
    
    if amp:
        coefs = abs(coefs**2)
    
    im=ax.pcolormesh(x,f,coefs, cmap='jet', alpha=0.3, shading='auto')
    print(plotCOI)
    if plotCOI:
        wavelet_support,_ = get_wavelet_properties(x,1/f,wavelet_name)
        COI_mask = get_COI_mask(x,1/f,wavelet_name)
        COI_line1,COI_line2 = get_COI_lines(x,1/f,wavelet_name)
        ax,im = plot_COI_mask(ax,x,f,coefs,COI_mask)
        plot_COI_lines(ax,x,COI_line1,COI_line2)
    
    if ylog:
        ax.set_yscale('log')
    ax.set(ylim=(np.min(f), np.max(f)))
    # fig.colorbar(im, ax = ax)
    return im, ax


    
