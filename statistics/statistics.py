# =============================================================================
#  Authors
# =============================================================================
# Name: Olivier PANICO
# corresponding author: olivier.panico@free.fr

# =============================================================================
#  Imports
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

def get_skewness(A):
    return skew(A)

def get_kurtosis(A):
    return kurtosis(A)

def get_stats(A):
    Askew = get_skewness(A)
    Akurt = get_kurtosis(A)
    return Askew, Akurt



def get_pdf(data,n_bins=20):
    
    # Create a histogram of the data
    hist, bins = np.histogram(data, bins=n_bins, density=False)

    # Calculate the bin widths
    bin_widths = bins[2] - bins[1]

    # Normalize the histogram to obtain the PDF
    pdf_ = hist / (bin_widths.sum() * len(data))
    return bins[1:], pdf_

def eval_self_similar_pdf(data_box,n_bins=30):
    #data_pdf = (data_box - np.mean(data_box))/np.std(data_box)
    bins, pdf_ = pdf(data_box,n_bins)
    return (bins - np.mean(data_box))/np.std(data_box) , pdf_*np.std(data_box)
