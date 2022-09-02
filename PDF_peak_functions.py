#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lhapdf
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import scipy as sp
from scipy import integrate
from scipy import optimize
from matplotlib.pyplot import Rectangle

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
minor = AutoMinorLocator()


# In[2]:


from scipy.optimize import minimize

def valence_PDF_func(x,q, qrk_num, pdfs, imem):
    return pdfs[imem].xfxQ(qrk_num, x, q)- pdfs[imem].xfxQ(-qrk_num, x, q)

def x_peak_and_height_search(q_var, PDF_set, quark_num, imem):
    n_size = len(q_var)
    
    # Set up arrays
    x_peaks = np.zeros(n_size)
    peak_heights = np.zeros(n_size)
    
    
    x_guess = 0.2  # Initial guess

    for i in range(n_size):
        #Make function xf(x,Q^2)
        f_test_foo = lambda x: -valence_PDF_func(x,q_var[i], quark_num, PDF_set,imem) if  (x > 0. and x < 1.) else 0
        
        #Find max (min of -xf) and put into arrays
        res2 = minimize(f_test_foo, x_guess, tol=1e-10)
        x_peaks[i] = float(res2['x'])
        peak_heights[i] = abs(res2['fun']) 

        # New guess for new Q^2 
        x_guess = float(res2['x'])
    
    return x_peaks, peak_heights

def valence_uncertainty_band_symm_err (q_in, n_pts, pdfs, PDF_set, qrk_num, cl):
    q = q_in

    xdV_central = np.zeros(n_pts)
    xdV_uncsymm = np.zeros(n_pts)
    
    for i in range(n_pts):
        x = (1./(n_pts+1))*(i+1)
        xdV_All = np.zeros(PDF_set.size)


        for imem in range(PDF_set.size):
            xdV_All[imem] = pdfs[imem].xfxQ(qrk_num, x, q)- pdfs[imem].xfxQ(-qrk_num, x, q)

        unc = PDF_set.uncertainty(xdV_All, cl)
        xdV_central[i] = unc.central
        xdV_uncsymm [i]= unc.errsymm
    return xdV_central, xdV_uncsymm 


# In[3]:


def x_peak_and_height_unc(qvar, pset, quark_num , cl = 68):
    
    """
    Input
    
    qvar         - array of Q^2 values
    PDF_set       - LHAPDF PDF set
    quark _numer  - 1 for down, 2 for up, etc.
    cl            - Confidence level for errors
    _ _ _ _
    Output
    
    n = size(qvar)
    
    x_peaks       - n x 3 array of x peaks. 0 - central 1 - upper uncert. 2- lower uncert.
    peaks_heights - n x 3 array of peak heights  0 - central 1 - upper uncert. 2- lower uncert.  
    """

    n_size = len(qvar)
    
    PDF_set = pset.mkPDFs()
    
    # Set up arrays
    x_peaks = np.zeros((n_size,3))
    peak_heights = np.zeros((n_size,3))
    
    
    x_guess = 0.2  # Initial guess

    for i in range(n_size):
        
        # For given Q^2 get peaks,heights for each PDF set
        peaks_at_Q2   = np.zeros(pset.size)        
        heights_at_Q2 = np.zeros(pset.size)

        
        for imem in range(pset.size):
        
            f_test_foo = lambda x: -valence_PDF_func(x,qvar[i], quark_num, PDF_set,imem) if  (x > 0. and x < 1.) else 0

            #Find max (min of -xf) and put into arrays
            res2 = minimize(f_test_foo, x_guess, tol=1e-8)
            peaks_at_Q2[imem] = float(res2['x'])
            heights_at_Q2[imem] = abs(res2['fun']) 
            
        # Get central values with upper and lower uncertainties
        
        unc_peaks = pset.uncertainty(peaks_at_Q2,cl)
            
        x_peaks[i,0]   = unc_peaks.central                            # Central
        x_peaks[i,1]   = unc_peaks.central + unc_peaks.errplus        # Upper
        x_peaks[i,2]   = unc_peaks.central - unc_peaks.errminus       # Lower
        
        unc_heights = pset.uncertainty(heights_at_Q2,cl)
            
        peak_heights[i,0]   = unc_heights.central                            # Central
        peak_heights[i,1]   = unc_heights.central + unc_peaks.errplus        # Upper
        peak_heights[i,2]   = unc_heights.central - unc_peaks.errminus       # Lower
             
            
        # New guess for new Q^2 
        x_guess = float(x_peaks[i,0])
        
        #print("x peak = ", x_peaks[i,0])
        #print("peak heights = ", peak_heights[i,1], '/n' )
    
    return x_peaks, peak_heights

def x_peak_and_height_unc_errsymm(qvar, pset, quark_num , cl = 68):
    
    """
    Input
    
    qvar         - array of Q^2 values
    PDF_set       - LHAPDF PDF set
    quark _numer  - 1 for down, 2 for up, etc.
    cl            - Confidence level for errors
    _ _ _ _
    Output
    
    n = size(qvar)
    
    x_peaks       - n x 3 array of x peaks. 0 - central 1 - upper uncert. 2- lower uncert.
    peaks_heights - n x 3 array of peak heights  0 - central 1 - upper uncert. 2- lower uncert.  
    """

    n_size = len(qvar)
    
    PDF_set = pset.mkPDFs()
    
    # Set up arrays
    x_peaks = np.zeros((n_size,4))
    peak_heights = np.zeros((n_size,4))
    
    
    x_guess = 0.2  # Initial guess

    for i in range(n_size):
        
        # For given Q^2 get peaks,heights for each PDF set
        peaks_at_Q2   = np.zeros(pset.size)        
        heights_at_Q2 = np.zeros(pset.size)

        
        for imem in range(pset.size):
        
            f_test_foo = lambda x: -valence_PDF_func(x,qvar[i], quark_num, PDF_set,imem) if  (x > 0. and x < 1.) else 0

            #Find max (min of -xf) and put into arrays
            res2 = minimize(f_test_foo, x_guess, tol=1e-8)
            peaks_at_Q2[imem] = float(res2['x'])
            heights_at_Q2[imem] = abs(res2['fun']) 
            
        # Get central values with upper and lower uncertainties
        
        unc_peaks = pset.uncertainty(peaks_at_Q2,cl)
            
        x_peaks[i,0]   = unc_peaks.central                            # Central
        x_peaks[i,1]   = unc_peaks.central + unc_peaks.errsymm        # Upper
        x_peaks[i,2]   = unc_peaks.central - unc_peaks.errsymm        # Lower
        x_peaks[i,3]   = unc_peaks.errsymm                            # Error symmetric
        
        unc_heights = pset.uncertainty(heights_at_Q2,cl)
            
        peak_heights[i,0]   = unc_heights.central                            # Central
        peak_heights[i,1]   = unc_heights.central + unc_peaks.errsymm        # Upper
        peak_heights[i,2]   = unc_heights.central - unc_peaks.errsymm       # Lower
        peak_heights[i,3]   = unc_peaks.errsymm                             #Error symmetric
             
            
        # New guess for new Q^2 
        x_guess = float(x_peaks[i,0])
        
        #print("x peak = ", x_peaks[i,0])
        #print("peak heights = ", peak_heights[i,1], '/n' )
    
    return x_peaks, peak_heights

def valence_uncertainty_band (q_in, n_pts, PDF_set, qrk_num, cl):

    """
    Input: q, no of points, PDF set, quark number, confidence level,

    Output: 3 arrays of central, lower error, upper error for PDF from x=0 to 1 
    with  n_pts

    """
    q = q_in

    #Set up arrays 

    xdV_central = np.zeros(n_pts)
    xdV_upper   = np.zeros(n_pts)
    xdV_lower   = np.zeros(n_pts)
    
    pdfs = PDF_set.mkPDFs()
    
    for i in range(n_pts):
        x = (1./n_pts)*i
        xdV_All = np.zeros(PDF_set.size)


        for imem in range(PDF_set.size):
            xdV_All[imem] = pdfs[imem].xfxQ(qrk_num, x, q)- pdfs[imem].xfxQ(-qrk_num, x, q)

        unc = PDF_set.uncertainty(xdV_All, cl)
        xdV_central[i] = unc.central
        xdV_upper[i]  = unc.central + unc.errplus_pdf
        xdV_lower[i]  = unc.central - unc.errminus_pdf
    return xdV_central, xdV_lower, xdV_upper

def PDF_uncertainty_band (q_in, n_pts, PDF_set, qrk_num, cl):

    """
    Input: q, no of points, PDF set, quark number, confidence level,

    Output: 3 arrays of central, lower error, upper error for PDF from x=0 to 1 
    with  n_pts (Just quark PDF, not valence PDF)

    """
    q = q_in

    #Set up arrays 

    xdV_central = np.zeros(n_pts)
    xdV_upper   = np.zeros(n_pts)
    xdV_lower   = np.zeros(n_pts)
    
    pdfs = PDF_set.mkPDFs()
    
    for i in range(n_pts):
        x = (1./n_pts)*i
        xdV_All = np.zeros(PDF_set.size)


        for imem in range(PDF_set.size):
            xdV_All[imem] = pdfs[imem].xfxQ(qrk_num, x, q)

        unc = PDF_set.uncertainty(xdV_All, cl)
        xdV_central[i] = unc.central
        xdV_upper[i]  = unc.central + unc.errplus_pdf
        xdV_lower[i]  = unc.central - unc.errminus_pdf
    return xdV_central, xdV_lower, xdV_upper


def PDF_at_Q_dataframe(q, n_pts, pset, cl= 68):
    """
    Creates dataframe of a PDF at particular q value 
    with central and upper/lower uncertainty values
    for both down and up valence
    
    Input: q, no. of points, pset, confidence level
    
    Output: Pandas dataframe with these columns:
    xp  hu_min  hu_central  hu_max  hd_min  hd_central  hd_max
    """
    
    # Create central values and uncertainties
    xdv_central , xdv_lower , xdv_upper = peak.valence_uncertainty_band (q, n_pts, pset , 1, 68)
    xuv_central , xuv_lower , xuv_upper = peak.valence_uncertainty_band (q, n_pts, pset , 2, 68)
    
    # Create x values
    xp = xp = np.array( [(1./n_pts)*i  for i in range(n_pts)] ) 
    
    # Make a Pandas dataframe and fill it up
    dataset = pd.DataFrame()

    dataset['xp'] =xp[1:]

    dataset['hu_min'] = xuv_lower[1:]
    dataset['hu_central'] = xuv_central[1:]
    dataset['hu_max'] = xuv_upper[1:]

    dataset['hd_min'] = xdv_lower[1:]
    dataset['hd_central'] = xdv_central[1:]
    dataset['hd_max'] = xdv_upper[1:]

    return dataset     

# In[12]:


def combined_param_with_err(param,param_err):
    n = len(param)
    combined_list = []
    
    for i in range(n):
        combined_list.append(param[i])
        combined_list.append(param_err[i])
       
    return combined_list        


# # Fitting tools and $\alpha_S$

# In[1]:


from scipy.optimize import curve_fit

def fit_special_curve_parameters(gen_func, xdata, ydata, p0=None, sigma =None):
    popt, pcov = curve_fit(gen_func, xdata,ydata, p0=None,sigma=None)
    unc = np.sqrt(np.diag(pcov))
    return popt, unc      

def alpha_S(Q2, pdfs_set):
    return pdfs_set[0].alphasQ2(Q2)

def exponential_func(x,a,b):
    return np.multiply(a,np.exp(np.multiply(b,x)))

def linear_func(x,a,b):
    return np.add(np.multiply(a,x), b)    


# In[ ]:

def PDF_at_Q_dataframe(q, n_pts, pset, cl= 68):
    """
    Creates dataframe of a PDF at particular q value 
    with central and upper/lower uncertainty values
    for both down and up valence
    
    Input: q, no. of points, pset, confidence level
    
    Output: Pandas dataframe with these columns:
    xp  hu_min  hu_central  hu_max  hd_min  hd_central  hd_max
    """
    
    # Create central values and uncertainties
    xdv_central , xdv_lower , xdv_upper = peak.valence_uncertainty_band (q, n_pts, pset , 1, 68)
    xuv_central , xuv_lower , xuv_upper = peak.valence_uncertainty_band (q, n_pts, pset , 2, 68)
    
    # Create x values
    xp = xp = np.array( [(1./n_pts)*i  for i in range(n_pts)] ) 
    
    # Make a Pandas dataframe and fill it up
    dataset = pd.DataFrame()

    dataset['xp'] =xp[1:]

    dataset['hu_min'] = xuv_lower[1:]
    dataset['hu_central'] = xuv_central[1:]
    dataset['hu_max'] = xuv_upper[1:]

    dataset['hd_min'] = xdv_lower[1:]
    dataset['hd_central'] = xdv_central[1:]
    dataset['hd_max'] = xdv_upper[1:]

    return dataset     

"""
xf is the model PDF. It is a function of Bjorken x. Parameters are residual RMS and mass ratio of residual to nucleon

xf is obtained from integrating the integrand from xb to 1

"""

def integrand(xv,xb, rms ,m_ratio):
    A = (2./3.)*(0.938272**2)*(rms/0.197327)**2
    B = 1-m_ratio
    return xb*np.exp(-A*(xv-B)**2)*(xv-xb)**3/xv**3

def xf(xb, rms ,m_ratio):
    integral = quad(integrand, xb, 1, args=(xb, rms, m_ratio))
    return integral[0]



