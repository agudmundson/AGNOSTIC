
__author__  = 'Aaron Gudmundson'
__email__   = 'agudmund@uci.edu'
__date__    = '2020/12/05'
__version__ = '10.0.0'
__status__  = 'beta'

from scipy.optimize import curve_fit, minimize                                              # Optimization
import matplotlib.patches as mpatches                                                       # Figure Legends
import matplotlib.pyplot as plt                                                             # Plotting
import scipy.signal as sig 																	# Signal Processing
import scipy.io as sio                                                                      # SciPy (Local Import for Non-Local Call)
import pandas as pd                                                                         # DataFrames
import numpy as np                                                                          # Arrays
import time as t0                                                                           # Determine Run Time
import subprocess
import struct                                                                               # Reading Binary
import copy
import glob                                                                                 # Bash-like File Reading
import sys                                                                                  # Interaction w/System
import os                                                                                   # Interaction w/Operating System

np.set_printoptions(threshold=np.inf, precision=3, linewidth=300, suppress=False)           # Terminal Numpy Settings
np.seterr(divide='ignore', invalid='ignore')                                                # Terminal Numpy Warnings

def get_noise(samp, nreal, nimag):
	noise     = np.vectorize(complex)(nreal, nimag)
	noise     = np.real(np.fft.fftshift(np.fft.fft(noise)))
	noise     = np.std(noise)
	snr       = samp/noise
	return snr

def opt_noise(theta, samp, nreal, nimag, target, N):
	noise     = np.vectorize(complex)(nreal, nimag)
	noise    *= theta[0]
	noise     = np.real(np.fft.fftshift(np.fft.fft(noise, n=N)))
	noise     = np.std(noise[:1000])
	return np.sqrt( np.mean( ((samp/noise) - target)**2 ) )

def opt_fwhm(theta, point, real, imag, time, zppm, axis, target, n, phs_1):
	spec  = np.vectorize(complex)(real, imag) 												# Complex Signal
	spec *= np.exp(-1 * time**2 * np.pi *  theta[0]**2) 									# Gaussian Decay
	spec  = np.fft.fftshift(np.fft.fft(spec[:n], n=axis.shape[0])) 							# Fourier Transform
	spec *= np.exp(1j * (zppm - 4.70) * phs_1 * np.pi/180) 									# 1st Order Phase Correction
	spec  = np.real(spec) 																	# Real Component 

	spec /= np.max(spec) 																	# Max Value
	spec -= np.max(spec)/2 																	# Half Max Value

	left  = np.where(np.abs(spec[:point]) == np.min(np.abs(spec[:point])))[0] 				# Left  Point at Half Max
	rght  = np.where(np.abs(spec[point:]) == np.min(np.abs(spec[point:])))[0]  				# Right Point at Half Max
	rght += point

	fwhm  = axis[int(rght)] - axis[int(left)] 					 							# Full-Width at Half Maximum
	return np.sqrt( np.mean((fwhm - target)**2) ) 											# Error 

def _fwhm(point, in_spec, axis=None):
    
    spec_  = copy.deepcopy(in_spec)
    spec_ /= np.max(spec_)
    famp   = np.max(spec_)
    hamp   = famp/2
    
    spec_ -= hamp
    left   = np.where(np.abs(spec_[:point]) == np.min(np.abs(spec_[:point])))[0]
    rght   = np.where(np.abs(spec_[point:]) == np.min(np.abs(spec_[point:])))[0] 
    rght  += point
    
    if axis is not None:
        return (axis[int(rght)] - axis[int(left)], rght, left)
    else:
        return (rght-left, right, left)

def clinical_concentrations(std_dev='2.0'):

	## These Values were derived from Gudmundson 2022 (in prep).
	#  Values are a linear relationship between healthy and clinical. 
	#  Values are all 2 standard deviations away.
	#  

	clin     = {
				'Seiz': {'Cre' : np.array([0.918, 1.012]), 									# Seizure       Used Cre
						 'PCr' : np.array([0.918, 1.012]), 									# Seizure       Used Cre
						 'GPC' : np.array([0.731, 1.147]), 									# Seizure       Used tCho
						 'PCh' : np.array([0.731, 1.147]), 									# Seizure       Used tCho
						 'GABA': np.array([0.930, 1.173]),   								# Seizure       (Fixed Effect)
						 'Glu' : np.array([0.787, 1.247]),   								# Seizure       Used Glx
						 'Gln' : np.array([0.787, 1.247]),   								# Seizure       Used Glx
						 'GSH' : np.array([0.887, 1.243]),   								# Seizure 
						 'Myo' : np.array([0.802, 1.134]),   								# Seizure 
						 'NAA' : np.array([0.751, 1.002]),   								# Seizure       Used tNAA 
						 'NAG' : np.array([0.751, 1.002])},		 							# Seizure       Used tNAA 
				'Str' : {'Cre' : np.array([0.684, 1.146]),  								# Stroke        Used Cre 
						 'PCr' : np.array([0.684, 1.146]),  								# Stroke        Used Cre
						 'GPC' : np.array([0.855, 1.527]),  								# Stroke        Used GPC 
						 'PCh' : np.array([0.855, 1.527]),  								# Stroke        Used GPC 
						 'Glu' : np.array([0.874, 1.140]),   								# Stroke        Used Glx
						 'Gln' : np.array([0.874, 1.140]),   								# Stroke        Used Glx
						 'Lac' : np.array([1.000, 6.922]),   								# Stroke        Modified Lower Range (Was -.706 to 6.922)
						 'Myo' : np.array([0.827, 1.265]),   								# Stroke
						 'NAA' : np.array([0.727, 1.074]),   								# Stroke        Used NAA 	
						 'NAG' : np.array([0.727, 1.074])},  								# Stroke        Used NAA 
				'TBI' : {'Asp' : np.array([0.785, 0.910]),  								# TBI
						 'Cre' : np.array([0.814, 1.162]),  								# TBI           Used tCr  
						 'PCr' : np.array([0.814, 1.162]),  								# TBI           Used tCr  
						 'GPC' : np.array([0.930, 1.057]),  								# TBI           Used GPC 
						 'PCh' : np.array([0.930, 1.057]),  								# TBI           Used GPC 
						 'GABA': np.array([0.860, 0.984]),   								# TBI
						 'Glu' : np.array([0.824, 1.214]),   								# TBI           Used Glx
						 'Gln' : np.array([0.824, 1.214]),   								# TBI           Used Glx
						 'Myo' : np.array([0.737, 1.315]),   								# TBI     
						 'NAA' : np.array([0.795, 1.001]),   								# TBI           Used tNAA 	
						 'NAG' : np.array([0.795, 1.001])},  								# TBI           Used tNAA 
				'D1'  : {'Asp' : np.array([0.895, 1.496]),  								# Diabetes1
						 'Cre' : np.array([0.977, 1.039]),  								# Diabetes1     Used tCr  
						 'PCr' : np.array([0.977, 1.039]),  								# Diabetes1     Used tCr  
						 'GPC' : np.array([1.034, 1.140]),  								# Diabetes1     Used GPC 
						 'PCh' : np.array([1.034, 1.140]),  								# Diabetes1     Used GPC 
						 'Glu' : np.array([0.895, 1.216]),   								# Diabetes1     Used Glx
						 'Gln' : np.array([0.956, 1.353]),   								# Diabetes1     Used Glx
						 'GSH' : np.array([0.872, 1.435]),   								# Diabetes1   
						 'Myo' : np.array([0.893, 1.092]),   								# Diabetes1   
						 'NAA' : np.array([0.947, 1.008]),   								# Diabetes1     Used tNAA 	
						 'NAG' : np.array([0.947, 1.008]),   								# Diabetes1     Used tNAA 	
						 'Scy' : np.array([0.501, 0.992]),   								# Diabetes1
						 'Tau' : np.array([0.754, 1.322])},  								# Diabetes1
				'Canc': {'Cre' : np.array([0.256, 1.340]),  								# Cancer        Used tCr  
						 'PCr' : np.array([0.256, 1.340]),  								# Cancer        Used tCr  
						 'GPC' : np.array([1.139, 1.949]),  								# Cancer        Used tCh 
						 'PCh' : np.array([1.139, 1.949]),  								# Cancer        Used tCh 
						 'Glu' : np.array([0.780, 1.320]),   								# Cancer        Used Glx
						 'Gln' : np.array([0.780, 1.320]),   								# Cancer        Used Glx
						 'Lac' : np.array([1.000, 9.999]),   								# Cancer        Modified - Lactate did not come through in Analysis
						 'Myo' : np.array([0.829, 1.519]),   								# Cancer      
						 'NAA' : np.array([0.509, 0.956]),   								# Cancer        Used NAA 	
						 'NAG' : np.array([0.509, 0.956])},   								# Cancer        Used NAA 
				'Pain': {'GPC' : np.array([0.943, 1.285]),  								# Pain          Used tCh 
						 'PCh' : np.array([0.943, 1.285]),  								# Pain          Used tCh 
						 'GABA': np.array([0.896, 1.168]),   								# Pain
						 'Glu' : np.array([0.790, 1.121]),   								# Pain          Used Glx
						 'Gln' : np.array([0.790, 1.121]),   								# Pain          Used Glx
						 'Myo' : np.array([0.942, 1.049]),   								# Pain        
						 'NAA' : np.array([0.775, 1.280]),   								# Pain          Used tNAA 	
						 'NAG' : np.array([0.775, 1.280])},   								# Pain          Used tNAA
				'Mgrn': {'Asp' : np.array([0.434, 1.409]),  								# Migraine
						 'Cre' : np.array([0.921, 1.011]),  								# Migraine      Used tCr (Fixed Effect)
						 'PCr' : np.array([0.921, 1.011]),  								# Migraine      Used tCr  
						 'GPC' : np.array([0.959, 1.137]),  								# Migraine      Used GPC 
						 'PCh' : np.array([0.959, 1.137]),  								# Migraine      Used GPC 
						 'Glu' : np.array([0.841, 1.119]),   								# Migraine      Used Glx
						 'Gln' : np.array([0.841, 1.119]),   								# Migraine      Used Glx
						 'Myo' : np.array([0.866, 1.032]),   								# Migraine      (Fixed Effect)
						 'NAA' : np.array([0.755, 1.067]),   								# Migraine      Used tNAA 	
						 'NAG' : np.array([0.755, 1.067])},  								# Migraine      Used tNAA  
				'Fib' : {'Cre' : np.array([0.760, 1.429]),  								# Fibromyalgia  Used Cre  
						 'PCr' : np.array([0.760, 1.429]),  								# Fibromyalgia  Used Cre  
						 'GPC' : np.array([0.840, 1.236]),  								# Fibromyalgia  Used tCh 
						 'PCh' : np.array([0.840, 1.236]),  								# Fibromyalgia  Used tCh 
						 'GABA': np.array([0.724, 0.937]),   								# Fibromyalgia 
						 'Glu' : np.array([1.005, 1.104]),   								# Fibromyalgia
						 'Gln' : np.array([0.711, 1.107]),   								# Fibromyalgia
						 'Myo' : np.array([0.844, 1.232]),   								# Fibromyalgia      
						 'NAA' : np.array([0.847, 1.061]),   								# Fibromyalgia  Used tNAA 	
						 'NAG' : np.array([0.847, 1.061])},   								# Fibromyalgia  Used tNAA 
				'PTSD': {'Cre' : np.array([0.940, 1.235]),  								# PTSD          Used Cre  
						 'PCr' : np.array([0.940, 1.235]),  								# PTSD          Used Cre  
						 'GPC' : np.array([0.843, 1.283]),  								# PTSD          Used tCh 
						 'PCh' : np.array([0.843, 1.283]),  								# PTSD          Used tCh 
						 'GABA': np.array([0.982, 1.059]),   								# PTSD         
						 'Glu' : np.array([0.892, 1.134]),   								# PTSD        
						 'Gln' : np.array([0.892, 1.134]),   								# PTSD        
						 'Myo' : np.array([0.939, 1.198]),   								# PTSD              
						 'NAA' : np.array([0.969, 1.156]),   								# PTSD          Used NAA 	
						 'NAG' : np.array([0.969, 1.156])},   								# PTSD          Used NAA 
				'OCD' : {'Cre' : np.array([0.890, 1.320]),  								# OCD           Used tCr  
						 'PCr' : np.array([0.890, 1.320]),  								# OCD           Used tCr  
						 'GPC' : np.array([0.784, 1.223]),  								# OCD           Used tCh 
						 'PCh' : np.array([0.784, 1.223]),  								# OCD           Used tCh 
						 'Glu' : np.array([0.868, 1.243]),   								# OCD           Used Glx
						 'Gln' : np.array([0.868, 1.243]),   								# OCD           Used Glx
						 'Myo' : np.array([0.743, 1.437]),   								# OCD               
						 'NAA' : np.array([0.846, 1.100]),   								# OCD           Used tNAA 	
						 'NAG' : np.array([0.846, 1.100])},   								# OCD           Used tNAA 
				'Depr': {'Cre' : np.array([0.938, 1.021]),  								# Depression    Used tCr  
						 'PCr' : np.array([0.938, 1.021]),  								# Depression    Used tCr  
						 'GPC' : np.array([0.741, 1.158]),  								# Depression    Used tCh 
						 'PCh' : np.array([0.741, 1.158]),  								# Depression    Used tCh 
						 'GABA': np.array([0.769, 1.400]),   								# Depression   
						 'Glu' : np.array([0.872, 1.119]),   								# Depression  
						 'Gln' : np.array([0.894, 1.177]),   								# Depression  
						 'GSH' : np.array([0.822, 1.082]),   								# Depression  
						 'Myo' : np.array([0.874, 1.239]),   								# Depression        
						 'NAA' : np.array([0.864, 1.080]),   								# Depression    Used tNAA 	
						 'NAG' : np.array([0.864, 1.080])},   								# Depression    Used tNAA 
				'Adc' : {'Cre' : np.array([0.775, 1.161]),  								# Addiction     Used tCr  
						 'PCr' : np.array([0.775, 1.161]),  								# Addiction     Used tCr  
						 'GPC' : np.array([0.788, 1.202]),  								# Addiction     Used tCh 
						 'PCh' : np.array([0.788, 1.202]),  								# Addiction     Used tCh 
						 'GABA': np.array([0.669, 1.289]),   								# Addiction     
						 'Glu' : np.array([0.807, 1.229]),   								# Addiction     Used Glu
						 'Gln' : np.array([0.807, 1.229]),   								# Addiction     Used Glu
						 'Gly' : np.array([0.969, 1.335]),   								# Addiction
						 'GSH' : np.array([0.935, 1.442]),   								# Addiction       
						 'Myo' : np.array([0.820, 1.135]),   								# Addiction             
						 'NAA' : np.array([0.761, 1.195]),   								# Addiction     Used tNAA 	
						 'NAG' : np.array([0.761, 1.195])},   								# Addiction     Used tNAA 
				'Schz': {'Cre' : np.array([0.948, 1.045]),  								# Schizophrenia Used tCr  
						 'PCr' : np.array([0.948, 1.045]),  								# Schizophrenia Used tCr  
						 'GPC' : np.array([0.946, 1.157]),  								# Schizophrenia Used tCh 
						 'PCh' : np.array([0.946, 1.157]),  								# Schizophrenia Used tCh 
						 'GABA': np.array([0.732, 1.261]),   								# Schizophrenia 
						 'Glu' : np.array([0.857, 1.164]),   								# Schizophrenia Used Glx
						 'Gln' : np.array([0.857, 1.164]),   								# Schizophrenia Used Glx
						 'GSH' : np.array([0.868, 1.038]),   								# Schizophrenia   
						 'Myo' : np.array([0.806, 1.239]),   								# Schizophrenia         
						 'NAA' : np.array([0.910, 1.103]),   								# Schizophrenia Used tNAA 	
						 'NAG' : np.array([0.910, 1.103])},   								# Schizophrenia Used tNAA 			
				'Psy' : {'Cre' : np.array([0.983, 1.059]),  								# Psychosis     Used tCr  
						 'PCr' : np.array([0.983, 1.059]),  								# Psychosis     Used tCr  
						 'GPC' : np.array([0.892, 1.127]),  								# Psychosis     Used tCh 
						 'PCh' : np.array([0.892, 1.127]),  								# Psychosis     Used tCh 
						 'GABA': np.array([0.725, 1.176]),   								# Psychosis     
						 'Glu' : np.array([0.813, 1.172]),   								# Psychosis     Used Glx
						 'Gln' : np.array([0.813, 1.172]),   								# Psychosis     Used Glx
						 'Gly' : np.array([1.131, 1.423]),   								# Psychosis     Used Glx
						 'GSH' : np.array([0.917, 1.034]),   								# Psychosis       
						 'Myo' : np.array([0.892, 1.090]),   								# Psychosis            
						 'NAA' : np.array([0.910, 1.048]),   								# Psychosis     Used tNAA 	
						 'NAG' : np.array([0.910, 1.048])},   								# Psychosis     Used tNAA 		
				'Pers': {'Cre' : np.array([0.961, 1.110]),  								# Personality   Used tCr  
						 'PCr' : np.array([0.961, 1.110]),  								# Personality   Used tCr  
						 'GPC' : np.array([0.925, 1.007]),  								# Personality   Used tCh 
						 'PCh' : np.array([0.925, 1.007]),  								# Personality   Used tCh 
						 'Glu' : np.array([0.949, 1.207]),   								# Personality   Used Glx
						 'Gln' : np.array([0.949, 1.207]),   								# Personality   Used Glx
						 'GSH' : np.array([0.917, 1.034]),   								# Personality      
						 'Myo' : np.array([0.989, 1.081]),   								# Personality               
						 'NAA' : np.array([0.880, 0.997]),   								# Personality   Used NAA 	
						 'NAG' : np.array([0.880, 0.997])},   								# Personality   Used NAA 									 			 
				'Bip' : {'Cre' : np.array([0.900, 1.061]),  								# Bipolar       Used tCr  
						 'PCr' : np.array([0.900, 1.061]),  								# Bipolar       Used tCr  
						 'GPC' : np.array([0.854, 1.269]),  								# Bipolar       Used tCh 
						 'PCh' : np.array([0.854, 1.269]),  								# Bipolar       Used tCh 
						 'Glu' : np.array([0.907, 1.115]),   								# Bipolar       Used Glx
						 'Gln' : np.array([0.907, 1.115]),   								# Bipolar       Used Glx
						 'GSH' : np.array([0.957, 1.150]),   								# Bipolar             
						 'Myo' : np.array([0.812, 1.209]),   								# Bipolar                      
						 'NAA' : np.array([0.863, 1.109]),   								# Bipolar       Used tNAA 	
						 'NAG' : np.array([0.863, 1.109])},   								# Bipolar       Used tNAA 	
				'MS'  : {'GPC' : np.array([0.880, 1.077]),  								# M.Sclerosis   Used GPC 
						 'PCh' : np.array([0.880, 1.077]),  								# M.Sclerosis   Used GPC 
						 'GABA': np.array([0.851, 1.017]),   								# M.Sclerosis
						 'Glu' : np.array([0.887, 1.030]),   								# M.Sclerosis   Used Glx
						 'Gln' : np.array([0.887, 1.030]),   								# M.Sclerosis   Used Glx
						 'GSH' : np.array([0.844, 1.069]),   								# M.Sclerosis         
						 'Myo' : np.array([0.892, 1.078]),   								# M.Sclerosis   (Fixed)                
						 'NAA' : np.array([0.924, 1.044]),   								# M.Sclerosis   Used tNAA 	
						 'NAG' : np.array([0.924, 1.044])},   								# M.Sclerosis   Used tNAA 	
				'PD'  : {'Cre' : np.array([0.850, 1.100]),  								# Parkinson's   Used tCr  
						 'PCr' : np.array([0.850, 1.100]),  								# Parkinson's   Used tCr  
						 'GPC' : np.array([0.780, 1.201]),  								# Parkinson's   Used tCh
						 'PCh' : np.array([0.780, 1.201]),  								# Parkinson's   Used tCh
						 'GABA': np.array([0.679, 1.390]),  								# Parkinson's
						 'Glu' : np.array([0.887, 1.224]),   								# Parkinson's   Used Glx
						 'Gln' : np.array([0.887, 1.224]),   								# Parkinson's   Used Glx
						 'Myo' : np.array([0.810, 1.190]),   								# Parkinson's                  
						 'NAA' : np.array([0.756, 1.240]),   								# Parkinson's   Used tNAA 	
						 'NAG' : np.array([0.756, 1.240])},   								# Parkinson's   Used tNAA 	
				'ETrm': {'Cre' : np.array([0.924, 1.053]),  								# Ess.Tremor    Used tCr  
						 'PCr' : np.array([0.924, 1.053]),  								# Ess.Tremor    Used tCr  
						 'GPC' : np.array([0.851, 1.044]),  								# Ess.Tremor    Used tCh 
						 'PCh' : np.array([0.851, 1.044]),  								# Ess.Tremor    Used tCh 
						 'GABA': np.array([0.802, 1.218]),  								# Ess.Tremor    
						 'Glu' : np.array([1.050, 1.434]),   								# Ess.Tremor    Used Glx (Fixed)
						 'Gln' : np.array([1.050, 1.434]),   								# Ess.Tremor    Used Glx (Fixed)
						 'NAA' : np.array([0.919, 1.136]),   								# Ess.Tremor    Used tNAA (Fixed)
						 'NAG' : np.array([0.919, 1.136])},   								# Ess.Tremor    Used tNAA (Fixed)	
				'Dem' : {'Asc' : np.array([1.132, 1.231]),  								# Dementia
						 'Asp' : np.array([1.028, 1.168]),  								# Dementia 
						 'Cre' : np.array([1.010, 1.028]),  								# Dementia      Used tCr  
						 'PCr' : np.array([1.010, 1.028]),  								# Dementia      Used tCr  
						 'GPC' : np.array([0.850, 1.150]),  								# Dementia      Value Between GPC and TCh
						 'PCh' : np.array([0.850, 1.150]),  								# Dementia      Value Between GPC and TCh
						 'GABA': np.array([0.513, 1.183]),  								# Dementia      
						 'Glu' : np.array([0.771, 1.139]),   								# Dementia      Used Glx
						 'Gln' : np.array([0.955, 1.172]),   								# Dementia      Used Glx
						 'Myo' : np.array([0.801, 1.397]),   								# Dementia                      
						 'NAA' : np.array([0.723, 1.038]),   								# Dementia      Used tNAA 	
						 'NAG' : np.array([0.723, 1.038]),   								# Dementia      Used tNAA 	
						 'PhE' : np.array([0.699, 1.075]),   								# Dementia
						 'Scy' : np.array([0.476, 1.312]),   								# Dementia	    (Fixed)
						 'Tau' : np.array([0.882, 1.013])},   								# Dementia	
				'E4'  : {'Asp' : np.array([1.028, 1.168]),  								# APOE4 
						 'GPC' : np.array([0.965, 1.019]),  								# Dementia      Used tCh
						 'PCh' : np.array([0.965, 1.019]),  								# Dementia      Used tCh
						 'GABA': np.array([0.513, 1.183]),  								# Dementia      
						 'Glc' : np.array([0.971, 1.028]),   								# Dementia
						 'Glu' : np.array([0.836, 1.126]),   								# Dementia  
						 'Gln' : np.array([0.909, 1.232]),   								# Dementia
						 'GSH' : np.array([0.834, 1.103]),   								# Dementia                      
						 'Myo' : np.array([0.959, 1.092]),   								# Dementia                      
						 'NAA' : np.array([0.895, 1.063]),   								# Dementia      Used NAA 	
						 'NAG' : np.array([0.895, 1.063])},   								# Dementia	    Used NAA 
				}

	return clin

def concentrations(dataset, std_dev='2.5'):
	
	if   dataset == 'Gudmundson_Healthy':
	 	conc = {# Metab            Low Range Hgh Range 	
	 			'Ace'   : np.array([   0.00 ,   0.00]), 									# Acetate
				'Ala'   : np.array([   0.47 ,   0.77]), 									# Alanine - Fit Value
				'Asc'   : np.array([   0.36 ,   1.53]), 									# Ascorbate
				'Asp'   : np.array([   0.00 ,   4.66]),  									# Aspartate
				'ATP'   : np.array([   0.00 ,   0.00]),  									# Adenosine Triphosphate
				'bHB'   : np.array([   0.00 ,   0.00]),  									# Beta-HydroxyButyrate
				'bHG'   : np.array([   0.00 ,   0.00]),  									# Beta-HydroxyGlutarate
				'Cit'   : np.array([   0.00 ,   0.00]),  									# Citrate
				'Cre'   : np.array([   1.41 ,  10.50]), 									# Creatine
				'Cys'   : np.array([   0.00 ,   0.00]), 									# Cysteine
				'ETA'   : np.array([   0.00 ,   0.00]), 									# EthanolAmine
				'EtOH'  : np.array([   0.00 ,   0.00]), 									# EthylAlcohol
				'GABA'  : np.array([   0.52 ,   1.99]), 									# GABA Near
				'GABG'  : np.array([   0.52 ,   1.99]), 									# GABA Govindaraju
				'GlcA'  : np.array([   0.94 ,   1.53]), 									# GlucoseA - Fit Value
				'GlcB'  : np.array([   0.94 ,   1.53]), 									# GlucoseB - Fit Value
				'Gln'   : np.array([   0.26 ,   3.64]), 									# Glutamine
				'Glu'   : np.array([   3.88 ,  13.17]), 									# Glutamate
				'GPC'   : np.array([   0.05 ,   5.00]), 									# GlyceroPhosphoCholine (Actual Value was 3.14; Extending to generalize better)
				'GSH'   : np.array([   0.16 ,   2.41]), 									# Glutathione
				'Glyn'  : np.array([   0.94 ,   1.53]), 									# Glycine - Fit Value
				'Glyc'  : np.array([   0.00 ,   0.00]), 									# Glycerol
				'H2O'   : np.array([   0.00 ,   0.00]), 									# H2O
				'Hist'  : np.array([   0.00 ,   0.00]), 									# Histamine
				'Hisd'  : np.array([   0.00 ,   0.00]), 									# Histadine
				'HCr'   : np.array([   0.00 ,   0.00]), 									# HomeCarnosine
				'Lac'   : np.array([   0.00 ,   1.44]), 									# Lactate

				'MM_092': np.array([   1.00 ,  30.00]), 									# MacroMolecule 0.92
				'MM_121': np.array([   1.00 ,   8.00]), 									# MacroMolecule 1.21
				'MM_139': np.array([   1.00 ,  35.00]), 									# MacroMolecule 1.39
				'MM_167': np.array([   1.00 ,  15.00]), 									# MacroMolecule 1.67
				'MM_204': np.array([   1.00 ,  35.00]), 									# MacroMolecule 2.04
				'MM_226': np.array([   1.00 ,  20.00]), 									# MacroMolecule 2.26
				'MM_256': np.array([   1.00 ,   5.00]), 									# MacroMolecule 2.56
				'MM_270': np.array([   1.00 ,   7.00]), 									# MacroMolecule 2.70
				'MM_299': np.array([   1.00 ,  10.00]), 									# MacroMolecule 2.99
				'MM_321': np.array([   1.00 ,   7.00]), 									# MacroMolecule 3.21
				'MM_362': np.array([   1.00 ,   5.00]),										# MacroMolecule 3.62
				'MM_375': np.array([   1.00 ,  10.00]),										# MacroMolecule 3.75
				'MM_386': np.array([   1.00 ,   4.00]),										# MacroMolecule 3.86
				'MM_403': np.array([   1.00 ,   7.00]),										# MacroMolecule 4.03

				'Myo'   : np.array([   2.08 ,  14.00]), 									# MyoInositol (Actual Value was 9.98; Extending to generalize better)
				'NAA'   : np.array([   5.38 ,  18.00]), 									# N-AcetylAspartate (Actual Value was 13.61; Extending to generalize better)
				'NAG'   : np.array([   0.26 ,   2.26]), 									# N-AcetlyAsparate-Glutamate (Actual Value was 1.74; Extending to generalize better)
				'PCh'   : np.array([   0.01 ,   2.00]), 									# PhosphoCholine - Fit Value
				'PCr'   : np.array([   3.38 ,   6.44]), 									# PhosphoCreatine - Fit Value
				'PhE'   : np.array([   1.41 ,   2.30]), 									# PhosphoEthanolAmine - Fit Value
				'PEtOH' : np.array([   0.00 ,   0.00]), 									# PhosphoEthylAlcohol
				'Phy'   : np.array([   0.00 ,   0.00]), 									# PhenylAlanine
				'Scy'   : np.array([   0.00 ,   0.39]), 									# ScylloInositol
				'Ser'   : np.array([   0.00 ,   0.00]), 									# Serine
				'Tau'   : np.array([   0.00 ,   2.89]), 									# Taurine
				'Thr'   : np.array([   0.00 ,   0.00]), 									# Threonine
				'Try'   : np.array([   0.00 ,   0.00]), 									# Tryptophan
				'Tyr'   : np.array([   0.00 ,   0.00]), 									# Tyrosine
				'Val'   : np.array([   0.00 ,   0.00]), 									# Valine
			   }

	elif   dataset == 'Fit' and std_dev == '2.5':
	 	conc = {# Metab            Low Range Hgh Range 	
	 			'Ace'   : np.array([   0.00 ,   0.00]), 									# Acetate
				'Ala'   : np.array([   0.47 ,   0.77]), 									# Alanine
				'Asc'   : np.array([   0.94 ,   1.53]), 									# Ascorbate
				'Asp'   : np.array([   1.88 ,   3.06]),  									# Aspartate
				'ATP'   : np.array([   0.00 ,   0.00]),  									# Adenosine Triphosphate
				'bHB'   : np.array([   0.00 ,   0.00]),  									# Beta-HydroxyButyrate
				'bHG'   : np.array([   0.00 ,   0.00]),  									# Beta-HydroxyGlutarate
				'Cit'   : np.array([   0.00 ,   0.00]),  									# Citrate
				'Cre'   : np.array([   3.38 ,   6.44]), 									# Creatine
				'Cys'   : np.array([   0.00 ,   0.00]), 									# Cysteine
				'ETA'   : np.array([   0.00 ,   0.00]), 									# EthanolAmine
				'EtOH'  : np.array([   0.00 ,   0.00]), 									# EthylAlcohol
				'GABA'  : np.array([   0.94 ,   1.53]), 									# GABA Near
				'GABG'  : np.array([   0.94 ,   1.53]), 									# GABA Govindaraju
				'GlcA'  : np.array([   0.94 ,   1.53]), 									# GlucoseA
				'GlcB'  : np.array([   0.94 ,   1.53]), 									# GlucoseB
				'Gln'   : np.array([   1.89 ,   4.84]), 									# Glutamine
				'Glu'   : np.array([   8.66 ,  16.25]), 									# Glutamate
				'GPC'   : np.array([   0.01 ,   1.80]), 									# GlyceroPhosphoCholine
				'GSH'   : np.array([   0.94 ,   1.53]), 									# Glutathione
				'Glyn'  : np.array([   0.94 ,   1.53]), 									# Glycine
				'Glyc'  : np.array([   0.00 ,   0.00]), 									# Glycerol
				'H2O'   : np.array([   0.00 ,   0.00]), 									# H2O
				'Hist'  : np.array([   0.00 ,   0.00]), 									# Histamine
				'Hisd'  : np.array([   0.00 ,   0.00]), 									# Histadine
				'HCr'   : np.array([   0.00 ,   0.00]), 									# HomeCarnosine
				'Lac'   : np.array([   0.47 ,   0.77]), 									# Lactate

				'MM_092': np.array([   1.00 ,  34.38]), 									# MacroMolecule 0.92
				'MM_121': np.array([   1.00 ,   8.48]), 									# MacroMolecule 1.21
				'MM_139': np.array([   1.00 ,  64.47]), 									# MacroMolecule 1.39
				'MM_167': np.array([   1.00 ,  20.62]), 									# MacroMolecule 1.67
				'MM_204': np.array([   1.00 ,  69.12]), 									# MacroMolecule 2.04
				'MM_226': np.array([   1.00 ,  34.42]), 									# MacroMolecule 2.26
				'MM_256': np.array([   1.00 ,   4.15]), 									# MacroMolecule 2.56
				'MM_270': np.array([   1.00 ,   7.00]), 									# MacroMolecule 2.70
				'MM_299': np.array([   1.00 ,  12.39]), 									# MacroMolecule 2.99
				'MM_321': np.array([   1.00 ,  10.45]), 									# MacroMolecule 3.21
				'MM_362': np.array([   1.00 ,   5.10]),										# MacroMolecule 3.62
				'MM_375': np.array([   1.00 ,  12.00]),										# MacroMolecule 3.75
				'MM_386': np.array([   1.00 ,   3.84]),										# MacroMolecule 3.86
				'MM_403': np.array([   1.00 ,   7.95]),										# MacroMolecule 4.03

				'Myo'   : np.array([   3.29 ,  12.18]), 									# MyoInositol
				'NAA'   : np.array([   9.75 ,  18.80]), 									# N-AcetylAspartate
				'NAG'   : np.array([   0.94 ,   1.53]), 									# N-AcetlyAsparate-Glutamate 
				'PCh'   : np.array([   0.01 ,   1.74]), 									# PhosphoCholine
				'PCr'   : np.array([   3.38 ,   6.44]), 									# PhosphoCreatine
				'PhE'   : np.array([   1.41 ,   2.30]), 									# PhosphoEthanolAmine
				'PEtOH' : np.array([   0.00 ,   0.00]), 									# PhosphoEthylAlcohol
				'Phy'   : np.array([   0.00 ,   0.00]), 									# PhenylAlanine
				'Scy'   : np.array([   0.23 ,   0.39]), 									# ScylloInositol
				'Ser'   : np.array([   0.00 ,   0.00]), 									# Serine
				'Tau'   : np.array([   1.41 ,   2.30]), 									# Taurine
				'Thr'   : np.array([   0.00 ,   0.00]), 									# Threonine
				'Try'   : np.array([   0.00 ,   0.00]), 									# Tryptophan
				'Tyr'   : np.array([   0.00 ,   0.00]), 									# Tyrosine
				'Val'   : np.array([   0.00 ,   0.00]), 									# Valine
			   }

	return conc

def Gen_Data_Training(basedir, field, necho, fname, dataset='Gudmundson', batch_size=20, n=2048, verbose=True): # Function to Generate Data 
	
	## Define some Constants
	#  SpecWidth and Dwell are initially based on Maximum Possible. 24 possibilities are available through Subsampling FID and changing number of points.
	# 

	larmor    = field[0] * 42.5760 															# Larmor Frequency
	if field[0] == 3.00: 																		
		sw    = 8000 																		# 8000 Hz at 3.00T
	else:
		sw    = larmor * 62.633095 															# Match Frequency Range
	dt        = 1/sw 																		# Current Dwell Time (This may change with sub-sampling below)
	sw        = int(np.round( sw,0)) 														# Round to Whole Number

	## This dataset has the ability to include points before the top of the echo.
	#  There are 300 points in the array defined within the basis set.
	#  The actual number of points before is constrained by the number possible between the last refocusing pulse.
	#  The pb_strt variable below is the number of points possible.
	#  In the standard 'out-of-the-box' implementation, this feature was not included.
	#  When using this feature, make sure to appropriately add T2 decay
	# 

	te2       = necho[0]/2000 																# TE = (TE1 + TE2 + TE3; Where TE2 = 1/2 of TE)
	te3       = te2/2 																		# TE = (TE1 + TE2 + TE3; Where TE1 and TE3 = 1/4 of TE)
	pb        = int(te3/dt)																	# Number of Points Possible Before Echo
	pb_strt   = 300 - pb 																	# Determined Max 300 points (Hard-Coded Array Length) given TE=150ms

	delay    = np.array([.00003125]) * 0													# Length of Delay Between Last Refocusing Pulse and Acquisition
	delay_   = '{:.8f}'.format(delay[0]) 													# For Ease in Filenames: String of Delay Time
	delay_   = 'delay-{}sec'.format(delay_[2:]) 											# For Ease in Filenames: String of Delay Time

	print('Field: {:7.2f}T'.format(field[0]))
	print('TE   : {:7.0f}ms'.format(necho[0]))
	print('SW   : {:7.0f}Hz'.format(sw))
	print('DT   : {:8.6f}s'.format(dt))
	
	necho_    = 'TE-{:03d}'.format(necho[0]) 												# Define for Ease in Filenames: String of Echo Time
	field_    = 'B0-{:<03}'.format(str(field[0]).replace('.','')) 							# Define for Ease in Filenames: String of Field Strength
	field_    = field_[:6]
	sw_       = 'sw-{:4d}'.format(int(np.round( sw,0)))       								# Define for Ease in Filenames: String of Bandwidth
	pb_       = 'pb-{:03d}'.format(pb) 				      									# Define for Ease in Filenames: String of N Points Before Echo

	# Load in the Basis Set:
	#  There are 270 basis sets.
	#  18 Field Strengths and 15 Echo Times
	#  Different Field Strengths will increase the Hertz or distance between spin resonances.
	#  Different Echo Times will change the coupling evolution. 
	#  Recommended to use more to really capture all the possible patterns that exist within the data.
	# 
	#  Note** This is a simplified version of the gen_data script.
	#           Here the choice for specwidth and number of points is already made to reduce computational load.
	#           The script is still structured the same way..  which may lead to some redundencies in the code.
	# 

	# npoints   = np.random.choice(np.array([512, 1024, 2048, 4096]), batch_size, replace=True) 	# Number of Points for Current Simulation
	npoints   = np.random.choice(np.array([4096]), batch_size, replace=True) 				# Number of Points for Current Simulation
	subsmple  = np.random.choice(np.arange(2,5)  , batch_size, replace=True) 				# Number of Points for Current Simulation
	subsmple_ = copy.deepcopy(subsmple) - 2 												# Center on 0 for Indexing

	glb_idx   = fname.split('.')[0]
	fname     = 'AG_{}_{}_{}_{}_{}_{}'.format(field_, necho_, sw_, pb_, delay_, fname )		# Filename
	# datadir   = '{}/training_data/Revised_Combined'.format(basedir) 						# Local File Directory
	# datadir   = '{}/basis_norm'.format(basedir) 											# FIBRE File Directory
	datadir   = 'basis_norm'.format(basedir) 											# FIBRE File Directory
	
	try:
		data  = 'data_{}_{}_{}_{}_{}.npz'.format(field_, necho_, sw_, pb_, delay_ )			# Filename 
		data  = np.load('{}/{}'.format(datadir, data))
	except:
		pb_   = 'pb-{:03d}'.format(pb+1) 				      								# Define for Ease in Filenames: String of N Points Before Echo
		data  = 'data_{}_{}_{}_{}_{}.npz'.format(field_, necho_, sw_, pb_, delay_ )			# Filename 
		data  = np.load('{}/{}'.format(datadir, data))

	spins     = list(data['names']) 														# Unique Spins 

	m         = np.zeros([n, len(spins), batch_size], dtype=np.complex_)  					# Final Data Array
	water     = np.zeros([n, 5,          batch_size], dtype=np.complex_) 					# Water Array
	time      = np.zeros([n,             batch_size]                   )  					# Time Array
	ppm       = np.zeros([n,             batch_size] 				   )  					# ppm Array
	sws       = np.zeros([               batch_size]                   )

	for ii in range(batch_size):
		sw_tmp  = data['SW'][subsmple_[ii]]
		dt_     = (dt * subsmple[ii]) 														# Updated Subsampling Dwell Time
		sws[ii] = 1 / dt_ 			 														# Current Spectral Width										
		# if ii < 10:
		# 	print('{:3d} {:3d} {:3d} {:7.2f} {:7.2f}'.format(ii, subsmple[ii], subsmple_[ii], sw_tmp, sws[ii]))
		sws[ii] = data['SW'][subsmple_[ii]] 			 									# Current Spectral Width										

		time[ :, ii]  = np.linspace(0, dt_ * n, n) 											# Time axis
		ppm[  :, ii]  = np.linspace(0, sws[ii]/larmor, n) 									# ppm axis

		m[    :,:,ii] = data['data'][300:,  :, subsmple_[ii],     ] 						# Subsample and Take N Points
		water[:,:,ii] = data['data'][300:, 78, subsmple_[ii], None] 						# Water Basis
		
	ppm      -= ppm[n//2,:] 																# Center ppm axis
	ppm      += 4.7 																		# Center at Water

	data      = None 																		# Free up Space

	conc_dict = concentrations('Gudmundson_Healthy')  										# Concentration Ranges by Unique Spin
	clin_dict = clinical_concentrations() 													# Concentrations Ranges from Clinical Populations
	
	conc      = np.zeros([len(spins), 2]) 													# Concentration Ranges Convert to Array
	conc_     = np.zeros([len(spins), batch_size]) 											# Concentrations Array

	## Read in Metabolite T2 Decay Rates
	#    These values were calculated using the multiple meta-regression model from Gudmundson 2022 (In Prep)
	#    Values were computed at 1.5T where Extrinsic-T2 has the smallest impact. 
	#    Values were computed for various parameters including localization sequence, tissue type, etc.
	#    The output is a prediction or expected mean value.
	#    To create a range, we took the lowest and highest values after computing across the parameters above.
	#    The range was further extended by using 7.5% (arbitrary) of the 95% confidence interval.
	# 

	T2_Values = pd.read_csv('{}/T2_Equation_Ranges_Deep.csv'.format(datadir)) 				# Meta-Analysis T2 Relaxation Rates
	T2_Names  = list(set(T2_Values.Metabolite)) 											# Meta-Analysis Metabolite Names
	lbl       = np.zeros([len(spins), 2]) 			 										# Metab Lorentzian Decay (Milliseconds)
	lbl_      = np.zeros([len(spins), batch_size]) 			 								# Metab Lorentzian Decay (Milliseconds)

	## Assign Metabolite Concentrations for Clinical Populations
	#    Using the Clinical Dictionary, find the current Clinical Group.
	#    Next, determine whether the current metabolite is impacted by this disease
	#    Finally, pull from a random uniform distribution a value.
	#    The value represents a linear multiplier
	#    This value will move healthy concentration into the range of the clinical group.

	clin_conc = np.ones([len(T2_Names), batch_size]) 										# Array for Clinical Concentrations 
	clin_grps = ['Healthy'] 																# This will be the Clinical Names
	clin_grps.extend(list(sorted(clin_dict.keys()))) 										# This will be the Clinical Names
	pop       = np.random.choice(np.arange(0,22), size=batch_size) 							# Healthy = 0; Clinical > 0 (Corresponds to List Indices clin_grps)

	for ii in range(pop.shape[0]): 															# Assign Clinical Concentrations
		if pop[ii] > 0:
			curr = clin_dict[clin_grps[pop[ii]]]

			for jj in range(len(T2_Names)):
				if T2_Names[jj] in list(curr.keys()):
					clin_conc[jj, ii] *= np.random.uniform(curr[T2_Names[jj]][0], curr[T2_Names[jj]][1], 1)
					# print('{:3d}: {:<12} {:<8} {:6.3f} {:6.3f}  |   {:6.3f}'.format(ii, clin_grps[pop[ii]][:12], T2_Names[jj][:8], curr[T2_Names[jj]][0], curr[T2_Names[jj]][1], clin_conc[jj,ii]))

	clssfier0 = copy.deepcopy(pop) 															# Healthy vs Clinical Classifier
	clssfier0[clssfier0 > 0] = 1  															# Healthy = 0; Clinical = 1

	MMs       = []
	mets      = []
	spin_dict = {}

	for ii in range(len(spins)): 															# Match Concentrations Dict w/Basis Set		
		if 'MM' in spins[ii][:3]:
			spin  = spins[ii]
			MMs.append(ii)
		else:
			spin  = spins[ii].split('_')[0] 												# Remove ppm from Basis Name
			mets.append(ii)
		
		if spins[ii] in T2_Names:
			if spins[ii] not in list(set(spin_dict.keys())):
				spin_dict[spins[ii]] = [[],[]]
			spin_dict[spins[ii]][0].append(ii)
			spin_dict[spins[ii]][1].append(spins[ii])
			T2_Values_ = T2_Values[T2_Values.Metabolite == spins[ii]].reset_index(drop=True)
		
		elif spin in T2_Names:
			if spin not in list(set(spin_dict.keys())):
				spin_dict[spin] = [[],[]]
			spin_dict[spin][0].append(ii)
			spin_dict[spin][1].append(spins[ii])
			T2_Values_ = T2_Values[T2_Values.Metabolite == spin].reset_index(drop=True)
		
		else:
			if spin not in list(set(spin_dict.keys())):
				spin_dict[spin] = [[],[]]

			spin_dict[spin][0].append(ii)
			spin_dict[spin][1].append(spins[ii])
			T2_Values_ = pd.DataFrame({'T2_Low': [1], 'T2_Hgh': [1]})

		conc[ii,:]= conc_dict[spin] 	 													# Corresponding Metabolite Concentration
		lbl[ii,0] = T2_Values_.T2_Low[0]
		lbl[ii,1] = T2_Values_.T2_Hgh[0]

	MM_Names      = ['MM_092', 'MM_121', 'MM_139', 'MM_167', 'MM_204', 
				     'MM_226', 'MM_256', 'MM_270', 'MM_299', 'MM_321', 
				     'MM_362', 'MM_375', 'MM_386', 'MM_403']
	MMs           = np.squeeze(np.array([MMs])) 											# Macromolecule Indices
	mets          = np.squeeze(np.array([mets]))											# Metabolite Indices

	fwhm          = np.zeros([len(spins),batch_size]) 										# Final FWHM After Gaussian Decay
	fwhm[:,:]     = np.random.uniform(2,18, batch_size)[None,:]								# Final FWHM After Gaussian Decay
	
	## Full Width Half Max of MMs. These values are all estimates from Figure 6 in Murali-Manohar 2020 (DOI: 10.1002/mrm.28174)
	fwhm[MMs[ 0],:] = np.random.uniform( 30,50, batch_size)[None,:]							# MM 092 - Final FWHM After Gaussian Decay
	fwhm[MMs[ 1],:] = np.random.uniform( 30,50, batch_size)[None,:]							# MM 121 - Final FWHM After Gaussian Decay
	fwhm[MMs[ 2],:] = np.random.uniform( 30,55, batch_size)[None,:]							# MM 139 - Final FWHM After Gaussian Decay
	fwhm[MMs[ 3],:] = np.random.uniform( 45,70, batch_size)[None,:]							# MM 167 - Final FWHM After Gaussian Decay
	fwhm[MMs[ 4],:] = np.random.uniform( 45,70, batch_size)[None,:]							# MM 204 - Final FWHM After Gaussian Decay
	fwhm[MMs[ 5],:] = np.random.uniform( 50,70, batch_size)[None,:]							# MM 226 - Final FWHM After Gaussian Decay
	fwhm[MMs[ 6],:] = np.random.uniform( 40,60, batch_size)[None,:]							# MM 256 - Final FWHM After Gaussian Decay
	fwhm[MMs[ 7],:] = np.random.uniform( 20,40, batch_size)[None,:]							# MM 270 - Final FWHM After Gaussian Decay
	fwhm[MMs[ 8],:] = np.random.uniform( 40,60, batch_size)[None,:]							# MM 299 - Final FWHM After Gaussian Decay
	fwhm[MMs[ 9],:] = np.random.uniform( 60,80, batch_size)[None,:]							# MM 321 - Final FWHM After Gaussian Decay
	fwhm[MMs[10],:] = np.random.uniform( 35,55, batch_size)[None,:]							# MM 362 - Final FWHM After Gaussian Decay
	fwhm[MMs[11],:] = np.random.uniform( 25,45, batch_size)[None,:]							# MM 375 - Final FWHM After Gaussian Decay
	fwhm[MMs[12],:] = np.random.uniform( 25,45, batch_size)[None,:]							# MM 386 - Final FWHM After Gaussian Decay
	fwhm[MMs[13],:] = np.random.uniform( 25,45, batch_size)[None,:]							# MM 403 - Final FWHM After Gaussian Decay

	GABA_Choice   = np.random.randint(0,2,batch_size) 										# 0 = Near; 1=Govindaraju
	GABA0         = np.where(GABA_Choice == 1)[0] 											# 
	GABA1         = np.where(GABA_Choice == 0)[0] 											# 
	Glc_Choice    = np.random.randint(0,2,batch_size) 										# 0 = GlcA; 1=GlcB
	Glc0          = np.where(Glc_Choice == 1)[0] 											# 
	Glc1      	  = np.where(Glc_Choice == 0)[0] 											# 

	# Determine Concentration and Lorentzian/T2-intrinsic Component
	for ii in range(len(T2_Names)):
		curr_spin = np.array(spin_dict[T2_Names[ii]][0])
		lbl_[ curr_spin,:] = np.random.uniform( lbl[ curr_spin[0],0], lbl[ curr_spin[0],1], batch_size) # T2 Intrinsic
		conc_[curr_spin,:] = np.random.uniform( conc[curr_spin[0],0], conc[curr_spin[0],1], batch_size) # Concentrations

		for jj in range(batch_size):
			conc_[curr_spin,jj] *= clin_conc[ii,jj]

	conc_[ 14, :]    = conc_[ 16, :]														# Match Creatine Concentrations
	conc_[ 15, :]    = conc_[ 16, :]														# Match Creatine Concentrations

	conc_[135, :]    = conc_[138, :]														# Match PhosphoCreatine Concentrations
	conc_[136, :]    = conc_[138, :]														# Match PhosphoCreatine Concentrations
	conc_[137, :]    = conc_[138, :]														# Match PhosphoCreatine Concentrations

	## There are two possible GABA signals and two structural variants of Glucose signals.
	#  Here we select 1 of each to be used within the simulations.
	#  Though, rather than picking 1, we randomly select across the batch by setting the other to 0 concentration.
	#  This may help networks better generalize to in-vivo inputs.
	# 

	conc_[24, GABA0] = 0 																	# GABA Near (Choice == 1)
	conc_[25, GABA0] = 0 																	# GABA Near (Choice == 1)
	conc_[26, GABA0] = 0 																	# GABA Near (Choice == 1)
	conc_[27, GABA1] = 0 																	# GABA Near (Choice == 0)
	conc_[28, GABA1] = 0 																	# GABA Near (Choice == 0)
	conc_[29, GABA1] = 0 																	# GABA Near (Choice == 0)

	conc_[48, Glc0 ] = 0 																	# Glucose A (Choice == 1)
	conc_[49, Glc0 ] = 0 																	# Glucose A (Choice == 1)
	conc_[50, Glc0 ] = 0 																	# Glucose A (Choice == 1)
	conc_[51, Glc0 ] = 0 																	# Glucose A (Choice == 1)
	conc_[52, Glc0 ] = 0 																	# Glucose A (Choice == 1)
	conc_[53, Glc0 ] = 0 																	# Glucose A (Choice == 1)
	conc_[54, Glc0 ] = 0 																	# Glucose A (Choice == 1)

	conc_[55, Glc1 ] = 0 																	# Glucose B (Choice == 0)
	conc_[56, Glc1 ] = 0 																	# Glucose B (Choice == 0)
	conc_[57, Glc1 ] = 0 																	# Glucose B (Choice == 0)
	conc_[58, Glc1 ] = 0 																	# Glucose B (Choice == 0)
	conc_[59, Glc1 ] = 0 																	# Glucose B (Choice == 0)
	conc_[60, Glc1 ] = 0 																	# Glucose B (Choice == 0)
	conc_[61, Glc1 ] = 0 																	# Glucose B (Choice == 0)

	# Noise, Frequency Shifts, and Phase Rotations
	noise         = np.random.normal(0, 1, (n, 2, batch_size)) 								# Normally Distributed Noise
	noise        *= (conc_[None,117,:] + conc_[None,122,:])[None,:] 						# Same range as Metabolites	
	lbl_[MMs,:]   = np.random.uniform(     20,    60, (MMs.shape[0],batch_size)) 			# MMs   Lorentzian Decay (Milliseconds)

	hz_313        = np.round(.313 * larmor,0) 												# Determine Hertz across Field Strength to Match .313ppm Frequency Shift
	print('Freq : {}{:3.0f}Hz'.format(u'\u00B1', hz_313))
	frq           = np.random.uniform( -hz_313,  hz_313, batch_size) 						# Frequency Shift  (Hertz; ==> .313ppm)

	phs0          = np.random.uniform(-np.pi ,   np.pi, batch_size) 						# 0th Order Phase  (Radians) 
	phs1          = np.random.uniform(  0.00 ,   0.340, batch_size) 						# 1st Order Phase  (Radians)
	phs1_pivot    = np.random.uniform(  2.00 ,   5.00 , batch_size) 						# 0 Point in 1st Order Phase (ppm)

	snr           = np.random.uniform( 5, 70, batch_size) 									# Random SNR for Training
	snr_amp       = np.exp(-1 * snr * .05212) 												# Approximated SNR offline for given amplitude to this Fx
	snr_mult      = np.zeros([batch_size])													# Multiplier to Achieve Target SNR

	## Create Water Signal
	center        = 4.7
	water_shift   = lambda x,ppm: x * np.exp(-2j * time[:] * np.pi * (larmor * (center-ppm)))
	water_lbl     = np.random.uniform(.050,.100, (5, batch_size)) 							# Water Lorentzian Decay
	water        *= np.exp(-1 * time[:,None,:] * ( 1/water_lbl[None,:,:] ))					# Water Lorentzian Decay

	water_cmp     = np.zeros([5, batch_size])
	water_cmp[0,:]= np.random.uniform(4.679, 4.711, batch_size) 							# Water Component 1
	water_cmp[1,:]= np.random.uniform(4.599, 4.641, batch_size) 							# Water Component 2
	water_cmp[2,:]= np.random.uniform(4.801, 4.759, batch_size) 							# Water Component 3
	water_cmp[3,:]= np.random.uniform(4.449, 4.541, batch_size) 							# Water Component 4
	water_cmp[4,:]= np.random.uniform(4.859, 4.901, batch_size) 							# Water Component 5

	water[:,0,:] *= water_shift(water[:,0,:], water_cmp[0,:]) 								# Shift Water Component 1 [4.675 - 4.705]
	water[:,1,:] *= water_shift(water[:,1,:], water_cmp[1,:]) 								# Shift Water Component 2 [4.595 - 4.645]
	water[:,2,:] *= water_shift(water[:,2,:], water_cmp[2,:]) 								# Shift Water Component 3 [4.755 - 4.805]
	water[:,3,:] *= water_shift(water[:,3,:], water_cmp[3,:]) 								# Shift Water Component 4 [4.495 - 4.545]
	water[:,4,:] *= water_shift(water[:,4,:], water_cmp[4,:]) 								# Shift Water Component 5 [4.855 - 4.905]

	water_phs     = np.zeros([5, batch_size]) 												# Water Phase
	water_phs[0,:]= np.random.uniform(-10,   10, batch_size) 								# Water Component 1 Phase (Degrees)
	water_phs[1,:]= np.random.uniform( 15,   45, batch_size) 								# Water Component 1 Phase (Degrees)
	water_phs[2,:]= np.random.uniform(-60,  -30, batch_size) 								# Water Component 1 Phase (Degrees)
	water_phs[3,:]= np.random.uniform( 45,  -70, batch_size) 								# Water Component 1 Phase (Degrees)
	water_phs[4,:]= np.random.uniform(105,  135, batch_size) 								# Water Component 1 Phase (Degrees)
	water_phs    *= np.pi/180 																# Convert to radians
	water        *= np.exp(-1j * water_phs)

	water_pos     = np.zeros([batch_size])
	water_pos[:]  = np.random.randint(0,2,batch_size) 										# Water Positve/Negative (Pos=0; Neg=1)
	water        *= np.exp(1j * np.pi * water_pos[None,None,:])								# Apply Positive/Negative Phase Rotation

	water_amp     = np.zeros([5, batch_size]) 												# Water Amplitude
	water_amp[0,:]= np.random.uniform(1.00,20.00,batch_size) 								# Water Amplitude [Scaling = Amp * max(Metabolites)]
	water_amp[1,:]= np.random.uniform(0.35, 0.55,batch_size) 								# Component 2 is apprx. half of Component 1
	water_amp[2,:]= np.random.uniform(0.35, 0.55,batch_size) 								# Component 3 is apprx. half of Component 1
	water_amp[3,:]= np.random.uniform(0.10, 0.25,batch_size) 								# Component 4 is apprx. 1/5  of Component 1
	water_amp[4,:]= np.random.uniform(0.10, 0.25,batch_size) 								# Component 5 is apprx. 1/5  of Component 1

	water_ncomps  = np.zeros([5, batch_size]) 												# Number of Components to Include
	for ii in range(batch_size): 															# Iterate over Batch
		a         = np.random.randint(0,6,1)[0] 											# Determine Number of Components to Include (0-5 possible)
		a         = np.random.choice(np.arange(5), a, replace=False) 						# Randomly Choose Components
		water_ncomps[a,ii] = 1 																# 0 = Excluded; 1 = Included
		water_amp[1:,ii] *= water_amp[0,ii]
		water_amp[:,ii]  *= water_ncomps[:,ii]

	b0        	  = '{:<03}'.format(str(field[0]).replace('.','')) 							# Create Field Name	
	te        	  = '{:03d}'.format(necho[0]) 												# Create Echo Name

	m            *= conc_[None,:,:] 														# Concentrations
	m            *= np.exp(-1 * time[:,None,:] * (1000/lbl_[None,:,:]) ) 					# Lorentzian Decay

	glb           = np.zeros([len(spins)  , batch_size]) 									# Gaussian Line Broadening
	water_glb     = np.zeros([5           , batch_size]) 									# Water Gaussian Decay
	MetabFWHM     = np.zeros([              batch_size]) 									# Metabolite FWHM
	MMFWHM        = np.zeros([MMs.shape[0], batch_size])									# MM FWHM
	namp          = np.zeros([              batch_size]) 									# Noise Amplitudes
	SNR           = np.zeros([              batch_size]) 									# Resulting SNR
	
	## Optimize for Controlled Full Width Half Max (FWHM) and Signal to Noise Ratio (SNR)
	#  Here, we match the Full Width Half Max and SNR for all possible Spectral Widths and Number of points.
	#  This adds another layer of control as users can test networks against the exact same signal under different paramaters.
	#  
	#  Note** This is a simplified version of the gen_data script.
	#           Here the choice for specwidth and number of points is already made to reduce computational load.
	#           The script is still structured the same way..  which may lead to some redundencies in the code.

	phs_1         = {'1.4': 1.75, '1.5': 0.0, '1.6': 4.50, 							# Discovered Slight 1st Order Phase for some B0
					 '1.7': 1.75, '1.8': 0.0, '1.9': 4.25,							# Resulting Corrections in Degrees
					 '2.0': 1.75, '2.1': 0.0, '2.2': 4.00,	
					 '2.3': 1.75, '2.4': 0.0, '2.5': 3.75,
					 '2.6': 1.75, '2.7': 0.0, '2.8': 3.50,
					 '2.9': 1.75, '3.0': 0.0, '3.1': 3.25,}

	phs_1_   	  = phs_1[str(np.round(field[0], 1))] 								# Read Dictionary Value for 1st Order Phase Correction
	n_            = int(npoints[ii])

	if verbose == True:
		print('                  {:<5}    {:<6}  {:<6} {:<6}  {:<6}  {:<5} {:<4}   {:<6} {:<5}'.format('T_SNR', 'SNR', 'Diff', 'T_FWHM', 'A_FWHM', 'Diff', 'GLB', 'MM_u', 'MM_Sd'))
	for ii in range(batch_size):
		Z             = 2**17 																# Zero Filled Length
		Zhz           = np.linspace(0,sws[ii],Z) 											# Zero Filled Hertz Axis
		Zhz          -= Zhz[Z//2] 				 											# Zero Filled Hertz Axis

		Zppm          = np.linspace(0, sws[ii]/larmor, Z)  									# Zero Filled ppm Axis
		Zppm         -= Zppm[Z//2] 						  									# Zero Filled ppm Axis
		Zppm         += 4.7  	 						  									# Zero Filled ppm Axis
		naa_idx       = np.where( np.abs(Zppm-2.01) == np.min(np.abs(Zppm-2.01)) )[0][0]	# NAA 2.008 Index
		# MM_idx        = np.where( np.abs(Zppm-0.92) == np.min(np.abs(Zppm-0.92)) )[0][0]	# MM  0.920 Index

		## FWHM of Metabolites (Based on NAA 2.008, but all Metabolites will have approximately the same FWHM)
		tNAA             = copy.deepcopy(m[:,117,ii]) 										# NAA 2.008 Signal
		args             = (naa_idx, tNAA.real, tNAA.imag, time[:,ii], Zppm, Zhz, fwhm[117,ii], n_, phs_1_)# Optimization Arguments
		theta            = minimize(opt_fwhm, x0=np.array([3]), bounds=np.array([[.1, 35]]), args=args, options={'ftol': 1e-6}, method='powell')

		glb[:,ii]  		 = theta.x[0]														# Assign Gaussian Component to All Metabolites
		gauss_jitter     = np.random.uniform(-.40, .40, len(spins))	 						# Define Random Jitter For Small Differences in Gauss Decay Across Spins
		glb[:,ii] 		+= gauss_jitter 													# Add Random Jitter 
		
		water_glb[:,ii]  = theta.x[0] 														# Assign Gaussian Component to Water Components
		water_jitter     = np.random.uniform(-.40, .40, 5)	 								# Define Random Jitter For Small Differences in Gauss Decay Water Components
		water_glb[:,ii] += water_jitter 													# Add Random Jitter	

		tNAA            *= np.exp(-1 * time[:,ii]**2 * np.pi * theta.x[0]**2) 				# Multiply Gaussian
		tNAA             = np.fft.fftshift(np.fft.fft(tNAA[:n_], n=Z)) 		 				# Fourier Transform
		tNAA            *= np.exp(1j * (Zppm - 4.70) * phs_1_ * np.pi/180) 					# 1st Order Phase Correction
		tNAA             = np.real(tNAA) 													# Real Component
		tNAA            /= np.max(tNAA)  													# Normalize
		MetabFWHM[ii]    = _fwhm(naa_idx, tNAA, axis=Zhz)[0] 								# Determine and Confirm FWHM from Actual Spectrum

		## FWHM of MacroMolecules
		for jj in range(len(MM_Names)):
			MM_          = MM_Names[jj].split('_')[-1]
			MM_          = '{}.{}'.format(MM_[0], MM_[1:])
			MM_          = float(MM_)

			Midx         = MMs[jj]
			MM_idx       = np.where( np.abs(Zppm-MM_) == np.min(np.abs(Zppm-MM_)) )[0][0]	# MM Current Index

			MM_fwhm      = copy.deepcopy(m[:, MMs[jj], ii]) 								# MM Current Signal
			args         = (MM_idx, MM_fwhm.real, MM_fwhm.imag, time[:,ii], Zppm, Zhz, fwhm[MMs[jj],ii], n_, phs_1_)
			theta        = minimize(opt_fwhm, x0=np.array([40]), bounds=np.array([[20, 80]]), args=args, options={'ftol': 1e-6}, method='powell') # 
			glb[Midx,ii] = theta.x[0]														# Assign Gaussian Component to All Metabolites
			gauss_jitter = np.random.uniform(-2.5, 2.5, 1)	 								# Define Random Jitter For Small Differences in Gauss Decay Across MM Spins
			glb[Midx,ii]+= gauss_jitter 													# Add Random Jitter	
			
			MM_fwhm     *= np.exp(-1 * time[:,ii]**2 * np.pi * theta.x[0]**2)				# Multiply Gaussian
			MM_fwhm      = np.fft.fftshift(np.fft.fft(MM_fwhm[:n_], n=Z))  					# Fourier Transform
			MM_fwhm     *= np.exp(1j * (Zppm - 4.70) * phs_1_ * np.pi/180) 					# 1st Order Phase Correction
			MM_fwhm      = np.real(MM_fwhm)  												# Real Component
			MM_fwhm     /= np.max(MM_fwhm)  												# Normalize
			
			MMFWHM[jj,ii]= _fwhm(MM_idx, MM_fwhm, axis=Zhz)[0] 								# Determine and Confirm FWHM from Actual Spectrum


		## SNR 
		noise[:,:,ii]*= snr_amp[ii] 														# This just puts the SNR in a better starting point for Optimization below.

		Z             = 2**17 																# Zero Filled Length
		Zhz           = np.linspace(0,sws[ii],Z) 											# Zero Filled Hertz Axis
		Zhz          -= Zhz[Z//2] 				 											# Zero Filled Hertz Axis

		Zppm          = np.linspace(0, sws[ii]/larmor, Z)  									# Zero Filled ppm Axis
		Zppm         -= Zppm[Z//2] 						  									# Zero Filled ppm Axis
		Zppm         += 4.7  	 						  									# Zero Filled ppm Axis
		naa_idx       = np.where( np.abs(Zppm-2.01) == np.min(np.abs(Zppm-2.01)) )[0][0]	# NAA 2.008 Index
		# MM_idx        = np.where( np.abs(Zppm-0.92) == np.min(np.abs(Zppm-0.92)) )[0][0]	# MM  0.920 Index
				
		noise_         = copy.deepcopy(noise[:n_, :, ii])									# Match Number of Points 
		noise_         = np.vectorize(complex)(noise_[:,0], noise_[:,1]) 					# Vectorize to Complex

		tNAA_clean     = (copy.deepcopy(m[:n_, 117, ii]) * np.exp(-1 * time[:n_,ii]**2 * np.pi * glb[117,ii])
					     +copy.deepcopy(m[:n_, 122, ii]) * np.exp(-1 * time[:n_,ii]**2 * np.pi * glb[122,ii]))

		clean_spec     = np.fft.fftshift(np.fft.fft(tNAA_clean, n=time.shape[0])) 			# Fourier Transform tNAA
		clean_spec    *= np.exp(1j * (ppm[:,ii] - 4.70) * phs_1_ * np.pi/180) 				# 1st Order Phase Correction
		tNAA_amp       = np.max(np.real(clean_spec))							   			# tNAA Amplitude 
		
		args           = (tNAA_amp, noise_.real, noise_.imag, snr[ii], time.shape[0])
		theta          = minimize(opt_noise, x0=np.array([1]), bounds=np.array([[.001, 40]]), args=args, method='L-BFGS-B')

		namp[ii]       = theta.x[0]  											   			# Noise Amplitude
		noise_        *= theta.x[0] 														# Multiply Temperory Noise Vector by Optimized Amplitude
		noise_spec     = np.fft.fftshift(np.fft.fft(noise_, n=time.shape[0]))				# Fourier Transform 
		noise_spec    *= np.exp(1j * (ppm[:,ii] - 4.70) * phs_1_ * np.pi/180) 				# 1st Order Phase Correction
		nstd_new       = np.std(np.real(noise_spec)) 										# Final Noise Standard Deviation

		SNR[ii]        = tNAA_amp/nstd_new 													# Final SNR (NAA/Std_Noise)

		if verbose == True:  																# List out Parameters for each Iteration
			u_mm       = np.mean( np.abs(MMFWHM[:,ii] - fwhm[MMs,ii]) ) 					# Summed MM Error in FWHM 
			std_mm     = np.std(  np.abs(MMFWHM[:,ii] - fwhm[MMs,ii]) ) 					# StdDev MM Error in FWHM 

			string     = '{} {:3d} {} {:3d}: '.format(field[0], necho[0], glb_idx, ii)
			string     = '{} {:5.2f}  {:6.2f}  {:6.2f} | '.format(string, snr[ii], SNR[ii], np.abs(snr[ii]-SNR[ii]))
			string     = '{} {:5.2f}  {:5.2f}  {:5.2f} {:5.2f} |'.format(string, fwhm[117,ii], MetabFWHM[ii], np.abs(fwhm[117,ii]-MetabFWHM[ii]), glb[117,ii])
			string     = '{} {:5.3f}  {:5.3f}'.format(string, u_mm, std_mm)
		
			print(string)

	Zhz         = None																		# Delete Variable
	Zppm        = None																		# Delete Variable

	m          *= np.exp(-1 * np.pi * time[:,None,:]**2 *       glb[:,:]**2) 				# Gaussian Component Metabolites/MMs
	water      *= np.exp(-1 * np.pi * time[:,None,:]**2 * water_glb[:,:]**2) 				# Gaussian Component Water

	noise       = np.vectorize(complex)(noise[:,0,:], noise[:,1,:]) 						# Complex Noise
	noise      *= namp[None,:] 																# Scale the Noise to Determined SNR

	MM_jitter   = np.random.uniform(-5, 5, (MMs.shape[0], batch_size)) 						# Add Jitter to make up for uncertainty in MM location
	m[:,MMs,:] *= np.exp(2j * time[:,None,:] * np.pi * MM_jitter)  							# Add Jitter to make up for uncertainty in MM location

	## This is a data augmentation step
	#  Here, we want to make it possible to remove some, but not all, Metabolite or MMs
	#  All the Metabolites, MMs, or Water Signals can be removed by simply not adding them into the inputs.
	#  
	#  Since all the spin signals are ultimately summed, individual signals must be subtracted out at time of trianing.
	#  Said another way, these options are NOT implemented in the final data.. The drop_sig array must be subtracted at training.
	# 
	#  There are 4 options that we've made available where there is a 10% chance for removal to occur:
	#    - 1 = Creatine
	#    - 2 = NAA
	#    - 3 = GPC
	#    - 4 = Randomly Selected MM

	drop_sig         = np.zeros([m.shape[0], batch_size], dtype=np.complex_)  				# Drop Signal
	batch_drop       = np.random.choice( np.array([0,1]), size=batch_size, p=np.array([.10, .90]))
	batch_drop_choice= np.zeros([batch_size])
	batch_drop_didx  = np.zeros([batch_size])
	
	for ii in range(batch_size):
		if batch_drop[ii] == 0:
			drop     = np.random.choice( np.array([1,2,3,4]), size=(1))[0]
			batch_drop_choice[ii] = drop
			
			if drop == 1:
				drop_idxs = np.array([14,15,16])           									# Drop Creatine (Idxs are Individual Spins within Array)
			elif drop == 2:
				drop_idxs = np.array([117,118,119,120,121])           						# Drop NAA (Idxs are Individual Spins within Array)
			elif drop == 3:
				drop_idxs = np.array([30,31,32,33,34,35,36,37])           					# Drop GPC (Idxs are Individual Spins within Array)
			elif drop == 4:
				didx   = np.random.choice( np.arange(99,113), size=(1))[0] 					# Drop Single MM (Idxs are Individual Spins within Array)
				drop_idxs = np.array([didx]) 
				batch_drop_didx[ii] = didx 													# Note which MM Signal is Dropped

			drop_sig[:,ii] = np.sum(m[:,drop_idxs, ii], axis=1) 							# Add Drop Signals that Can be Subtracted During Training

	## Ensure Signals are scaled and normalized correctly. Typically between -1 to 1.
	#    In doing so, ensure Metabolites are appropriately scaled to typical relative water amplitudes (Values Predefined Above).
	#    This step could be done during Training - but doing it here, will save time and memory load during training.
	#
	#  We want to make sure that we confine our data within a world of -1 to 1.
	#    So if we apply phase shifts, positive or negative water, etc. the data will always remain between -1 to 1
	#    The Mult arrays contain the scaling factors:
	#      - Metabolites will scale to 1  				(Last dimension of array - 0 for no Phase, 1 for Phase)
	#      - Water will scale to 5-20 x Metabolites 	(Last dimension of array - 0 for no Phase, 1 for Phase)
	#      - The Mult Arrays of shape Batch x 8 x 3; where 8 = Available SpecWidths and 3 = Available Npoints
	#    The final scaling to -1 to 1 must be done during training after the decision of:
	#      - Number of Points
	#      - Dwell Time/Spectral Width (by subsampling full fid)
	#      - Phase Rotations and any other Manipulations
	#
	#  Note** This is a simplified version of the gen_data script.
	#           Here the choice for specwidth and number of points is already made to reduce computational load.
	#           The script is still structured the same way..  which may lead to some redundencies in the code.

	## Metabolite/MM/Noise Scaling
	m_mult        = np.zeros([batch_size, 2, 2]) 											# Max Value scales Metabolites + MM to 1
	w_mult        = np.zeros([batch_size, 2, 2]) 											# Max Value scales Water to 1

	m_noPhs       = copy.deepcopy(m) 														# Metabs + MM without Phase/Freq Shifts
	m_noPhs       = np.nansum(m_noPhs, axis=1) 												# Sum Across All Metab and MM Spins
	m_noPhs      += copy.deepcopy(noise) 													# Add the Noise
	
	m_wPhs        = copy.deepcopy(m) 														# Metabs + MM with    Phase/Freq Shifts
	m_wPhs        = np.nansum(m_wPhs , axis=1) 												# Sum Across All Metab and MM Spins
	m_wPhs       += copy.deepcopy(noise) 													# Add the Noise
	m_wPhs       *= np.exp(-1j * phs0[None,:]) 												# Phase Shifts
	m_wPhs       *= np.exp( 2j * time * np.pi * frq[None,:])								# Frequency Shifts

	m_mult[:,0,0] = np.max(np.abs(np.real(m_noPhs)), axis=0) 								# Get Max for Real before Phase/Freq Shifts
	m_mult[:,1,0] = np.max(np.abs(np.imag(m_noPhs)), axis=0) 								# Get Max for Imag before Phase/Freq Shifts
	m_mult[:,0,1] = np.max(np.abs(np.real(m_wPhs )), axis=0)								# Get Max for Real After  Phase/Freq Shifts
	m_mult[:,1,1] = np.max(np.abs(np.imag(m_wPhs )), axis=0) 								# Get Max for Imag After  Phase/Freq Shifts

	m_mult        = np.max(m_mult, axis=1) 													# Max of Real and Imag components

	## Water Scaling
	w_noPhs       = copy.deepcopy(water) 													# Water without Phase/Freq Shifts
	w_noPhs       = np.nansum(w_noPhs, axis=1) 												# Sum Across All Water Components
	w_wPhs        = copy.deepcopy(water) 													# Water with    Phase/Freq Shifts
	w_wPhs        = np.nansum(w_wPhs , axis=1) 												# Sum Across All Water Components
	w_wPhs       *= np.exp(-1j * phs0[None,:]) 												# Phase Shifts
	w_wPhs       *= np.exp( 2j * time * np.pi * frq[None,:])								# Frequency Shifts

	w_mult[:,0,0] = np.max(np.abs(np.real(w_noPhs)), axis=0) 								# Get Max for Real before Phase/Freq Shifts
	w_mult[:,1,0] = np.max(np.abs(np.imag(w_noPhs)), axis=0) 								# Get Max for Imag before Phase/Freq Shifts
	w_mult[:,0,1] = np.max(np.abs(np.real(w_wPhs )), axis=0)								# Get Max for Real After  Phase/Freq Shifts
	w_mult[:,1,1] = np.max(np.abs(np.imag(w_wPhs )), axis=0) 								# Get Max for Imag After  Phase/Freq Shifts
	w_mult        = np.max(w_mult, axis=1) 													# Max of Real and Imag components
	water        *= water_amp[None,:,:] 													# Scale the Water Components
		
	## Final Data Reshaping for Ease of NN training (Make Batch Size 1st Dimension)
	#  Should have done this from the start... but here we are.
	final_metab             = np.nansum( copy.deepcopy(m[    :,mets,:]), axis=1)
	final_water             = np.nansum( copy.deepcopy(water[:,   :,:]), axis=1)
	final_mm                = np.nansum( copy.deepcopy(m[    :, MMs,:]), axis=1)
	final_noise             = 			 copy.deepcopy(noise[:,     :])
	
	final_metab             = final_metab.transpose( (1,0) )
	final_water             = final_water.transpose( (1,0) )
	final_mm                = final_mm.transpose(    (1,0) )
	final_noise             = final_noise.transpose( (1,0) )
	drop_sig                = drop_sig.transpose(    (1,0) )
	time                    = time.transpose(        (1,0) )
	ppm                     = ppm.transpose(         (1,0) )
	conc_                   = conc_.transpose(       (1,0) )
		
	water_cmp               = water_cmp.transpose(   (1,0) )
	water_ncomps            = water_ncomps.transpose((1,0) )
	water_amp               = water_amp.transpose(   (1,0) )
	
	lbl_                    = lbl_.transpose(        (1,0) )
	glb                     = glb.transpose(         (1,0) )

	water_lbl               = water_lbl.transpose(   (1,0) )
	water_glb               = water_glb.transpose(   (1,0) )
	MMFWHM                  = MMFWHM.transpose(      (1,0) )
	
	## Below is the Final Dictionary saved with all the data and parameters to create/recreate the data
	#  On the Right Hand Side is the Array Sizes and Description for Each
	#  Note** The array shape reflects the full version of the gen_data script.
	#           In this simpliefied version:
	#             - 16384 --> 4096     (Total FID Length)
	#             -     8 -->    1     (Subsampling or Spectral Widths)
	#             -     3 -->    1     (Number of Points)

																							# Array Shape    |  Description
	finl_dict               = {} 															#                |  CNN Data Dictionary
	finl_dict['Dataset'   ] = dataset 														# String         |  Dataset used in Data Simulations for Concentrations
	finl_dict['Field_Str' ] = np.ones([batch_size]) * field 								# Batch x 1      |  Field Strength Used (Tesla)
	finl_dict['Echo_Times'] = np.ones([batch_size]) * necho 								# Batch x 1      |  Echo Times Used (ms)
	finl_dict['sw'        ] = sws														    # Batch x 1      |  Spectral Widths Available (Hz) (Dependent on Subsampling and Field Strength)
	finl_dict['subsample' ] = subsmple   												    # Batch x 1      |  Subsampling Stride Corresponding to SpecWidth
	finl_dict['nPoints'   ] = npoints 														# Batch x 1      |  Number of Points (i.e. 512,1024,2048)
	finl_dict['Metab'     ] = final_metab 													# Batch x 4096   |  Metabolites    w/ Concentration, Lorentzian LB, and Gaussian LB
	finl_dict['MM'        ] = final_mm 														# Batch x 4096   |  Macromolecules w/ Concentration, Lorentzian LB, and Gaussian LB
	finl_dict['water'     ] = final_water													# Batch x 4096   |  Water          w/ 5Components  , Lorentzian LB, Gaussian LB, Scaling 5x-20x Metabolites
	finl_dict['noise'     ] = final_noise 													# Batch x 4096   |  Normal Distributed Noise
	finl_dict['time'      ] = time 															# Batch x 4096   |  Time Axis (seconds)
	finl_dict['ppm'       ] = ppm 															# Batch x 4096   |  Frequency Axis (ppm)
	finl_dict['Amplitude' ] = conc_ 														# Batch x 182    |  Concentration Used for Each Spin (Metabolite & MM) 
	finl_dict['water_pos' ] = water_pos 													# Batch x 1      |  Water is Positive or Negative (0=Pos; 1=Neg)
	finl_dict['water_comp'] = water_cmp 													# Batch x 5      |  PPM value of Each Component
	finl_dict['waterNcomp'] = water_ncomps 													# Batch x 5      |  Components Included
	finl_dict['water_amp' ] = water_amp  													# Batch x 5      |  Water Scaling
	finl_dict['noise_amp' ] = namp 															# Batch x 1      |  Noise Scaling (Equivalent SNR Across SpecWidth and Npoints)
	finl_dict['freq_shift'] = frq 															# Batch x 1      |  Frequency Shifts
	finl_dict['phase0'    ] = phs0 															# Batch x 1      |  0th Order Phase 
	finl_dict['phase1'    ] = phs1 															# Batch x 1      |  1st Order Phase
	finl_dict['phase1_piv'] = phs1_pivot													# Batch x 1      |  1st Order Phase Pivot Point
	finl_dict['SNR'       ] = SNR 															# Batch x 1      |  SNR (NAA_Amp / StdDev_Noise)
	finl_dict['LBL'       ] = lbl_															# Batch x 182    |  Lorentzian Line Broadening Metab/MM
	finl_dict['LBG'       ] = glb															# Batch x 182    |  Gaussian   Line Broadening Metab/MM
	finl_dict['m_mult'    ] = m_mult														# Batch x 2      |  Norm Metabolite (Metab --> 1   )
	finl_dict['w_mult'    ] = w_mult														# Batch x 2      |  Norm Water      (Water --> 5-20) Correctly Scales Water Relative to Metab
	finl_dict['LBL_Water' ] = water_lbl														# Batch x 5      |  Lorentzian Line Broadening Water
	finl_dict['LBG_water' ] = water_glb												    	# Batch x 5      |  Gaussian   Line Broadening Water
	finl_dict['FWHM_MM'   ] = MMFWHM														# Batch x 14     |  Target FWHM of Macromolecules (14 Macromolecules)
	finl_dict['FWHM_Metab'] = MetabFWHM														# Batch x 1      |  Target FWHM of Metaboites (FWHM of NAA)
	finl_dict['Healthy'   ] = clssfier0 												    # Batch x 1      |  Healthy = 0; Clinical = 1		
	finl_dict['Clinical'  ] = pop  															# Batch x 1      |  Healthy = 0; Clinical > 0 (See Clin_Names)
	finl_dict['Clin_Names'] = clin_grps  													# List (21)      |  Corresponding Names of Population Number from 'Clinical' | Note* 0 is Healthy
	finl_dict['Drop_Sig'  ] = drop_sig 														# Batch x 4096   |  Some/All Metab/MM Signal to be Subtracted (See Batch_Drop and dIdx_Drop)
	finl_dict['Batch_Drop'] = batch_drop_choice 											# Batch x 1      |  Randomly Leave Off Some/All Metabolites/Macromolecules
	finl_dict['dIdx_Drop' ] = batch_drop_didx 												# Batch x 1      |  Randomly Leave Off Some/All Metabolites/Macromolecules

	print('\n\n\n')
	# print('Did not Save !!!!  ***')
	# print('Did not Save !!!!  ***')
	
	np.savez('general_data/{}'.format(fname), **finl_dict) 									# ec2 Directory Save

if __name__ == '__main__':
	
	fpath     = os.path.realpath(__file__) 												    # Base Directory
	print(fpath)
	# basedir   = '/'.join(fpath.split('/')[:-1]) 											# Base Directory
	basedir = ''
	# basedir = 'C:/Users/agudm/Desktop'

	field     = np.array([2.0]) 															# Field Strength													
	# necho     = np.array([70 ]) 															# Echo Time
	
	# field     = np.array([2.80, 2.70, 2.60,             ])								# Field Strength
	# field     = np.array([2.50, 2.40, 2.30, 2.20        ])								# Field Strength

	# field     = np.array([3.00, 3.10, 2.90,             ])								# Field Strength
	# field     = np.array([2.10, 2.00, 1.90, 1.80,       ])								# Field Strength
	# field     = np.array([1.90, 1.80,       ])											# Field Strength
	
	# field     = np.array([1.50, 1.40,                   ])								# Field Strength
	# field     = np.array([1.70, 1.60, 1.50, 1.40,       ])								# Field Strength

	necho     = np.array([10, 15, 20, 25, 30, 35, 40, 45, 
	  					  50, 55, 60, 65, 70, 75, 80,     ]) 								# Echo Times
	# necho     = np.array([10, 15, 20, 25, 30, 35, 40, 45]) 								# Echo Times
	# necho     = np.array([50, 55, 60, 65, 70, 75, 80,   ]) 								# Echo Times

	for ff in range(field.shape[0]):
		for ee in range(necho.shape[0]):
			print('Field: {}  |  TE: {}'.format(field[ff], necho[ee]))
			time_00 = t0.time()
			for ii in range(4,8): # 0,8
				fname   = 'data_{:<03}_{:03d}_{:03d}.npz'.format( str(field[ff]).replace('.',''), necho[ee], ii)
				print(fname)
				fname   = '{:03d}.npz'.format( ii)

				time_01 = t0.time()
				# Gen_Data_Training(basedir, field, necho, fname, batch_size=(450), n=4096)		
				Gen_Data_Training(basedir, np.array([field[ff]]), np.array([necho[ee]]), fname, batch_size=(120), n=4096)		
				print('{:3d}/4:  {:<16} Current: {:7.2f}  | Total {:10.2f}'.format(ii+1, fname, t0.time() - time_01, t0.time() - time_00))

			print('--'*20)
			print('\n')