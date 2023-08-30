

import pandas as pd                                                                     	# DataFrames
import numpy as np                                                                      	# Arrays/LinAlg
import time as t0                                                                       	# Determine Run Time
import copy                                                                             	# Copy Arrays
import sys                                                                              	# Interact with System Files

def add_OOV(larmor, time):
	N        = time.shape[0] 																# Number of Examples
	p        = np.random.choice(np.array([0,1]), size=N, p=np.array([.15, .85])) 			# Probability of OOV

	tpoint   = np.random.uniform(.010,  .400, N) 											# Center of OOV
	times_   = copy.deepcopy(time) - tpoint[:,None] 										# Time Relative to OOV Center

	width    = np.random.randint(500,  8000, N).astype(np.uint16) 							# Width of OOV
	ppm_loc  = np.random.uniform(  1,     4, N) 											# ppm Location of OOV
	amp      = np.random.uniform(3e-4,15e-4, N) 											# Amplitude (Arbitrary)
	amp      = np.random.uniform( .05,  .15, N) 											# Amplitude (Arbitrary)

	echo     = np.ones([N, 4096], dtype=np.complex_) 										# OOV - Instantiate Array
	echo    *= p[:,None] 																	# OOV - Probability of Occurring
	echo    *= np.exp(-1  * width[:,None] * (times_**2) ) 									# OOV - Gaussian Envelope
	echo    *= np.exp(-2j * time * np.pi * larmor[:,None] * (4.7 - ppm_loc[:,None]))		# OOV - Shift to ppm Location
	echo    *= np.sqrt(width[:,None]) 														# OOV - Ensure Constant Width
	echo    /= np.max(echo)  																# OOV - Normalize Height (for Ease of Amp)
	echo    *= amp[:,None] 																	# OOV - Amplitude Factor

	frq_hz   = (larmor[:,None] * (4.7 - ppm_loc[:,None])) 									# OOV - Frequency in Hz
	echo     = {'echo'  : echo   , 															# OOV array
				'tpoint': tpoint , 															# Center Point in TimeDomain 
				'width' : width  , 															# Width of OOV
				'loc'   : ppm_loc, 															# Location of OOV
				'frq'   : frq_hz , 															# Frequency of OOV
				'amp'   : amp    , 															# Amplitude of OOV
				'prob'  : p      }                                                          # Probability of OOV

	return echo

def build_mask(oov, p):

	mask  = np.zeros(oov.shape) 															# Instantiate Mask
	abbr  = np.zeros([oov.shape[0], 256, 2, 1]) 											# Abbreviated Echo
	lidx  = np.zeros(oov.shape) 															# Instantiate Left Index
	ridx  = np.zeros(oov.shape) 															# Instantiate Right Index
	
	for ii in range(oov.shape[0]): 															# Iterate over Training Examples
		for jj in range(oov.shape[2]):														# Iterate over Real and Imag Component

			oov_   = oov[ii,:,jj,0] 														# Get Signal
			oov_   = np.abs(oov_) 															# Absolute Value for Positive Points

			mx_oov = np.max(oov_) 															# Max Value 

			mask_  = np.zeros([oov.shape[1]]) 												# Initiate Mask Array
			mask_[oov_ > (mx_oov * .05)] = 1 												# Mask Region

			lft    = np.argmax(mask_[:: 1] == 1) 											# Left Index for Mask
			rgt    = np.argmax(mask_[::-1] == 1)  											# Right Index for Mask (reversed)
			rgt    = oov.shape[1] - rgt + 1													# Right Index for Mask

			if p[ii] == 1:
				mask[ii, lft:rgt, jj, 0] = 1												# Add to Mask
				
	return mask, abbr

def preprocess_freq(fname, bsize=120):
	
	# Distance to Match Frequency Across Field Strengths
	cre_idx   = 3.0255789576 																# Frequency Adjust such that all Spectra across acquisition schemes are aligned
	match_frq = {
			'1.4_2': 3.02525064225495,'1.4_3': 3.02540991466177,'1.4_4': 3.02525064225495, 	# 1.4 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'1.5_2': 3.02557895759664,'1.5_3': 3.02557895759664,'1.5_4': 3.02557895759664, 	# 1.5 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'1.6_2': 3.02592604067203,'1.6_3': 3.02576674159892,'1.6_4': 3.02568709206236, 	# 1.6 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'1.7_2': 3.02522425204115,'1.7_3': 3.02538352695775,'1.7_4': 3.02546316441604, 	# 1.7 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'1.8_2': 3.02557895759664,'1.8_3': 3.02557895759664,'1.8_4': 3.02557895759664, 	# 1.8 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'1.9_2': 3.02546879831753,'1.9_3': 3.02578739253398,'1.9_4': 3.02570774397987, 	# 1.9 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.0_2': 3.02520577889150,'2.0_3': 3.02536505556493,'2.0_4': 3.02544469390164, 	# 2.0 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.1_2': 3.02557895759664,'2.1_3': 3.02557895759664,'2.1_4': 3.02557895759664, 	# 2.1 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.2_2': 3.02548382003741,'2.2_3': 3.02580241139585,'2.2_4': 3.02572276355624, 	# 2.2 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.3_2': 3.02566995874024,'2.3_3': 3.02535140279632,'2.3_4': 3.02543104178230, 	# 2.3 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.4_2': 3.02557895759664,'2.4_3': 3.02557895759664,'2.4_4': 3.02557895759664, 	# 2.4 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.5_2': 3.02549523654452,'2.5_3': 3.02549523654452,'2.5_4': 3.02573417843428, 	# 2.5 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.6_2': 3.02565945860829,'2.6_3': 3.02534090066662,'2.6_4': 3.02542054015204, 	# 2.6 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.7_2': 3.02557895759664,'2.7_3': 3.02557895759664,'2.7_4': 3.02557895759664, 	# 2.7 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.8_2': 3.02550420665724,'2.8_3': 3.02550420665724,'2.8_4': 3.02574314726702, 	# 2.8 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.9_2': 3.02565113091743,'2.9_3': 3.02533257139135,'2.9_4': 3.02541221127287, 	# 2.9 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'3.0_2': 3.02557895759664,'3.0_3': 3.02557895759664,'3.0_4': 3.02557895759664, 	# 3.0 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'3.1_2': 3.02551144061912,'3.1_3': 3.02551144061912,'3.1_4': 3.02551144061912,} # 3.1 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx

	# n         = 1024
	# n_        = n * 2

	n         = 2048
	n_        = 2048
	
	## Read in Data
	data      =  np.load(fname)																# Get Current Dataset

	y_metab   = data['Metab'][:, :]															# Metabolite 
	y_H2O     = data['water'][:, :]     													# Residual Water
	y_MM      = data['MM'   ][:, :]     													# MacroMolecules
	y_noise   = data['noise'][:, :] 														# Normally Distributed Noise

	y_metab  /= data['m_mult'][:,1,None] 													# Scale Metabolite (0 or 1 is no phase or phase)
	y_MM     /= data['m_mult'][:,1,None] 													# Scale MM         (0 or 1 is no phase or phase)
	y_noise  /= data['m_mult'][:,1,None]  													# Scale Noise      (0 or 1 is no phase or phase)
	y_H2O    /= data['w_mult'][:,1,None] 													# Scale Water      (0 or 1 is no phase or phase)

	b0        = data['Field_Str'].astype(np.float32) 	  									# Field Strength (conver to float)	
	b0       /= 10 																			# Divide by 10 (Currently 14-31)

	larmor    = b0 * 42.5760 																# Larmor Frequency
	sub       = data['subsample'] 															# Subsampling to Achieve Spectral Width

	phase     = data['phase0'    ][:,None] 													# 0th Order Phase Shift
	freq      = data['freq_shift'][:,None] 													# Frequency Shift
	time      = data['time'][:, :] 															# Time Axis

	data.close() 																			# Close  Data File
	del data 																				# Delete Data Variable to Manage Memory Usage

	# Add OOV Echo
	oov       = add_OOV(larmor, time) 														# Create OOVs
	y_echo    = oov['echo'] 																# OOV Array
	p_echo    = oov['prob'] 																# OOV Probabilty (80% Occurrence)
	params    = np.zeros([bsize, 3])                                                		# OOV Parameters
	params[:,0] = oov['width']                                                      		# Width
	params[:,1] = oov['frq'  ][:,0]                                                      	# Frequency (Hz)
	params[:,2] = oov['amp'  ]                                                      		# Amplitude

																							# Input:
	x_        = (  copy.deepcopy(y_metab) 													# Metaboite
				 + copy.deepcopy(y_MM   )													# Macromolecule
				 + copy.deepcopy(y_noise)													# Noise
				 + copy.deepcopy(y_H2O  )                           						# Water
				 + copy.deepcopy(y_echo ))                          						# OOV Echo


	# Frequency Match Across Field Strengths
	for ii in range(bsize): 																# Shift for Equivalent 3.027 CreCH3 Location
		name  = '{:3.1f}_{:1d}'.format(b0[ii], int(sub[ii]))  								# Match Name and Frequency Distance Needded for Correction
		
		x_[ii,:]      *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))
		y_metab[ii,:] *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))
		y_H2O[  ii,:] *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))
		y_MM[   ii,:] *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))
		y_noise[ii,:] *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))
		y_echo[ ii,:] *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))


	# Phase and Frequency Shifts
	x_       *= np.exp(-1j * phase              )  											# Phase Shift
	x_       *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift

	y_metab  *= np.exp(-1j * phase              )   										# Phase Shift 									
	y_metab  *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift

	y_H2O    *= np.exp(-1j * phase              )   										# Phase Shift 									
	y_H2O    *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift

	y_MM     *= np.exp(-1j * phase              )   										# Phase Shift 									
	y_MM     *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift

	y_noise  *= np.exp(-1j * phase              )   										# Phase Shift 									
	y_noise  *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift

	y_echo   *= np.exp(-1j * phase              )   										# Phase Shift 									
	y_echo   *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift


	# Instantiate Final Arrays
	x         = np.zeros([bsize, n , 2, 1]) 												# Final Input
	y_metab_  = np.zeros([bsize, n_, 2, 1]) 												# Final Metabolite
	y_H2O_    = np.zeros([bsize, n_, 2, 1]) 												# Final Water
	y_MM_     = np.zeros([bsize, n_, 2, 1]) 												# Final Macromolecule
	y_noise_  = np.zeros([bsize, n_, 2, 1]) 												# Final Noise
	y_echo_   = np.zeros([bsize, n_, 2, 1]) 												# Final Echo


	# Fourier Transform
	for ii in range(bsize):
		x_tmp     = np.fft.fftshift(np.fft.fft( x_[     ii,:n ] ))  						# Fourier Transform w/n  points
		y_met_tmp = np.fft.fftshift(np.fft.fft( y_metab[ii,:n_] ))  						# Fourier Transform w/n_ points
		y_H2O_tmp = np.fft.fftshift(np.fft.fft( y_H2O[  ii,:n_] ))  						# Fourier Transform w/n_ points
		y_MM_tmp  = np.fft.fftshift(np.fft.fft( y_MM[   ii,:n_] ))  						# Fourier Transform w/n_ points
		y_nse_tmp = np.fft.fftshift(np.fft.fft( y_noise[ii,:n_] ))  						# Fourier Transform w/n_ points
		y_ech_tmp = np.fft.fftshift(np.fft.fft( y_echo[ ii,:n_] ))  						# Fourier Transform w/n_ points

		## Final Arrays
		x[ii,:,0,0]        = np.real( x_tmp[    :n ]   )									# Real Component
		x[ii,:,1,0]        = np.imag( x_tmp[    :n ]   )									# Imag Component

		y_metab_[ii,:,0,0] = np.real( y_met_tmp[:n_] )										# Real Component
		y_metab_[ii,:,1,0] = np.imag( y_met_tmp[:n_] )										# Imag Component

		y_H2O_[  ii,:,0,0] = np.real( y_H2O_tmp[:n_] )										# Real Component
		y_H2O_[  ii,:,1,0] = np.imag( y_H2O_tmp[:n_] )										# Imag Component

		y_MM_[   ii,:,0,0] = np.real( y_MM_tmp[ :n_] )										# Real Component
		y_MM_[   ii,:,1,0] = np.imag( y_MM_tmp[ :n_] )										# Imag Component

		y_noise_[ii,:,0,0] = np.real( y_nse_tmp[:n_] )										# Real Component
		y_noise_[ii,:,1,0] = np.imag( y_nse_tmp[:n_] )										# Imag Component

		y_echo_[ ii,:,0,0] = np.real( y_ech_tmp[:n_] )										# Real Component
		y_echo_[ ii,:,1,0] = np.imag( y_ech_tmp[:n_] )										# Imag Component


	## Scaling
	xmax              = np.max(np.max(np.abs(x),axis=1),axis=1) 							# Max X Value
	x                /= xmax[:,None,None] 													# Norm to Max X
	y_metab_         /= xmax[:,None,None] 													# Norm to Max X
	y_H2O_           /= xmax[:,None,None] 													# Norm to Max X
	y_MM_            /= xmax[:,None,None] 													# Norm to Max X
	y_noise_         /= xmax[:,None,None] 													# Norm to Max X
	y_echo_          /= xmax[:,None,None] 													# Norm to Max X

	mask, abbr        = build_mask(y_echo_, p_echo) 										# Mask/Abbreviated OOV

	return x, y_metab_, y_H2O_, y_MM_, y_noise_, y_echo_, mask, abbr, params

def preprocess_time(fname, bsize=120):
	
	# Distance to Match Frequency Across Field Strengths
	cre_idx   = 3.0255789576 																# Frequency Adjust such that all Spectra across acquisition schemes are aligned
	match_frq = {
			'1.4_2': 3.02525064225495,'1.4_3': 3.02540991466177,'1.4_4': 3.02525064225495, 	# 1.4 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'1.5_2': 3.02557895759664,'1.5_3': 3.02557895759664,'1.5_4': 3.02557895759664, 	# 1.5 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'1.6_2': 3.02592604067203,'1.6_3': 3.02576674159892,'1.6_4': 3.02568709206236, 	# 1.6 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'1.7_2': 3.02522425204115,'1.7_3': 3.02538352695775,'1.7_4': 3.02546316441604, 	# 1.7 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'1.8_2': 3.02557895759664,'1.8_3': 3.02557895759664,'1.8_4': 3.02557895759664, 	# 1.8 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'1.9_2': 3.02546879831753,'1.9_3': 3.02578739253398,'1.9_4': 3.02570774397987, 	# 1.9 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.0_2': 3.02520577889150,'2.0_3': 3.02536505556493,'2.0_4': 3.02544469390164, 	# 2.0 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.1_2': 3.02557895759664,'2.1_3': 3.02557895759664,'2.1_4': 3.02557895759664, 	# 2.1 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.2_2': 3.02548382003741,'2.2_3': 3.02580241139585,'2.2_4': 3.02572276355624, 	# 2.2 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.3_2': 3.02566995874024,'2.3_3': 3.02535140279632,'2.3_4': 3.02543104178230, 	# 2.3 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.4_2': 3.02557895759664,'2.4_3': 3.02557895759664,'2.4_4': 3.02557895759664, 	# 2.4 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.5_2': 3.02549523654452,'2.5_3': 3.02549523654452,'2.5_4': 3.02573417843428, 	# 2.5 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.6_2': 3.02565945860829,'2.6_3': 3.02534090066662,'2.6_4': 3.02542054015204, 	# 2.6 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.7_2': 3.02557895759664,'2.7_3': 3.02557895759664,'2.7_4': 3.02557895759664, 	# 2.7 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.8_2': 3.02550420665724,'2.8_3': 3.02550420665724,'2.8_4': 3.02574314726702, 	# 2.8 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'2.9_2': 3.02565113091743,'2.9_3': 3.02533257139135,'2.9_4': 3.02541221127287, 	# 2.9 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'3.0_2': 3.02557895759664,'3.0_3': 3.02557895759664,'3.0_4': 3.02557895759664, 	# 3.0 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx
			'3.1_2': 3.02551144061912,'3.1_3': 3.02551144061912,'3.1_4': 3.02551144061912,} # 3.1 Tesla Closest CreCH3 ppm location --> Frequency Adjust to Center on cre_idx

	# n         = 1024 																		# Input Size
	# n_        = n * 2 																	# Output Size
	
	n         = 2048 																		# Input Size
	n_        = 2048 																		# Output Size

	## Read in Data
	data      =  np.load(fname)																# Get Current Dataset

	y_metab   = data['Metab'][:, :]															# Metabolite 
	y_H2O     = data['water'][:, :]     													# Residual Water
	y_MM      = data['MM'   ][:, :]     													# MacroMolecules
	y_noise   = data['noise'][:, :] 														# Normally Distributed Noise

	y_metab  /= data['m_mult'][:,1,None] 													# Scale Metabolite (0 or 1 is no phase or phase)
	y_MM     /= data['m_mult'][:,1,None] 													# Scale MM         (0 or 1 is no phase or phase)
	y_noise  /= data['m_mult'][:,1,None]  													# Scale Noise      (0 or 1 is no phase or phase)
	y_H2O    /= data['w_mult'][:,1,None] 													# Scale Water      (0 or 1 is no phase or phase)

	b0        = data['Field_Str'].astype(np.float32) 	  									# Field Strength (convert to float)	
	b0       /= 10 																			# Divide by 10 (Currently 14-31)

	larmor    = b0 * 42.5760 																# Larmor Frequency
	sub       = data['subsample'] 															# Subsampling to Achieve Spectral Width

	phase     = data['phase0'    ][:,None] 													# 0th Order Phase Shift
	freq      = data['freq_shift'][:,None] 													# Frequency Shift
	time      = data['time'][:, :] 															# Time Axis

	data.close() 																			# Close  Data File
	del data 																				# Delete Data Variable to Manage Memory Usage

	oov       = add_OOV(larmor, time) 														# Create OOVs
	y_echo    = oov['echo'] 																# OOV Array
	p_echo    = oov['prob'] 																# OOV Probabilty (80% Occurrence)
	params    = np.zeros([bsize, 3])                                                		# OOV Parameters
	params[:,0] = oov['width']                                                      		# Width
	params[:,1] = oov['frq'  ][:,0]                                                      	# Frequency (Hz)
	params[:,2] = oov['amp'  ]                                                      		# Amplitude

																							# Input:
	x_        = (  copy.deepcopy(y_metab) 													# Metaboite
				 + copy.deepcopy(y_MM   )													# Macromolecule
				 + copy.deepcopy(y_noise)													# Noise
				 + copy.deepcopy(y_H2O  )                           						# Water
				 + copy.deepcopy(y_echo ))                          						# OOV

	# Frequency Match Across Field Strengths
	for ii in range(bsize): 																# Shift for Equivalent 3.027 CreCH3 Location
		name  = '{:3.1f}_{:1d}'.format(b0[ii], int(sub[ii])) 
		
		x_[ii,:]      *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))
		y_metab[ii,:] *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))
		y_H2O[  ii,:] *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))
		y_MM[   ii,:] *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))
		y_noise[ii,:] *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))
		y_echo[ii,:]  *= np.exp(-2j * time[ii,:] * np.pi * (larmor[ii] * (match_frq[name] - cre_idx)))

	# Phase and Frequency Shifts
	x_       *= np.exp(-1j * phase              )  											# Phase Shift
	x_       *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift

	y_metab  *= np.exp(-1j * phase              )   										# Phase Shift 									
	y_metab  *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift

	y_H2O    *= np.exp(-1j * phase              )   										# Phase Shift 									
	y_H2O    *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift

	y_MM     *= np.exp(-1j * phase              )   										# Phase Shift 									
	y_MM     *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift

	y_noise  *= np.exp(-1j * phase              )   										# Phase Shift 									
	y_noise  *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift

	y_echo   *= np.exp(-1j * phase              )   										# Phase Shift 									
	y_echo   *= np.exp( 2j * freq * time * np.pi) 											# Frequency Shift


	# Final Arrays
	x                 = np.zeros([bsize, n, 2, 1]) 											# Final Input
	x[:,:,0,0]        = np.real(x_[:,:n])													# Real Component
	x[:,:,1,0]        = np.imag(x_[:,:n])													# Imag Component

	y_metab_          = np.zeros([bsize, n_, 2, 1]) 										# Final Metabolite
	y_metab_[:,:,0,0] = np.real(y_metab[:, :n_])											# Real Component
	y_metab_[:,:,1,0] = np.imag(y_metab[:, :n_])											# Imag Component

	y_H2O_            = np.zeros([bsize, n_, 2, 1]) 										# Final Water
	y_H2O_[:,:,0,0]   = np.real(y_H2O[:, :n_])												# Real Component
	y_H2O_[:,:,1,0]   = np.imag(y_H2O[:, :n_])												# Imag Component

	y_MM_             = np.zeros([bsize, n_, 2, 1]) 										# Final Macromolecule
	y_MM_[:,:,0,0]    = np.real(y_MM[:, :n_])												# Real Component
	y_MM_[:,:,1,0]    = np.imag(y_MM[:, :n_])												# Imag Component

	y_noise_          = np.zeros([bsize, n_, 2, 1]) 										# Final Noise
	y_noise_[:,:,0,0] = np.real(y_noise[:, :n_])											# Real Component
	y_noise_[:,:,1,0] = np.imag(y_noise[:, :n_])											# Imag Component

	y_echo_           = np.zeros([bsize, n_, 2, 1]) 										# Final OOV echo
	y_echo_[:,:,0,0]  = np.real(y_echo[:, :n_])												# Real Component
	y_echo_[:,:,1,0]  = np.imag(y_echo[:, :n_])												# Imag Component


	## Scaling
	xmax              = np.max(np.max(np.abs(x),axis=1),axis=1) 							# Max X Value
	x                /= xmax[:,None,None] 													# Norm to Max X
	y_metab_         /= xmax[:,None,None] 													# Norm to Max X
	y_H2O_           /= xmax[:,None,None] 													# Norm to Max X
	y_MM_            /= xmax[:,None,None] 													# Norm to Max X
	y_noise_         /= xmax[:,None,None] 													# Norm to Max X
	y_echo_          /= xmax[:,None,None] 													# Norm to Max X

	mask, abbr        = build_mask(y_echo_, p_echo) 										# Mask/Abbreviated OOV

	return x, y_metab_, y_H2O_, y_MM_, y_noise_, y_echo_, mask, abbr, params


if __name__ == '__main__':

	domain   = 'Freq'

	basedir  = '/datapool/home/agudmund/main/deep'
	traindir = '{}/training_data/AG'.format(basedir)
	iodir    = '{}/InputOutput/AG_CNN_{}'.format(basedir, domain)
	dataset  = 'data_AG'

	nfids    =  120 																		# FIDs per file
	start    =    0 																		# Start (  0,  540, 1080, 1620)
	stride   =   60 																		# Number of Files to include in each Output
	end      =  540 																		# End   (540, 1080, 1620, 2160)
	
	for ii in range(start, end, stride): 													# Iterate over Filenames from Start-->Finish
		
		sidx  = ii  																		# Save Index
		sname = '{}/{}_{:03d}_{}.npz'.format(iodir, dataset, ii//60, domain) 				# Save Name
		print('{:04d}:  {:03d}  |  {}'.format(ii, sidx, sname)) 							# Display Current

		x     = np.zeros([stride*nfids, 2048, 2, 1])	 									# Input (All Combined)
		# x     = np.zeros([stride*nfids, 1024, 2, 1])	 									# Input (All Combined)
		metab = np.zeros([stride*nfids, 2048, 2, 1]) 										# Metabolite
		water = np.zeros([stride*nfids, 2048, 2, 1])  										# Water
		mm    = np.zeros([stride*nfids, 2048, 2, 1])  										# Macromolecule
		noise = np.zeros([stride*nfids, 2048, 2, 1])  										# Noise
		oov   = np.zeros([stride*nfids, 2048, 2, 1])  										# OOV Echo
		mask  = np.zeros([stride*nfids, 2048, 2, 1])  										# Mask
		abbr  = np.zeros([stride*nfids,  256, 2, 1])  										# Abbreviated OOV
		params= np.zeros([stride*nfids,    3,     ])  										# OOV Simulation Parameters

		for jj in range(stride): 															# Separate Stride Number of Files per Output

			fname = '{}/{}_{:03d}.npz'.format(traindir, dataset, ii + jj) 					# Current Filename
			print('    ({}) {:04d}:  {:03d}  |  {}'.format(domain, ii, ii + jj, fname))		# Display Current

			if domain == 'Time':
				data  = preprocess_time(fname) 												# Iterate Validation Generator
			elif domain == 'Freq':
				data  = preprocess_freq(fname) 												# Iterate Validation Generator
			
			idx   = jj * nfids
			x[     idx:idx+nfids, :, :, :] = data[0] 										# Full Array
			metab[ idx:idx+nfids, :, :, :] = data[1] 										# Full Array
			water[ idx:idx+nfids, :, :, :] = data[2] 										# Full Array
			mm[    idx:idx+nfids, :, :, :] = data[3] 										# Full Array
			noise[ idx:idx+nfids, :, :, :] = data[4] 										# Full Array
			oov[   idx:idx+nfids, :, :, :] = data[5] 										# Full Array
			mask[  idx:idx+nfids, :, :, :] = data[6] 										# Full Array
			abbr[  idx:idx+nfids, :, :, :] = data[7] 										# Full Array
			params[idx:idx+nfids, :      ] = data[8] 										# Full Array

		## Output Dictionary
		data_dict = {'x'    : x     ,    													# Combined Input
					 'Metab': metab , 														# Metabolite Data
					 'Water': water , 														# Residual Water
					 'MM'   : mm    , 														# Macromolecules
					 'Noise': noise , 														# Noise
					 'OOV'  : oov   , 														# Out of Voxel Echo
					 'mask' : mask  , 														# Out of Voxel Mask
					 'abbr' : abbr  , 														# Out of Voxel Abbreviated/Amplified
					 'param': params} 														# Out of Voxel Simulation Parameters

		np.savez(sname, **data_dict) 														# Save as archived numpy file
