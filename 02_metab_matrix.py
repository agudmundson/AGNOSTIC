__author__ = 'Aaron Gudmundson'
__email__  = 'agudmund@uci.edu'

from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt																# Create Figures
import scipy.ndimage as ndimage																# Image Manipulation
import scipy.linalg as linalg
import scipy.sparse as sparse 																# Sparse Matrices
import scipy.stats as stats																	# Statistics
import scipy.signal as sig 																	# Signal Processing
import pandas as pd																			# Data Frames
import numpy as np																			# Arrays/Linear Algebra
import time as t0                                                                           # Determine Run Time
import subprocess																			# Terminal Control
import random
import copy 																				# DeepCopy Class Arrays
import glob																					# Bash Commands
import os																					# Interact w/Operating System

np.set_printoptions(threshold=np.inf, precision=2, linewidth=300, suppress=True)			# Set Numpy Display Params
np.seterr(divide='ignore', invalid='ignore')												# Set Numpy Operation Params
pd.set_option('display.width', 1000)														# Set Pandas Display Params

import metab_dict                                                                           # Metabolites

if __name__ == '__main__':

	fpath    = os.path.realpath(__file__) 													# Base Directory
	basedir  = '/'.join(fpath.split('/')[:-1]) 												# Base Directory

	N        = 4096

	m_dict   = metab_dict.metab_dict 														# Get Metabolite Dictionary
	m_list   = sorted(list(m_dict.keys())) 													# List of Metablites
	m_list.remove('Doub') 																	# Remove Generic Doublet
	m_list.remove('Trip') 																	# Remove Generic Triplet
	m_list.remove('Quar') 																	# Remove Generic Quartet
	m_list.remove('Cho' ) 																	# Remove Choline, since we have GPC and PCho

	print('Number of Simulations Run: {}'.format(len(m_list))) 								# Number of Simulations Run

	field    = np.arange(1.4, 3.11, .1)
	necho    = np.arange(10 , 81  ,  5)
	delay    = np.array([.00003125]) * 0													# Length of Delay Between Last Refocusing Pulse and Acquisition
	delay_   = '{:.8f}'.format(delay[0]) 													# For Ease in Filenames: String of Delay Time
	delay_   = 'delay-{}sec'.format(delay_[2:]) 											# For Ease in Filenames: String of Delay Time

	names    = [] 																			# Metabolite Names
	ncount   = 0 																			# Starting Metabolite Count

	for kk in range(len(m_list)): 															# Determine Number of Spins to go into final Array Below
		metab   = m_list[kk   ] 															# Get Current Metab
		minfo   = m_dict[metab] 															# Get Metabolite Info from metab_dict
		
		mdir    = '{}/simulated/B0-140_TE-010'.format(basedir) 								# Metabolite Directory - Doesn't matter which is used - just need something as a reference.
		plchldr = 'B0-140_TE-010_sw-3733_pb-009_delay-00000000sec.npy' 						# Metabolite Filename  - Doesn't matter which is used - just need something as a reference.
		
		marry   = np.load('{}/{}_{}'.format(mdir, metab, plchldr)) 							# Load the Reference
		
		omegas  = minfo['omega'] 															# ppm for Current Metabolite Spins
		oset    = list(set(omegas))				 											# Just Unique Spins

		for ll in range(len(oset)): 														# Iterate Over Spins
			if oset[ll] < 8.0 and oset[ll] > 0.0: 											# Exclude Spins outside typical range

				ncount  +=1 																# Keep Count of Unique Spins 
				if metab!= 'H2O': 															# Remove Simulation Index for Current Spin
					metab_ = ''.join([mm for mm in metab if not mm.isdigit()]) 				# 
				else: 																		# Ignore non-index number in H2O
					metab_ = 'H2O' 			

				idxs     = np.where(omegas == oset[ll])[0] 									# Find index of Current Spin
				omega_   = np.round(omegas[idxs[0]], 2) 									# Get ppm value
				ppm_val  = str(omega_).replace('.','') 										# Convert to String
				fullname = '{}_{:<03}'.format(metab_, ppm_val) 								# Create Full Name for Each Spin
				names.append(fullname)

	print('Number of Unique Spins   : {:3d}'.format(len(names))) 							# Display Total Number of Unique Spins

	for ii in range(necho.shape[0]):
	# for ii in range(1):
		te       = necho[ii]/1000 															# Echo Time 
		te2      = te /2 																	# Time Between Pulses
		te1      = te2/2 																	# Time to First Pulse
		te3      = te2/2 																	# Time to Acquisition

		for jj in range(field.shape[0]): 													# Iterate over Field Strengths
		# for jj in range(16,17): 															# Iterate over Field Strengths

			m    = np.zeros([N+300, ncount, 3], dtype=np.complex_) 							# Array for Time-Domain Data
			sws  = np.zeros([3]) 															# Note & Save the SW for each of the following
			ssamp= np.zeros([3])

			for ss in range(2,5):
				s        = copy.deepcopy(ss) - 2
				N_       = N * ss

				larmor   = field[jj] * 42.576 												# Larmor Frequency
				if field[jj] == 3.0: 														# 3.0T Bandwidth of 8000
					sw   = 8000
				else:
					sw   = larmor * 62.633095 												# Bandwidth Equivalent to the ~31ppm at 3.0T

				sw       = int(np.round(sw, 0)) 											# Rounded to Integer
				dwell    = 1/sw 															# Dwell Time
				dwell   *= ss 																# Subsampling

				sw_targ  = 1/dwell 															# Subsampled Dwell Time
				sws[s]   = copy.deepcopy(sw_targ) 											# Save each SW used
				ssamp[s] = copy.deepcopy(s)

				pb       = int(te3/ (1/sw) )												# Number of Points Possible Before Echo
				pb_strt  = 300 - pb 														# Determined Max 300 points (Hard-Coded Array Length) given TE=150ms

				necho_   = 'TE-{:03d}'.format(necho[ii]) 									# Define for Ease in Filenames: String of Echo Time
				field_   = 'B0-{:<03}'.format(str(field[jj]).replace('.','')) 				# Define for Ease in Filenames: String of Field Strength
				field_   = field_[:6]
				sw_      = 'sw-{:4d}'.format(int(np.round( sw,0)))       					# Define for Ease in Filenames: String of Bandwidth
				pb_      = 'pb-{:03d}'.format(pb) 				      						# Define for Ease in Filenames: String of N Points Before Echo

				print('\n{:5d} ({:5d} ): {} {} {} {:8.2f} {}'.format(ss, N_, field_, necho_, sw_, sw_targ, pb_)) # Show Parameters

				# m        = np.zeros([N+300, ncount], dtype=np.complex_) 					# Array for Time-Domain Data
				nHs      = np.zeros([       ncount]) 										# Number of Spins per Nucleus
				m_cnt    = 0 																# Maintain Count
				
				ppm      = np.linspace(0, sw_targ/larmor, N) 								# Define ppm axis
				ppm     -= ppm[N//2] 														# Center ppm axis
				ppm     += 4.7 																# Center on Water at ~4.7ppm

				# Build the Main Metabolite Array 
				for kk in range(len(m_list)): 												# Iterate over Metabolites
	 	
					metab   = m_list[kk   ] 												# Current Metabolite
					minfo   = m_dict[metab] 												# Get Metabolite Information from metab_dict
					
					mdir    = '{}/simulated/{}_{}'.format(basedir, field_, necho_) 			# Simulation Directory
					
					try:
						pb_      = 'pb-{:03d}'.format(pb) 				      				# Define for Ease in Filenames: String of N Points Before Echo
						mname   = '{}_{}_{}_{}_{}_{}.npy'.format(metab, field_, necho_, sw_, pb_, delay_ )	# Filename
						marry   = np.load('{}/{}'.format(mdir, mname)) 						# Load Data
					except:
						pb_      = 'pb-{:03d}'.format(pb+1) 				      			# Define for Ease in Filenames: String of N Points Before Echo
						mname   = '{}_{}_{}_{}_{}_{}.npy'.format(metab, field_, necho_, sw_, pb_, delay_ )	# Filename
						marry   = np.load('{}/{}'.format(mdir, mname)) 						# Load Data

					omegas  = minfo['omega'] 												# ppm for Current Metabolite Spins
					oset    = list(set(omegas))				 	 							# Just the Unique Spins
					hnuclei = np.where(omegas != 0.0)[0] 									# Remove X-Nuclei
					hgh_lim = np.where(omegas >= 8.0)[0] 									# Remove Signals above 8.0ppm
					low_lim = np.where(omegas <= 0.0)[0] 									# Remove Signals Below 0.00ppm

					if metab!= 'H2O': 														# Rename with Name and ppm of Spin
						metab_ = ''.join([mm for mm in metab if not mm.isdigit()]) 			# Remove Simulation Index (Cre1, Cre2, etc.) from splitting up large simulations 
					else:
						metab_ = 'H2O' 			 											# Handle H2O which has non-index number in string

					for ll in range(len(oset)): 											# Iterative over Unique Spins
						if oset[ll] < 8.0 and oset[ll] > 0.0:								# Exclude spins outside typical range

							idxs            = np.where(omegas == oset[ll])[0] 				# Index for Current Spin
							nHs[ m_cnt]     = minfo['nH'][idxs[0]]							# Note total Equivalent Nuclei 

							ppm_val         = np.round(oset[ll], 2) 						# Get ppm Value
							ppm_val         = str(ppm_val).replace('.','') 					# Convert to String
							fullname        = '{}_{:<03}'.format(metab_, ppm_val) 			# Create Full Name for each Spin

							m_temp          = np.sum(marry[:,idxs], axis=1) 				# Time-Domain Signal for Sum of Equivalent Nuclei that were Simulated
							m_temp          = m_temp[:N_+300] 								# Number of Points to Reach 4096 After Subsampling

							m[:300,m_cnt,s] = copy.deepcopy(m_temp[:300]) 					# Skip Subsampling 1st 300 Points
							m[300:,m_cnt,s] = copy.deepcopy(m_temp[300:N_+300:ss]) 			# Subsampling from 300:End
							# m[:, m_cnt]  = np.sum(marry[:,idxs], axis=1)                 	# 
							m[:, m_cnt,s]  *= nHs[m_cnt]									# Multiply to Account for Equivalent Nuclei that were not Explicitly Simulated
							nHs[ m_cnt]    *= idxs.shape[0] 								# Multiply to Account for Equivalent Nuclei (Non-Modeled multiplies; All Modeled just multplies by 1)

							if fullname    != names[m_cnt]: 								# Confirm Reference Name = Current Name
								print('{:3d} {:<10} {:<10} nH = {:2.0f} {:2.0f} | '.format(m_cnt, names[m_cnt], fullname, nHs[ m_cnt], nHs[ m_cnt]), '{:.3f}'.format(oset[ll]), '  ', idxs.shape, idxs, '   *****\n\n')

							m_cnt          +=1                								# Add 1 to Metabolite Count
				
				phs_1    = {'1.4': 1.75, '1.5': 0.0, '1.6': 4.50, 							# Discovered Slight 1st Order Phase for some B0
							'1.7': 1.75, '1.8': 0.0, '1.9': 4.25,							# Resulting Corrections in Degrees
							'2.0': 1.75, '2.1': 0.0, '2.2': 4.00,	
							'2.3': 1.75, '2.4': 0.0, '2.5': 3.75,
							'2.6': 1.75, '2.7': 0.0, '2.8': 3.50,
							'2.9': 1.75, '3.0': 0.0, '3.1': 3.25,}

				phs_1_   = phs_1[str(np.round(field[jj], 1))] 								# Read Dictionary Value for 1st Order Phase Correction

				## With the Main Array Built - Ensure the integral of 1H = 1 for All Metabolites
				time     = np.linspace(0, dwell * N, N ) 									# Time
				apod     = np.exp(-3.125 * time) 											# Apodize to Correct Phase/Shape for Integral

				cre_fid  = copy.deepcopy(m[300:, 15, s]) * apod 							# Use single 1H of Cre_665 as a Reference (Could have used any signal)
				cre_665  = np.fft.fftshift(np.fft.fft(cre_fid)) 							# Use single 1H of Cre_665 as a Reference (Could have used any signal)
				cre_665 *= np.exp(1j * (ppm - 4.70) * phs_1_ * np.pi/180)					# 1st Order Phase Correction
				cre_665  = np.real(cre_665) 										
				cre_665  = np.abs(cre_665)

				xdist    = np.linspace(ppm[0], ppm[-1], ppm.shape[0]) 						# Define the xaxis for intergrals
				area_665 = np.abs(np.trapz(cre_665 , xdist)) 								# Integrate for Cre_665
				
				# h1       = area_665 														# Set Cre_665 Area to 1
				h1       = np.array([124]) 													# Cre_665 generally equals ~124 - HardCode for Consistency Across Basis Sets

				for kk in range(m.shape[1]): 												# Iterate over the entire Array

					fid        = copy.deepcopy(m[300:,kk,s]) 								# FID
					fid       *= apod 														# Min Decay (Based on NAA T2 Max of 320ms  or equivalent rate of 3.125)
					spec       = np.fft.fftshift(np.fft.fft(fid))							# Get Spectrum of Current Spin (300 is Top of the Echo)
					spec      *= np.exp(1j * (ppm - 4.70) * phs_1_ * np.pi/180)				# 1st Order Phase Correction
					spec       = np.real(spec)
					spec       = np.abs(spec) 												# Use Absolute Value to Correct for Coupling
					area       = np.trapz(spec, xdist)  									# Get the Area
					orig_area  = copy.deepcopy(area)

					a          = h1 * nHs[kk] 												# Multiply the Number of Duplicate Spins not Simulated
					a          = a / area 													# Area per Proton
					
					m[:,kk,s] *= a 															# Multiply by Ratio
					m[:,kk,s] /= h1 														# Normalize Proton to 1

					fid        = copy.deepcopy(m[300:,kk,s]) * apod 						# FID
					spec       = np.fft.fftshift(np.fft.fft(fid)) 							# Confirm Integral = 1
					spec      *= np.exp(1j * (ppm - 4.70) * phs_1_ * np.pi/180)				# 1st Order Phase Correction
					spec       = np.real(spec)
					spec       = np.abs(spec) 												# Use Absolute Value to Correct for Coupling
					area       = np.trapz(spec, xdist) 										# Get the Area

					if kk in [0, 3, 4, 14, 15, 16, 78, 97, 98, 117, 135, 136, 138]: 		# Test Cases to Confirm Integral
						if names[kk] not in ['ATP_613', 'ATP_676', 'Hist_298', 'Hist_329', 'Hist_299', 'Hist_709', 'Hist_785', 'Hist_750']:
							print('{:<10} {:3.0f}: {:11.6f}  |  {:11.6f}  |  {:11.6f}  |  {:11.6f}       '.format(names[kk], nHs[kk], area, area_665, orig_area, h1[0]))
					
					if nHs[kk] !=  np.round(area, 3): 										# Error if 1H does NOT equal 1
						if names[kk] not in ['ATP_613', 'ATP_676', 'Hist_298', 'Hist_329', 'Hist_299', 'Hist_709', 'Hist_785', 'Hist_750']:
							print('{:<10} {:3.0f}: {:11.6f}  |  {:11.6f}  |  {:11.6f}  |  {:11.6f}   ***     ***     ***     ***     ***  '.format(names[kk], nHs[kk], area, area_665, orig_area, h1[0]))
					# else:
					# 	print('{:<10} {:3.0f}: {:8.12f}       '.format(names[kk], nHs[kk], area))

				m_   = {
						'Field'     : field[jj], 												# Field Strength
	            		'TE'        : necho[ii], 												# Echo Time
	            		'SW'        : sws      , 												# Spectral Widths
					    'Subsample' : ssamp    , 												# SubSampling
					    'data'      : m        , 												# Array of Time-Domain Signals
					    'names'     : names    , 												# List of Nuclei Names
					    'nHs'       : nHs      , 												# Number of Equivalent Nuclei
					    'norm_const': h1       , 												# Normalizing Constant such that Integral(1H) = 1.00 
					   }

				# print('{}/basis_no_decay/data_{}_{}_{:7.2f}_{}_{}.npz\n\n'.format(basedir, field_, necho_, sw_targ, pb_, delay_))
				np.savez('{}/basis_norm/data_{}_{}_{}_{}_{}.npz'.format(basedir, field_, necho_, sw_, pb_, delay_), **m_)