
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

def randomize(basedir):

	gendir    = '{}/generalize_data/AG'.format(basedir)
	trndir    = '{}/training_data/AG'.format(basedir)

	fields    = ['140', 
				 '150', '160', '170', '180', '190', 
				 '200', '210', '220', '230', '240', 
				 '250', '260', '270', '280', '290',
				 '300', '310' 					  ]
	nechos    = ['010', '015', '020', '025', '030', '035', '040', '045',
				 '050', '055', '060', '065', '070', '075', '080', ]

	ncond     = len(fields) * len(nechos)

	columns   = {'Dataset'    :  'String'            ,
				 'Field_Str'  :  'Array_Batch'       ,
				 'Echo_Times' :  'Array_Batch'       ,
				 'sw'         :  'Array_Batch'       ,
				 'subsample'  :  'Array_Batch'       ,
				 'nPoints'    :  'Array_Batch'       ,
				 'Metab'      :  'Array_Batch_4096'  , 				
				 'MM'         :  'Array_Batch_4096'  , 	
				 'water'      :  'Array_Batch_4096'  , 
				 'noise'      :  'Array_Batch_4096'  , 
				 'time'       :  'Array_Batch_4096'  , 
				 'ppm'        :  'Array_Batch_4096'  , 	
				 'Amplitude'  :  'Array_Batch_Spins' , 
				 'water_pos'  :  'Array_Batch'       ,
				 'water_comp' :  'Array_Batch_5'     ,
				 'waterNcomp' :  'Array_Batch_5'     , 
				 'water_amp'  :  'Array_Batch_5'     ,
				 'noise_amp'  :  'Array_Batch'       ,
				 'freq_shift' :  'Array_Batch'       ,
				 'phase0'     :  'Array_Batch'       ,
				 'phase1'     :  'Array_Batch'       ,
				 'phase1_piv' :  'Array_Batch'       ,
				 'SNR'        :  'Array_Batch'       ,
				 'LBL'        :  'Array_Batch_Spins' , 
				 'LBG'        :  'Array_Batch_Spins' , 
				 'm_mult'     :  'Array_Batch_2' 	 , 	
				 'w_mult'     :  'Array_Batch_2' 	 , 
				 'LBL_Water'  :  'Array_Batch_5'     , 
				 'LBG_water'  :  'Array_Batch_5'     , 
				 'FWHM_MM'    :  'Array_Batch_14'    , 
				 'FWHM_Metab' :  'Array_Batch'       ,
				 'Healthy'    :  'Array_Batch'       ,
				 'Clinical'   :  'Array_Batch'       ,
				 'Clin_Names' :  'List'              ,
				 'Drop_Sig'   :  'Array_Batch_4096'  ,				
				 'Batch_Drop' :  'Array_Batch'       ,
				 'dIdx_Drop'  :  'Array_Batch'       ,
				}
	clist     = list(columns.keys())

	fidx      = []
	nidx      = []
	cidx      = []

	filenumber= 7
	nfids     = 120
	N         = 4096

	for ii in range(len(fields)):
		for jj in range(len(nechos)):
			fidx.extend( [fields[ii]] * nfids )
			nidx.extend( [nechos[jj]] * nfids )
			cidx.extend( list(np.arange(nfids)) )

	print('N Conditions: {}'.format(ncond))
	print('Field Index : {}'.format(len(fidx)))
	print('Echo  Index : {}'.format(len(nidx)))
	print('FileNumber  : {}'.format(filenumber))

	cnt   = 0
	ridx  = np.arange(0, ncond * nfids)
	sidx  = np.zeros([ncond * nfids])
	np.random.shuffle(ridx)
	
	for ii in range(ncond):
	# for ii in range(1):

		data    = {}
		for jj in range(len(clist)):
			
			if columns[clist[jj]] == 'String':
				data[clist[jj]] = []

			elif columns[clist[jj]] == 'List':
				data[clist[jj]] = []
			
			elif columns[clist[jj]] == 'Array_Batch_4096':
				data[clist[jj]] = np.zeros([nfids, N], dtype=np.complex_)
			
			elif columns[clist[jj]] == 'Array_Batch_5':
				data[clist[jj]] = np.zeros([nfids, 5])
			
			elif columns[clist[jj]] == 'Array_Batch_14':
				data[clist[jj]] = np.zeros([nfids, 14])

			elif columns[clist[jj]] == 'Array_Batch_2':
				data[clist[jj]] = np.zeros([nfids, 2])

			elif columns[clist[jj]] == 'Array_Batch_Spins':
				data[clist[jj]] = np.zeros([nfids, 182])

			elif columns[clist[jj]] == 'Array_Batch':
				data[clist[jj]] = np.zeros([nfids])

			else:
				print('\nERROR **** \n')

		tjstart = t0.time()
		fname = 'data_AG_{:03d}.npz'.format(ii + (filenumber * ncond) )
		for jj in range(nfids):
			tstart  = t0.time()
			
			field_  = float(fidx[ridx[cnt]])/100
			necho_  = float(nidx[ridx[cnt]])

			larmor    = field_ * 42.5760 															# Larmor Frequency
			if field_ == 3.00:
				sw    = 8000 																		# 4000 Hz at 3.00T
			else:
				sw    = larmor * 62.633095 															# Match Frequency Range (Equivalent ~62ppm for all)
			dt        = 1/sw
			sw        = int(np.round( sw,0)) 														# Round to Whole Number for Simplicity

			te2       = necho_/2000 																# TE = (TE1 + TE2 + TE3; Where TE2 = 1/2 of TE)
			te3       = te2/2 																		# TE = (TE1 + TE2 + TE3; Where TE1 and TE3 = 1/4 of TE)
			pb        = int(te3/dt)																	# Number of Points Possible Before Echo
			pb_strt   = 300 - pb 																	# Determined Max 300 points (Hard-Coded Array Length) given TE=150ms

			delay    = np.array([.00003125]) * 0													# Length of Delay Between Last Refocusing Pulse and Acquisition
			delay_   = '{:.8f}'.format(delay[0]) 													# For Ease in Filenames: String of Delay Time
			delay_   = 'delay-{}sec'.format(delay_[2:]) 											# For Ease in Filenames: String of Delay Time

			necho_    = 'TE-{}'.format(nidx[ridx[cnt]]) 											# Define for Ease in Filenames: String of Echo Time
			field_    = 'B0-{:<03}'.format(fidx[ridx[cnt]]) 										# Define for Ease in Filenames: String of Field Strength
			sw_       = 'sw-{:4d}'.format(int(np.round(sw,0)))       								# Define for Ease in Filenames: String of Bandwidth
			pb_       = 'pb-{:03d}'.format(pb) 				      									# Define for Ease in Filenames: String of N Points Before Echo

			data_     = np.load('{}/AG_{}_{}_{}_{}_{}_{:03d}.npz'.format(gendir, field_, necho_, sw_, pb_, delay_, filenumber))

			for kk in range(len(clist)):
				try:
					if columns[clist[kk]] == 'List':
						data[clist[kk]].append( data_[clist[kk]][data_['Clinical'][cidx[ridx[cnt]]]] )
					
					elif columns[clist[kk]] == 'String':
						# print('{:3d} {:6d}  {:2d} {:<20} {:<10} {}'.format(jj, ridx[jj], kk, clist[kk], type(data_[clist[kk]]).__name__,  data_[clist[kk]], )) 				# Trouble Shooting
						data[clist[kk]].append( data_[clist[kk]])				
					
					elif columns[clist[kk]] == 'Array_Batch':
						# print('{:3d} {:6d}  {:2d} {:<20} {:<10}   '.format(jj, ridx[jj], kk, clist[kk], type(data_[clist[kk]][jj]).__name__),  data_[clist[kk]][jj]) 			# Trouble Shooting
						data[clist[kk]][jj] = data_[clist[kk]][cidx[ridx[cnt]]]

					elif columns[clist[kk]] == 'Array_Batch_2':
						# print('{:3d} {:6d}  {:2d} {:<20} {:<10}   '.format(jj, ridx[jj], kk, clist[kk], type(data_[clist[kk]]).__name__),  data_[clist[kk]].shape ) 			# Trouble Shooting
						data[clist[kk]][jj,:] = data_[clist[kk]][cidx[ridx[cnt]], :]

					elif columns[clist[kk]] == 'Array_Batch_5':
						# print('{:3d} {:6d}  {:2d} {:<20} {:<10}   '.format(jj, ridx[jj], kk, clist[kk], type(data_[clist[kk]]).__name__),  data_[clist[kk]].shape ) 			# Trouble Shooting
						data[clist[kk]][jj,:] = data_[clist[kk]][cidx[ridx[cnt]], :]

					elif columns[clist[kk]] == 'Array_Batch_14':
						# print('{:3d} {:6d}  {:2d} {:<20} {:<10}   '.format(jj, ridx[jj], kk, clist[kk], type(data_[clist[kk]][jj]).__name__),  data_[clist[kk]].shape) 		# Trouble Shooting
						data[clist[kk]][jj,:] = data_[clist[kk]][cidx[ridx[cnt]], :]

					elif columns[clist[kk]] == 'Array_Batch_Spins':
						# print('{:3d} {:6d}  {:2d} {:<20} {:<10}   '.format(jj, ridx[jj], kk, clist[kk], type(data_[clist[kk]][jj]).__name__),  data_[clist[kk]].shape) 		# Trouble Shooting
						data[clist[kk]][jj,:] = data_[clist[kk]][cidx[ridx[cnt]], :]

					else:
						# print('{:3d} {:6d}  {:2d} {:<20} {:<10}   '.format(jj, ridx[jj], kk, clist[kk], type(data_[clist[kk]]).__name__),  data_[clist[kk]].shape )   		# Trouble Shooting
						data[clist[kk]][jj,:] = data_[clist[kk]][cidx[ridx[cnt]], :]

				
				except Exception as e:
					print('{:3d} {:3d} {} {} '.format(jj, kk, clist[kk], type(data_[clist[kk]])), data_[clist[kk]].shape )

			sidx[ridx[cnt]] +=1
			print('{:2d}: Current: {:3d}  |  Global: {:6d}  |  Local: {:6d}  |  {:7.2f} s  |  Cnt {:6d}'.format(ii, jj+1, ridx[cnt], cidx[ridx[cnt]], t0.time() - tstart, cnt), end='\r')
			cnt+=1

		zcnt = np.where(sidx == 0)[0]
		ocnt = np.where(sidx >  1)[0]
		print('{:2d} {:<20}:  Current: {:6d}  |  Total: {:7.2f}  |  Avg: {:7.2f}  |  Chk: {:6d} {}'.format(ii, fname, (jj + 1 + (ii*nfids)),  t0.time() - tjstart, (t0.time() - tjstart)/nfids, zcnt.shape[0], ocnt.shape[0]))
		np.savez('{}/{}'.format(trndir, fname), **data)

	print(np.sum(sidx))
	print(np.where(sidx == 0)[0])
	print(np.where(sidx >  1)[0])
	print(np.sum(sidx))
	print(' ')

if __name__ == '__main__':
	
	fpath     = os.path.realpath(__file__) 												    # Base Directory
	basedir   = '/'.join(fpath.split('/')[:-1]) 											# Base Directory
	# basedir = 'C:/Users/agudm/Desktop'

	randomize(basedir)