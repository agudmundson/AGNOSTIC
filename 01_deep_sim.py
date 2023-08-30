__author__ = 'Aaron Gudmundson'
__email__  = 'agudmund@uci.edu'

# from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt																# Create Figures
# import scipy.ndimage as ndimage																# Image Manipulation
# import scipy.linalg as linalg
# import scipy.sparse as sparse 																# Sparse Matrices
# import scipy.stats as stats																	# Statistics
# import scipy.signal as sig 																	# Signal Processing
# import pandas as pd																			# Data Frames
import numpy as np																			# Arrays/Linear Algebra
import time as t0                                                                           # Determine Run Time
import subprocess																			# Terminal Control
import datetime
import random
import copy 																				# DeepCopy Class Arrays
import glob																					# Bash Commands
import os																					# Interact w/Operating System

import simulation                                                                           # Density Matrix
import metab_dict                                                                           # Metabolites

np.set_printoptions(threshold=np.inf, precision=2, linewidth=300, suppress=True)			# Set Numpy Display Params
np.seterr(divide='ignore', invalid='ignore')												# Set Numpy Operation Params
# pd.set_option('display.width', 1000)														# Set Pandas Display Params

def indices(w, w_, ppm, verbose=False):
	idx    = []
	for ii in range(w.shape[0]):
		idx1 = np.where(np.abs(ppm - w[ii] + w_[ii]) == np.min(np.abs(ppm - w[ii] + w_[ii])))[0] # ppm range Low
		idx2 = np.where(np.abs(ppm - w[ii] - w_[ii]) == np.min(np.abs(ppm - w[ii] - w_[ii])))[0] # ppm range High
		idx.append([int(idx1), int(idx2)]) 														 # Indices

	nidx1   = int(np.where(np.abs(ppm + 2.00      ) ==np.min(np.abs(ppm + 2.00      )))[0]) 	# Noise Est ppm range
	nidx2   = int(np.where(np.abs(ppm + 0.00      ) ==np.min(np.abs(ppm + 0.00      )))[0]) 	# Noise Est ppm range
	n_idx   = [[nidx1 , nidx2]] 																# Noise Indices

	for ii in range(1, len(idx)): 																# Ensure no Overlap
		ii_ = (ii - 1) 																			# Previous Index Set
		if ppm[idx[ii][0]] >= ppm[idx[ii_][0]] and ppm[idx[ii][0]] <= ppm[idx[ii_][1]]: 		# If 1st idx is between the last set.
			
			dif = ppm[idx[ii_][1]] - ppm[idx[ii][0]] 											# Difference Between Start and End
			mid = dif/2 																		# Find MidPoint

			idx[ii ][0] = int(np.where(np.abs(ppm - w[ii ] + mid) == np.min(np.abs(ppm - w[ii ] + mid)))[0]) # Replace w/ Non-Overlapping
			idx[ii_][1] = int(np.where(np.abs(ppm - w[ii_] - mid) == np.min(np.abs(ppm - w[ii_] - mid)))[0]) # Replace w/ Non-Overlapping

	idx     = np.array(idx)
	n_idx   = np.array(n_idx)

	return idx, n_idx

def _fwhm(w, idx0, idx1, spec, axis=None, ppm=None, plot=False):
	
	point = int(np.where(np.abs(ppm - w) == np.min(np.abs(ppm - w)))[0][0])

	famp  = np.max(spec[idx0:idx1])
	hamp  = famp/2
		
	left  = np.where(np.abs(spec[idx0:point]-hamp) == np.min(np.abs(spec[idx0:point]-hamp)))[0][0]
	rght  = np.where(np.abs(spec[point:idx1]-hamp) == np.min(np.abs(spec[point:idx1]-hamp)))[0][0] 
	
	left += idx0
	rght += point

	if plot == True:
		fig,ax = plt.subplots()
		ax.plot(ppm, spec, color='royalblue')
		ax.axhline(famp     , color='black'    , linestyle='dotted')
		ax.axhline(hamp     , color='black'    , linestyle='dotted')
		ax.axhline(0.0      , color='black'    , linestyle='dotted')
		ax.axvline(ppm[left], color='firebrick', linestyle='dashed')
		ax.axvline(ppm[rght], color='firebrick', linestyle='dashed')
		ax.axvline(ppm[idx0], color='black'    , linestyle='dotted')
		ax.axvline(ppm[idx1], color='black'    , linestyle='dotted')
		ax.set_xlim([0.0,2.0])
		ax.invert_xaxis()
		plt.show()

	if axis is not None:
		return (axis[int(rght)] - axis[int(left)], left, rght)
	else:
		return (rght-left, left, rght)

def create_basis(B0=3.0, N=16684):
 
  now        = datetime.datetime.now()
  print('Starting Simulations.. ', now.strftime('%m/%d/%Y  %H:%M:%S')) 
  print('\n')
  
  fpath  = os.path.realpath(__file__)
  basedir= '/'.join(fpath.split('/')[:-1])
  
  m_dict = metab_dict.metab_dict
  m_list = sorted(list(m_dict.keys()))
  random.shuffle(m_list)
  m_list.remove('Doub')
  m_list.remove('Trip')
  m_list.remove('Quar')
  m_list.remove('Cho' )
  m_list.remove('ATP1') 
  m_list.remove('ATP2') 
  m_list.remove('Hist')
  
  delay  = .00003125 * 0
  delay_ = '{:.8f}'.format(delay)
  delay_ = delay_[2:]
  
  # necho = np.array([29,30,31,32,33,34,35])
  necho  = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
  # necho  = np.array([50, 55, 60, 70, 75, 80])
  
  for ii in range(necho.shape[0]):
    start_time = t0.time()
    for jj in range(len(m_list)):

      metab  = m_list[jj] 																        # Current Metabolite
      
      te     = necho[ii]/1000 															      # Echo Time 
      te2    = te /2 																		          # Time Between Pulses
      te1    = te2/2 																		          # Time to First Pulse
      te3    = te2/2 																		          # Time to Acquisition
      
      Larmor = B0 * 42.576 																        # Larmor Frequency /1e6
      
      if B0  == 3.0: 																		          # Everything Reflects 3.0T w/8000Hz
      	sw = 8000 																		            # Spectral Width
      else: 
      	sw = Larmor * 62.633095 														      # Spectral Width
      
      sw     = int(np.round(sw, 0)) 														  # Round and Convert to Integer
      dwell  = 1/sw 																		          # Dwell Time
      
      pb     = int(te3/dwell)
      pb_strt= 300 - pb 																	        # Determined Max 300 points Before Echo given Max TE=150ms
      
      ph1    = np.array([0 ]) * np.pi/180 												# Phase Cycling Ref Pulse 1
      ph2    = np.array([90]) * np.pi/180 												# Phase Cycling Ref Pulse 2
      
      r      = m_dict[metab]
      same   = np.unique(r['omega'], return_index=True) 					# Identify Unique Spins
      
      hnuclei= np.where(r['omega'] != 0.0)												# Protons (Exclude X Nuclei)
      
      nspins = r['omega'].shape[0] 														    # Number of Spins Total
      fids   = np.zeros([N, nspins], dtype=np.complex_) 					# Inidividual Spins
      
      sdir  = '{}/simulated/B0-{:<03}_TE-{:03d}'.format(basedir, str(B0).replace('.',''), int(te*1000))
      if os.path.exists(sdir) == False:
      	print('\nMaking Directory: {}'.format(sdir))
      	os.mkdir(sdir)
              
      sname = '{}_B0-{:<03}_TE-{:03d}_sw-{:4d}_pb-{:03d}_delay-{}sec.npy'.format(metab, str(B0).replace('.',''), int(te*1000), int(np.round(sw,0)), pb, delay_)			
      sname = '{}/{}'.format(sdir, sname)
      
      if os.path.exists(sname) == True:
      	print('{:3d} {:3d} {:<30}  |  Completed..'.format(ii, jj, sname.split('/')[-1],))
      	continue      
      
      print('-- '*24)
      
      now_   = datetime.datetime.now()
      print(now_.strftime('%m/%d/%Y  %H:%M:%S'), '  |  Current Time: {:.2f} seconds  |   ~{:6.2f}% Complete'.format(t0.time() - start_time, 100*((t0.time() - start_time)/10700) ))	
      
      print('Necho: ', necho)
      print('{:2d}: {}'.format(jj, sname.split('/')[-1]))
      print('B0: {:.2f}  |  SW: {:.2f}  |  Dt: {:8.6f}   |   Points Before: {:3d}'.format(B0, sw, dwell, pb))
      print('TE: {}ms ({} | {} | {})'.format(int(te*1000), te1*1000,te2*1000,te3*1000))
      print('{:<14}  |  NSpins: {:2d}  |  UniqueSpins: {:2d}'.format(r['name'], nspins, same[0].shape[0]))
      print(r['omega'])
      print(r['jcoup'])
      
      tstart = t0.time()
      # for kk in range(ph1.shape[0]):
      t1 = t0.time()
      s  = simulation.Simulation(basedir = basedir,
      						   omega   = r['omega'], 
      						   jcoup   = r['jcoup'],
      						   cen_frq = 4.7,
      						   B0      = B0 ,
      						   sw      = sw ,
      						   N       = N  ,)
      
      s   = s.pulse(phi=np.pi/2, ax='x')
      s   = s.evolve(tstep=te1)
      s   = s.pulse_ph(phi=np.pi, phase=ph1[0])
      s   = s.evolve(tstep=te2)
      s   = s.pulse_ph(phi=np.pi, phase=ph2[0])
      # s   = s.evolve(tstep=te3)
      s   = s.evolve(tstep=delay)
      f   = s.detect(phase=np.pi/2)
      
      # print('Phase Cycle {}/{} {:.3f} Seconds'.format(kk+1, ph1.shape[0], t0.time()-t1), end='\r')
      # fids[:,kk,:] = f.fid_														    # FID w/  Phase Cycling
      
      fids[pb_strt:,:] = f.fid_[:-int(300-pb),:]   									# FID w/o Phase Cycling
      
      print('Total Time  {:.3f} Seconds            '.format(t0.time()-tstart))
      print('-- '*24)
      
      # fids[:,0,:] *= -1
      # fids[:,3,:] *= -1
      # fids         = np.sum(fids[:,:,hnuclei], axis=1)
      # fids         = np.squeeze(fids)
      
      np.save(sname, fids)

if __name__ == '__main__':

  # field      = np.array([1.4, 1.8, 2.2, 2.6, 3.0])
  # field      = np.array([1.5, 1.9, 2.3, 2.7, 3.1])
  # field      = np.array([1.6, 2.0, 2.4, 2.8])
  # field      = np.array([1.7, 2.1, 2.5, 2.9,    ])

  # field      = np.array([2.0, 2.8])
  # field      = np.array([1.6, 2.4])  
  field      = np.array([1.4, 1.8, 2.2, 2.6, 3.0, 1.5, 1.9, 2.3, 2.7, 3.1, 1.6, 2.0, 2.4, 2.8, 1.7, 2.1, 2.5, 2.9,])
  
  for ii in range(field.shape[0]):
    create_basis(B0=field[ii])

