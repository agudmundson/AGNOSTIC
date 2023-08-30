__author__ = 'Aaron Gudmundson'
__email__  = 'agudmund@uci.edu'

from scipy.sparse import csr_matrix, identity, kron
import matplotlib.pyplot as plt																# Create Figures
import scipy.ndimage as ndimage																# Image Manipulation
import scipy.linalg as linalg
import scipy.sparse as sparse 																# Sparse Matrices
import scipy.stats as stats																	# Statistics
import pandas as pd																			# Data Frames
import numpy as np																			# Arrays/Linear Algebra
import subprocess																			# Terminal Control
import copy 																				# DeepCopy Class Arrays
import glob																					# Bash Commands
import os																					# Interact w/Operating System

np.set_printoptions(threshold=np.inf, precision=2, linewidth=300, suppress=True)			# Set Numpy Display Params
np.seterr(divide='ignore', invalid='ignore')												# Set Numpy Operation Params
pd.set_option('display.width', 1000)														# Set Pandas Display Params

class Simulation:

	def __init__(self, basedir, omega, jcoup, B0, N=8192, sw=4000, lbl=1, lbg=1, cen_frq=4.7, nucleus='1H'):

		gyro         = {				 													# Gyromagnetic Ratio (MHz/Tesla) or (γ/2π)
						'1H'  : 42.5760, 													#  1H  Proton
						'2H'  :  6.5360, 													#  2H  Deuterium
						'13C' : 10.7084, 													# 13C  Carbon
						'15N' : -4.3160,													# 15N  Nitrogen
						'17O' : -5.7720, 													# 17O  Oxygen
						'23Na': 11.2620, 													# 23Na Sodium
						'31P' : 17.2350,													# 31P  Phosphorous
						}
		self.gyro    = gyro[nucleus] 														# Nucleus for Simulation 

		self.B0      = B0                                                                   # Field Strength (Tesla)
		self.N       = N 																	# Number of Datapoints
		self.sw      = sw 																	# Spectral Width
		self.cen_frq = cen_frq 																# Center Frequency (Typically H2O at 4.7)
		self.lblornz = lbl 																	# Lorentzian Line Broadening
		self.lbgauss = lbg  																# Gaussian Line Broadening

		self.W0      = -1 * 2 * np.pi * B0 * self.gyro * 1e6 								# Omega0 ( Radians/(seconds Tesla) )

		self.dwell   = 1/self.sw 															# Dwell Time (ms) 
		self.time    = np.linspace(0, self.dwell * N, N) 									# Time Domain Axis (ms)

		self.hz_ppm  = self.B0 * self.gyro   												# Hertz per ppm
		
		self.hz      = np.linspace(0, self.sw/self.hz_ppm, N)								# Freq Domain Axis in Hertz
		self.hz     -= self.hz[N//2] 														# Center to 0

		self.ppm     = np.linspace(0, sw/(self.gyro*self.B0), N) 							# Freq Domain Axis in ppm
		self.ppm    -= self.ppm[self.ppm.shape[0]//2] 										# Center to 0 
		self.ppm    += self.cen_frq															# Set Center Frequency

		self.basedir = basedir 	  		 													# Base Directory
		self.nspins  = len(omega)															# Number of Spins in Simulation
		self.fid	 = np.zeros([N], dtype=np.complex_) 									# FID Array (All Spins)
		self.fid_	 = np.zeros([N, self.nspins], dtype=np.complex_) 						# FID Array (Individual Spins)
		
		self.sx      = np.array([[0.0 ,  0.5 ], 											# Pauli Spin Matrices Sx
							     [0.5 ,  0.0 ]], dtype=np.complex_)
		self.sy      = np.array([[0.0 , -0.5j], 											# Pauli Spin Matrices Sy
							     [0.5j,  0.0 ]], dtype=np.complex_)
		self.sz      = np.array([[0.5 ,  0.0 ], 											# Pauli Spin Matrices Sz
							     [0.0 , -0.5 ]], dtype=np.complex_)

		msize        = np.power(2, self.nspins) 											# Determine Matrix Size
		self.unit    = np.identity(2)
		self.E       = np.identity(msize) 													# Identity of Matrix Size
		self.Ix      = np.zeros([msize, msize, self.nspins], dtype=np.complex_) 			# X Component
		self.Iy      = np.zeros([msize, msize, self.nspins], dtype=np.complex_) 			# Y Component
		self.Iz      = np.zeros([msize, msize, self.nspins], dtype=np.complex_) 			# Z Component
																							
																							# Create Initial State
		for ii in range(self.nspins): 														# Loop over Spins
			Ix  = 1 																		# Initialize X
			Iy  = 1																			# Initialize Y
			Iz  = 1																			# Initialize Z

			for jj in range(self.nspins):
				if ii == jj: 																# On Diagonol 
					Ix = np.kron(Ix, self.sx)												# X w/Unit 
					Iy = np.kron(Iy, self.sy)												# Y w/Unit
					Iz = np.kron(Iz, self.sz)												# Z w/Unit 
				else:							 											# Off Diagonal
					Ix = np.kron(Ix, self.unit) 											# Current X w/Unit
					Iy = np.kron(Iy, self.unit) 										    # Current Y w/Unit
					Iz = np.kron(Iz, self.unit) 											# Current Z w/Unit

			self.Ix[:, :, ii] = Ix															# Set X Magnetization
			self.Iy[:, :, ii] = Iy															# Set Y Magnetization
			self.Iz[:, :, ii] = Iz															# Set Z Magnetization

		self.H         = np.zeros([msize, msize             ], dtype=np.complex_)			# Hamiltonian (All Spins)
		self.h         = np.zeros([msize, msize, self.nspins], dtype=np.complex_)  		    # Hamiltonian (Individual Spins)

		self.omega_ppm = omega 																# Offset in ppm
		self.jcoup     = jcoup * np.pi														# J Coupling in Hz
		self.omega     = np.zeros([self.nspins]) 											# Offset in Hz

		for ii in range(self.nspins): 														# Zeeman Hamiltonian
			self.omega[ii] = (self.W0 / 1e6) * (omega[ii] - cen_frq)						# Offset
			self.H        += (self.omega[ii] * self.Iz[:,:,ii]) 							# Resonant
			self.h[:,:,ii] = (self.omega[ii] * self.Iz[:,:,ii]) 							# Resonant

		for ii in range(self.nspins):
			for jj in range(self.nspins):
				self.h[:,:,ii] += self.jcoup[ii, jj] * (  self.Ix[:,:,ii] * self.Ix[:,:,jj] # Coupling Constant; Ix
					                                    + self.Iy[:,:,ii] * self.Iy[:,:,jj] # Iy
					                                    + self.Iz[:,:,ii] * self.Iz[:,:,jj])# Iz

				self.H         += self.jcoup[ii, jj] * (  self.Ix[:,:,ii] * self.Ix[:,:,jj] # Coupling Constant; Ix
					                                    + self.Iy[:,:,ii] * self.Iy[:,:,jj] # Iy
					                                    + self.Iz[:,:,ii] * self.Iz[:,:,jj])# Iz

		self.rho_ = copy.deepcopy(self.Iz)   												# p0 (Individual Spins)														
		self.rho  = np.sum(self.Iz, axis=2)  												# p0 (All Spins)													

	def evolve(self, tstep=None): 															# Evolution Hamiltonian
		if tstep == None:																	# No TimeStep Provided
			tstep = .00025 																	# Default to 250 microseconds
		p         = linalg.expm(-1j * tstep * self.H) 										# Exponentiate
		p_        = linalg.expm( 1j * tstep * self.H) 										# Exponentiate
		self.rho  = (p @ self.rho) @ p_ 													# Computes exp() rho exp()*
		
		for ii in range(self.nspins):
			self.rho_[:,:,ii]  = (p @ self.rho_[:,:,ii]) @ p_ 								# Computes exp() rho exp()*

		return self

	def pulse(self, phi, ax='x'): 															# Pulse Hamiltonian
		I_       = np.sum(self.__dict__['I{}'.format(ax.lower())], axis=2) 					# Pulse on Ix, Iy, or Iz
		p        = linalg.expm(-1j * I_ * phi)									    		# Exponentiate
		p_       = linalg.expm( 1j * I_ * phi)									    		# Exponentiate

		self.rho = (p @ self.rho) @ p_ 														# Computes exp() rho exp()*

		for ii in range(self.nspins):
			self.rho_[:,:,ii] = (p @ self.rho_[:,:,ii]) @ p_ 								# Computes exp() rho exp()*
		
		return self
	
	def pulse_ph(self, phi, phase): 														# Pulse Hamiltonian
		Ix       = np.sum(self.Ix, axis=2) 													# Iz
		Iz       = np.sum(self.Iz, axis=2) 													# Iz
		
		p        = linalg.expm(-1j *  Iz * phase)											# Exponentiate
		px       = linalg.expm(-1j *  Ix * phi)
		p_       = linalg.expm(-1j * -Iz * phase)											# Exponentiate
		rho_lft  = (p @ px) @ p_ 
 
		p        = linalg.expm( 1j * -Iz * phase)											# Exponentiate
		px       = linalg.expm( 1j *  Ix * phi)
		p_       = linalg.expm( 1j *  Iz * phase)											# Exponentiate
		rho_rgt  = (p @ px) @ p_ 

		self.rho = (rho_lft @ self.rho) @ rho_rgt 											# Computes exp() rho exp()*

		for ii in range(self.nspins):
			self.rho_[:,:,ii] = (rho_lft @ self.rho_[:,:,ii]) @ rho_rgt  					# Computes exp() rho exp()*
		
		return self

	def pulse_edit(self, phi, phase, ppm, hwidth=.25): 										# Pulse Hamiltonian
		Ix       = np.sum(self.Ix, axis=2) 													# Ix
		Iz       = np.sum(self.Iz, axis=2) 													# Iz
		
		p        = linalg.expm(-1j *  Iz * phase)											# Exponentiate
		px       = linalg.expm(-1j *  Ix * phi)
		p_       = linalg.expm(-1j * -Iz * phase)											# Exponentiate
		rho_lft  = (p @ px) @ p_ 
 
		p        = linalg.expm( 1j * -Iz * phase)											# Exponentiate
		px       = linalg.expm( 1j *  Ix * phi)
		p_       = linalg.expm( 1j *  Iz * phase)											# Exponentiate
		rho_rgt  = (p @ px) @ p_ 

		for ii in range(self.nspins): 														# Iterate over Individual Spins
			if self.omega_ppm[ii] > ppm-hwidth and self.omega_ppm[ii] < ppm+hwidth: 		# Determine if Spin is within Edit Pulse Range
				print('{} Edit ***'.format(self.omega[ii]))
				self.rho_[:,:,ii] = (rho_lft @ self.rho_[:,:,ii]) @ rho_rgt  				# Computes exp() rho exp()*

		self.rho = np.sum(self.rho_[:,:,:], axis=-1) 										# Computes exp() rho exp()*
		
		return self


	def detect(self, phase, tstep=.00025): 													# Detection
		coil  = (np.sum(self.Ix, axis=2) - (1j * np.sum(self.Iy, axis=2))) 					# Receiver
		# print('Phase: {}'.format(np.exp(1j * phase)))
		coil *= np.exp(1j * phase)

		for ii in range(self.fid.shape[0]): 												# Compute Free Induction Decay
			self.fid[ii] = np.matrix.trace(coil @ self.rho) 								# Trace of detection w/rho
			
			for jj in range(self.nspins):
				self.fid_[ii,jj] = np.matrix.trace(coil @ self.rho_[:,:,jj]) 				# Trace of detection w/rho

			self.evolve(tstep=self.dwell)													# Evolve for time 'tstep' 

		return self		

if __name__ == '__main__':
	
	basedir = 'C:/Users/Aaron Gudmundson/Desktop/Deep_Sep'

	ph1  = np.array([0 ,  0, 90, 90]) * np.pi/180 											# Phase Cycling Ref Pulse 1
	ph2  = np.array([0 , 90,  0, 90]) * np.pi/180 											# Phase Cycling Ref Pulse 2
	fids = np.zeros([8192, 4], dtype=np.complex_)

	for ii in range(1):
		s   = Simulation(basedir = 'C:/Users/Aaron Gudmundson/Documents/GitHub/Deep',
			  			 omega   = np.array([ 2.2840, 2.2840, 1.8880, 1.8880, 3.0130, 3.0130]), 
				     	 jcoup   =  np.array([[   0.0,    0.0,  7.678,  6.980,    0.0,    0.0],
									          [   0.0,    0.0,  6.980,  7.678,    0.0,    0.0],
									          [   0.0,    0.0,    0.0,    0.0,  8.510,  6.503],
									          [   0.0,    0.0,    0.0,    0.0,  6.503,  8.510],
									          [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
									          [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],]),

				      #    omega   = (4.0974, 1.3142, 1.3142, 1.3142), 
					     # jcoup   = np.array([[ 0.0, 6.9330, 6.9330, 6.9330],
										#    [   0.0,   0.0 ,  0.0  ,   0.0 ],
										#    [   0.0,   0.0 ,  0.0  ,   0.0 ],
										#    [   0.0,   0.0 ,  0.0  ,   0.0 ]]),
					     cen_frq = 4.7 ,
					     B0      = 3.0 ,
					     N       = 8192,
					   )

		s   = s.pulse(phi=np.pi/2, ax='x')
		s   = s.evolve(tstep=.0005)
		s   = s.pulse_ph(phi=np.pi, phase=ph1[ii])
		s   = s.evolve(tstep=.0010)
		s   = s.pulse_ph(phi=np.pi, phase=ph2[ii])
		s   = s.evolve(tstep=.0005)
		fid = s.detect(phase=np.pi/2)

		print(fid.fid.shape)

		fids[:,ii]  = fid.fid
		# fids[:,ii] *= np.exp(-1 * np.linspace(0, .00025 * 8192 ,8192) * (np.pi * 1))

	fid        = (fids[:,1] - fids[:,0]) + (fids[:,2] - fids[:,3])
	spectrum   = np.real(np.fft.fftshift(np.fft.fft(fid, n=8192)))
	spectrum  /= spectrum.shape[0]

	ppm        = np.linspace(0, 4000/(42.5760*s.B0), 8192) 
	ppm       -= ppm[ppm.shape[0]//2]
	ppm       += 4.7

	fig, ax    = plt.subplots()
	ax.plot(ppm, spectrum, color='royalblue')
	ax.axvline(4.7, color='grey', linestyle='dotted')
	ax.set_xlim([0.0, 6.0])
	ax.invert_xaxis()
	plt.show()
