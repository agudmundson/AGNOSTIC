
__author__  = 'Aaron Gudmundson'
__email__   = 'agudmun2@jhmi.edu'
__date__    = '2023/30/08'


import numpy as np 																		# Numerical Operations and Arrays
import argparse 																		# Input Argument Parser
import sys 																				# System Interaction
import os 																				# Operating System Interaction


class AGNOSTIC_CONVERSION():

	def __init__(self, inputs):

		self.conv_type = inputs.conversion  											# jMRUI, NIfTI, or MATLAB
		self.filename  = inputs.filename 												# Input Filename
		self.start_idx = inputs.start 													# Index to Start
		self.end_idx   = inputs.end 													# Index to End
		self.label     = inputs.label													# Component Label

		self.filename  = self.basedir.replace('\\', '/') 								# Replace forward slashes for Windows
		self.outdir    = os.path.dirname(self.filename) 								# Set the Output Directory 
		self.outname   = 'fname-{}_label-{}_start-{}_end-{}.nii.gz'.format(
												self.filename.split('/')[-1][:-3],
												self.label, 
												self.start_idx,
												self.end_idx)

		if inputs.quiet == 0:
			print(
				'''
				Thank you for your interest in AGNOSTIC
				If you find this dataset useful, please cite:
				  https://www.biorxiv.org/content/10.1101/2023.05.08.539813v1

				We've created this stand-alone tool to easily convert AGNOSTIC components to either:
				  - NIfTI, as described in https://doi.org/10.1002/mrm.29418
				  - jMRUI text files
				
				OR to Convert the whole npz file to: 
				  - MATLAB structure.

				Note*
				  If converting to NIfTI, please ensure that nifti-mrs (https://pypi.org/project/nifti-mrs/) is installed.
				    - nifti-mrs can be installed with anaconda using: conda install -c conda-forge nifti-mrs
				
				  If converting to matlab, please ensure that scipy (https://scipy.org/) is installed.
				    - This should come standard with anaconda and mamba installations.

				To use this tool provide the following arguments:
				    - Conversion Type: "nifti" or "jmrui"
				    - Data File      : NumPy zipped archive file
				    - Starding Index : Integer (with 0 being the 1st index)
				    - Ending Index   : Integer (with 0 being the 1st index; exclusive)
				    - Label          : "Metabolite", "MM", "Noise", "Water" or User-defined (i.e., "OOV_Echo")

				From the command line, a call should look like:
				  >> python AGNOSTIC_Conversion -c nifti -f myfile.npz -s 0 -e 10 -l metabolite

				  or 

				  >> python AGNOSTIC_Conversion -c jmrui -f myfile.npz -s 0 -e 10 -lOOV_Echo

				  or

				  >> python AGNOSTIC_Conversion -c matlab -f myfile.npz -l full

				''')

		## Display User Inputs
		print('Selected Type   : {}'.format(self.conv_type))
		print('Input Directory : {}'.format(self.outdir))
		print('Input Directory : {}'.format(self.filename.split('/')[-1][:-3]))
		print('Component       : {}'.format(self.label))		
		print('Start Index     : {}'.format(self.start_idx))
		print('End Index       : {}'.format(self.end_idx))
		print('Output Directory: {}'.format(self.outdir))
		print('Output Filename : {}'.format(self.outname))
		print(' ')


		## Get Data
		self.data      = np.load(self.filename)


		## Convert entire npz file to MATLAB struct
		if self.conv_type.lower() == 'matlab' and self.label == 'Full':
			print(' ')
			print('Converting Entire {} file'.format(self.filename))
			import scipy.io as sio 																		# SciPy
			sdata    = {'AGNOSTIC': sdata}
			sio.savemat(outfile, sdata)
			return True


		## Select range from provided indices
		self.data      = self.data[self.label][     self.start_idx:self.end_idx,...]
		self.larmor    = self.data[self.larmor][    self.start_idx:self.end_idx,...]
		self.dwell     = self.data[self.dwell_time][self.start_idx:self.end_idx,...]


		## Note if error occurs during conversion of multiple examples
		self.error = False
		for ii in range(self.data.shape[0]):
			if self.error == True:
				print('Error at {} of {} ** ** **'.format(ii, self.data.shape[0]))
				break

			if self.conv_type.lower() == 'nifti':
				self.error = self.nifti_conversion( self.data[  ii,...], 
												    self.dwell[ ii    ], 
												    self.larmor[ii    ])

			elif self.conv_type.lower() == 'jmrui':
				self.error = self.jmrui_conversion( self.data[  ii,...], 
												    self.dwell[ ii    ], 
												    self.larmor[ii    ])

			elif self.conv_type.lower() == 'matlab':
				self.error = self.matlab_conversion(self.data[  ii,...], 
												    self.dwell[ ii    ], 
												    self.larmor[ii    ])				

			else:
				print('Please select either "nifti" or "jmrui" ')
				break

	def nifti_conversion(self, data, dwell, larmor, label=None):

		import nifti_mrs
		from nifti_mrs.create_nmrs import *

		try:
			nii = gen_nifti_mrs(data      = data[None, None, None, : None, None], 
	                            dwelltime = dwell,
	                            spec_freq = larmor)
			nii.save('{}.nii.gz'.format(self.outdir, self.outname))
			return True

		except Exception as e:
			print('Failed with error: ')
			print('  {}'.format(e))

			return False


	def jmrui_conversion(self, data, dwell, larmor, label=None):

		outfile  = '{}/{}.txt'.format(self.outdir, self.outname)

		fid_real  = np.real(data[:])
		fid_imag  = np.imag(data[:])
		spec_real = np.real(np.fft.fftshift(np.fft.fft(data)))
		spec_imag = np.imag(np.fft.fftshift(np.fft.fft(data)))

		with open(outfile, 'w') as jmrui:
			jmrui.write('jMRUI Data Textfile')
			jmrui.write('\n\nFilename: {}'.format(outfile))
			jmrui.write('\n\nPointsInDataset: '.format(data.shape[0]))
			jmrui.write('\nDatasetsInFile: 1')
			jmrui.write('\nSamplingInterval: {:.3e}'.format(dwell))
			jmrui.write('\nMagneticField: {:.2f}'.format(larmor))
			jmrui.write('\nTypeOfNucleus: 1H')
			jmrui.write('\nsig(real)\tsig(imag)\tfft(real)\tfft(imag)\n')

			for ii in range(data.shape[0]):
				jmrui.write('\n{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\n'.format(fid_real, fid_imag, spec_real, spec_imag))
		
		return True

	def matlab_conversion(self, data, dwell, larmor, label=None):

		import scipy.io as sio 																		# SciPy

		outfile  = '{}/{}.mat'.format(self.outdir, self.outname)

		sdata    = {label   : data  ,
				    'Dwell' : dwell ,
				    'Larmor': larmor}

		sdata    = {'AGNOSTIC': sdata}
		sio.savemat(outfile, sdata)

		return True

if __name__ == '__main__':

	parser     = argparse.ArgumentParser() 															# Input Argument Parser
	parser.add_argument('-c', '--conversion', help='Conversion Type: jMRUI, NIfTI, MATLAB'     , type=str)
	parser.add_argument('-f', '--filename'  , help='Zipped Archive NumPy Filename of AGNOSTIC ', type=str)
	parser.add_argument('-s', '--start'     , help='Data Starting index (0-based)' 			   , type=int, const=0)
	parser.add_argument('-e', '--end'       , help='Data Ending index' 					       , type=int, const=1)
	parser.add_argument('-l', '--label'     , help='Component Label in .npz file'			   , type=str) 
	parser.add_argument('-q', '--quiet'     , help='Suppress Messages' 						   , type=int, nargs='?', const=0) 

	args       = parser.parse_args() 																		# 

	AGNOSTIC_CONVERSION(args)




