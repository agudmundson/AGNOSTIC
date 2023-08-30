
__author__  = 'Aaron Gudmundson'
__email__   = 'agudmun2@jhmi.edu'
__date__    = '2022/08/01'
__version__ = '01.0.0'
__status__  = 'beta'


from scipy.optimize import minimize, curve_fit
import matplotlib.patches as mpatches                                                   	# Figure Legends
import matplotlib.pyplot as plt                                                         	# Plotting Figures
import scipy.signal as sig                                                              	# Signal Processing
import pandas as pd                                                                     	# DataFrames
import numpy as np                                                                      	# Arrays/LinAlg
import time as t0                                                                       	# Determine Run Time
import resource 																			# CPU/RAM
import copy                                                                             	# Copy Arrays
import sys                                                                              	# Interact with System Files
import os
import gc 																					# Garbage Collection

# import nvidia_smi 																		# GPU
from pynvml import *																		# GPU

from tensorflow.keras import backend as K 													# Keras Backend
from tensorflow.keras import callbacks                                                  	# Keras Callbacks
from tensorflow.keras import metrics                                                    	# Keras Training Metrics
from tensorflow.keras import backend                                                    	# Keras Math Operations
from tensorflow.keras import models                                                     	# Keras Models
from tensorflow.keras import layers                                                     	# Keras Layers
from tensorflow.keras import Input                                                      	# Keras Inputs      
from tensorflow.keras import Model                                                      	# Keras Model Object

from tensorflow.config import threading, experimental 										# Tensorflow GPU and Parralel Processing
from tensorflow import config
gpus = config.list_physical_devices('GPU')
print('\nGPU Device:\n  ', gpus[0], '\n\n')
try:
	config.set_logical_device_configuration(gpus[0], [config.LogicalDeviceConfiguration(memory_limit=5000)]) # 2500
	# config.set_logical_device_configuration(gpus[0], [config.LogicalDeviceConfiguration(memory_limit=5000)])
	# experimental.set_memory_growth(gpus[0], True)
except Exception as e:
	print('\n\n\nError   ***')
	print(e)

# threading.set_inter_op_parallelism_threads(2)
# threading.set_intra_op_parallelism_threads(2)

from tensorflow.math import multiply, subtract, square, divide, sigmoid, count_nonzero
from tensorflow.math import round as tf_round
from tensorflow.data import Dataset

from tensorflow import function as tfFunction
from tensorflow import float32 as tf_float32
from tensorflow import float64 as tf_float64
from tensorflow import convert_to_tensor                                                	# Tensorflow Convert Object to Tensor
from tensorflow import print as tfprint                                                 	# Tensorflow Random Initializers
from tensorflow import clip_by_value                                                    	# Tensorflow Gradient Clipping
from tensorflow import GradientTape                                                     	# Tensorflow Gradient Mapping
from tensorflow import optimizers                                                       	# Tensorflow Gradient Algorithms
from tensorflow import reduce_mean
from tensorflow import reduce_sum
from tensorflow import losses                                                           	# Tensorflow Loss Metrics
from tensorflow import random                                                           	# Tensorflow Random Initializers
from tensorflow import zeros_like                                                       	# Tensorflow Zeros
from tensorflow import gradients                                                        	# Tensorflow Gradients
from tensorflow import ones                                                       			# Tensorflow Ones
from tensorflow import reshape
from tensorflow import constant                                                       		# Tensorflow Constant
from tensorflow import cast                                                       			# Tensorflow Ones
from tensorflow import config
from tensorflow import py_function, numpy_function
from tensorflow import executing_eagerly

class SingleCNN():

	def __init__(self, basedir, trndir, modeldir, bsize, target, domain='Time', verbose=False, loss_fx='MeanSquaredError'):

		self.dname    = 'AG' 																# Dataset 'AG' w/Dimensions 120, 4096 (Batch, Datapoints)
																							#  270 Basis Sets (1.4T-3.1T; 10ms-80ms)
																							#    3 Subsamples (2,3,4)
																							#  Clinical and Healthy Populations
		self.data_sz  = 120

		self.trn_ii   = 0 																	# Current Training Data File
		self.trn_jj   = 0 																	# Current Index within Training File
		self.trn_strt = 0 																	# Starting Point of Training Data
		self.trn_lim  = 25 																	# Limit for Training Data

		self.val_ii   = 35  																# Current Validation Data File
		self.val_jj   = 0 																	# Current Index within Validation Files
		self.val_strt = 0   																# Starting Point of Validation Data
		self.val_lim  = 35																	# Limit for Validation Data

		self.basedir  = basedir 															# Base Directory
		self.trndir   = trndir 																# Training/Validation/Testing Data 
		self.modeldir = modeldir 															# Save Out Models

		self.bsize    = bsize 																# BatchSize

		self.basedir    = basedir 															# Base Directory
		self.trndir     = trndir 															# Training/Validation/Testing Data 
		self.modeldir   = modeldir 															# Save Out Models
		self.target     = target.lower() 													# Target Metabolite, MM, or Water
		self.stime      = t0.time()
		self.ctime      = t0.time()

		self.bsize      = bsize 															# BatchSize
		self.copy       = lambda x  : copy.deepcopy(x) 										# Deep Copy
		self.verbose    = verbose 															# Output Information

		# Model Layer/Functions
		self.dense      = lambda units, x: layers.Dense(units)(x) 							# Dense Connections
		self.norm       = lambda x: layers.BatchNormalization()(x) 							# Batch Normalization
		self.relu       = lambda x: layers.LeakyReLU()(x) 									# Leaky Rectified Linear Unit

		self.kwargs     = {'kernel_size'       : (3, 2)     , 								# Convolutional Defualts 2D
					       'padding'           : 'same'     ,
					  	   'kernel_initializer': 'he_normal'}

		self.kwargs_z   = {'strides'           : (1, 2)     , 								# Convolutional Defualts 1D
						   'padding'           : 'same'     ,
						   'kernel_initializer': 'he_normal',}

		self.conv       = lambda x, filters, strides: layers.Conv2D(filters=filters,strides=strides,**self.kwargs)(x)
		self.conv_z     = lambda x, filters, k=2 : layers.Conv2D(filters=filters, kernel_size=(k, 1), **self.kwargs_z)(x)

		self.conv_str1  = lambda filters, x    : self.relu(self.norm(self.conv(x, filters, strides=(1, 1))))
		self.conv_str2  = lambda filters, x    : self.relu(self.norm(self.conv(x, filters, strides=(2, 1))))
		self.conv_dimZ  = lambda filters, k, x : self.relu(self.norm(self.conv_z(x, filters, k=k)))

		self.proj       = lambda stride, filters, x : layers.Conv2D(filters=filters, strides=stride, kernel_size=(1, 1),padding='same')(x)

		self.zoom       = lambda x: layers.UpSampling3D(size=(2, 1))(x)
		self.tran_orig  = lambda x, filters, strides : layers.Conv2DTranspose(filters=filters, strides=strides, **self.kwargs)(x)

		self.tran_str2  = lambda filters, x : self.relu(self.norm(self.tran_orig(x, filters, strides=(2, 1))))
		self.tran_strZ  = lambda filters, x : self.relu(self.norm(self.tran_orig(x, filters, strides=(1, 2))))

		## Model Parameters
		lrate           = 1e-3 																# Initial Learning Rate
		decay           = 0.90 																# Decay Rate
		dstep           = 50400 															# Steps/Decay (BatchSize * 7 (arb))
		# self.lrate      = optimizers.schedules.ExponentialDecay( 							# Exponential Learning Rate Decay
		# 						lrate, 														# Initial Learning Rate
		# 						decay_steps = dstep, 										# Steps/Decay
		# 						decay_rate  = decay, 										# Decay Rate
		# 						staircase   = True) 										# Staircase

		self.lrate      = 3e-4
		self.opt        = optimizers.Adam(learning_rate=self.lrate)                     	# Optimizer

		
		def dice(y, y_, epsilon=1):
			y_true      = cast(y, tf_float32) 												# Ground-Truth
			y_pred      = sigmoid(y_) 														# Prediction

			numerator   = (2 * reduce_sum(y_true * y_pred)) + epsilon						# Intersection * 2
			denominator = reduce_sum(y_true + y_pred) + epsilon 							# Union + 1

			return 1 - numerator / denominator

		self.loss_fx    = dice 															    # 1 - Dice Coefficient

		self.reg        = .1 																# Regularization Parameter
		self.model      = self.build_model() 												# Instantiate Model 00
		
		if domain.lower()  == 'freq':  														# Train in the Freq Domain
			self.tf_train   = self.tf_train_freq
			self.tf_valid   = self.tf_valid_freq
		elif domain.lower()  == 'time': 													# Train in the Time Domain
			self.tf_train   = self.tf_train_time
			self.tf_valid   = self.tf_valid_time

	def weighted_MSE(self, y, y_, sw): 														# Weighted Mean Squared Error
		y_    = cast(y_, tf_float64) 														# Convert Prediction to Float64
		loss  = multiply(square(subtract(y, y_)), sw) 										# weighted Squared Error
		return reduce_mean(loss) 															# Mean

	def tf_train_freq(self, fname, target='Metab'):
		with np.load(fname) as data:
			return data['x'], data[target]#, ((data['mask'] * 10) + 1)

	def tf_train_time(self, fname, target='Metab'):
		with np.load(fname) as data:
			return data['x'], data[target]#, ((data['mask'] * 10) + 1)

	def tf_valid_freq(self, fname, target='Metab', N=7200):
		with np.load(fname) as data:
			return data['x'][:N,:,:,:], data[target][:N,:,:,:]#, ((data['mask'][:N,:,:,:] * 10) + 1) 

	def tf_valid_time(self, fname, target='Metab', N=7200):
		with np.load(fname) as data:
			return data['x'][:N,:,:,:], data[target][:N,:,:,:]#, ((data['mask'][:N,:,:,:] * 10) + 1)

	def a_print(self, layer, shape, res=False):
		
		if layer[0] == '9':
			a = '\n\t\t'
		else:
			a = ''
		for ii in range(len(layer)):
			try:
				a = '{}Layer {:3d}  Shape ('.format(a, int(layer[ii]))
			except:
				a = '{}Layer {}  Shape ('.format(a, layer[ii])

			for jj in range(1, len(shape[ii])):
				a = '{} {:4d}'.format(a, shape[ii][jj])    
			if res == True:
				a = '{}) --> '.format(a)
				res = False
			else:
				a = '{})     '.format(a)
		print(a)

	def build_model(self):																							
																							# Points, Comps, Channels
		inputs = Input(shape=(2048, 2, 1), name='x0') 										#   2048,  2,   1  (In)

		l1   = self.conv_str2( 16, self.conv_str1(  8,    inputs   ))						#   1024,  2,  16  ( 1)
		l2   = self.conv_str2( 32, self.conv_str1( 16,    l1       )) 						#    512,  2,  32  ( 2)
		l3   = self.conv_str2( 64, self.conv_str1( 32,    l2       )) 						#    256,  2,  64  ( 3)
		l4   =                     self.conv_str2( 80,    l3       )						#    128,  2,  80  ( 4)
		l5   =                     self.conv_str2( 80,    l4       ) 						#     64,  2,  80  ( 5)
		l6   =                     self.conv_dimZ( 96, 2, l5       )  						#     64,  1,  96  ( 6)
		l7   =                     self.conv_str2(128,    l6       ) 						#     32,  1, 128  ( 7)
		l8   =                     self.conv_str2(128,    l7       ) 						#     16,  1, 128  ( 8)
		l9   =                     self.conv_str2(160,    l8       )						#      8,  1, 160  ( 9)
		l10  = self.conv_str2(192, self.conv_str1(160,    l9       )) 						#      4,  1, 192  (10)
		l11  = self.tran_str2(160, self.conv_str1(160,    l10      )) 						#      8,  1, 160  (11)
		l12  =                     self.tran_str2(128,    l11  + l9) 						#     16,  1, 128  (12)
		l13  =                     self.tran_str2(128,    l12  + l8) 						#     32,  1, 128  (13)
		l14  =                     self.tran_str2( 96,    l13  + l7) 						#     64,  1,  96  (14)
		l15  =                     self.tran_strZ( 80,    l14  + l6) 						#     64,  2,  80  (15)
		l16  =                     self.tran_str2( 80,    l15  + l5) 						#    128,  2,  80  (16)
		l17  =                     self.tran_str2( 64,    l16  + l4) 						#    256,  2,  64  (17)
		l18  = self.tran_str2( 32, self.conv_str1( 64,    l17  + l3)) 						#    512,  2,  32  (18)
		l19  = self.tran_str2( 16, self.conv_str1( 16,    l18  + l2)) 						#   1024,  2,  16  (19)
		l20  = self.tran_str2(  4, self.conv_str1(  8,    l19  + l1)) 						#   2048,  2,   4  (20)

		logits = layers.Conv2D(filters=1, name='y0', **self.kwargs)(l20)			    	#   2048,  2,   1  (Out)

		return Model(inputs=inputs, outputs=logits)   

	def compile_model(self, learning_rate=1e-4): 
		
		self.model.compile(optimizer = self.opt,
						   loss      = {'y0': losses.mse,
										'y1': losses.mse},
						   experimental_run_tf_function=False)

	def load_weights(self, modelname):
		fname        = '{}/{}.hdf5'.format(self.modeldir, modelname)
		self.model.load_weights(fname)

	def save_model(self, modelname):
		fname        = '{}/{}.hdf5'.format(self.modeldir, modelname)
		self.model.save(fname)

	@tfFunction
	def train_step_quick(self, x, y): 														# Custom TrainStep
		with GradientTape() as tape:               											# Define 2 Gradient Spaces
			ypred    = self.model(x, training=True) 										# Get Predictions
			loss     = self.loss_fx(y, ypred)											    # Loss 0 (Metabolite   )

		grads    = tape.gradient(loss, self.model.trainable_variables)      				# Gradients for
		self.opt.apply_gradients(zip(grads,self.model.trainable_variables))     			# Gradient Descent
		return loss

	@tfFunction
	def valid_step_quick(self, x, y): 														# Custom ValidStep
		ypred    = self.model(x, training=False) 											# Get Predictions - No Updates
		loss     = self.loss_fx(y, ypred) 											 		# Loss
		return loss

	# def gpu_memory(self,): 																	# GPU function using nvidia_smi.py Directly
	# 	nvidia_smi.nvmlInit()
	# 	handle  = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
	# 	info    = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
	# 	gpu_mem = np.array([info.used, info.free, info.total]) 

	# 	nvidia_smi.nvmlShutdown()
	# 	return gpu_mem

	def gpu_memory(self,):  																# GPU function from pynvml that wraps nvidia_smi.py
		nvmlInit()
		handle  = nvmlDeviceGetHandleByIndex(0)
		info    = nvmlDeviceGetMemoryInfo(handle)
		gpu_mem = np.array([info.used, info.free, info.total]) 

		nvmlShutdown()
		return gpu_mem

if __name__ == '__main__':

	session     =    1 																		# Current Session
	epochs      =  1800 																	# Number of Epochs
	Ntrainfiles =   25 																		# Number of Training Files 				
	Nrepeats    = epochs // 25 																# Number of Repeats through Data														
	epoch_size  = 7200 																		# Number of Training Examples per Epoch
	batch_size  =   60 																		# Batch Size
	steps       = epoch_size // batch_size

	train_step  =  120																		# Training Steps per Epoch
	valid_step  =    6 																		# Validation Steps per Epoch
	# target      = 'OOV'  																	# Network Target
	target      = 'mask'  																	# Network Target
	mask        = 'mask'  																	# Network Loss Mask
	domain      = 'Freq' 																	# Network Target Domain (Freq or Time)
	override    = True 																		# Override Previous Training Sessions/Log

	basedir     = '/mnt/hippocampus/starkdata1/Aaron_FIBRE/main/deep' 						# Base Directory
	# basedir     = '/datapool/home/agudmund/main/deep' 										# Base Directory
	trndir      = '{}/training_data/AG'.format(basedir) 									# Training Data Directory
	iodir       = '{}/InputOutput/AG_CNN_{}'.format(basedir, domain) 						# Input/Outputs Data Directory

	mname       = 'AG_OOV_{}_CNN_{}'.format(target , domain) 								# Current Model Name
	modeldir    = '{}/models/{}'.format(basedir, mname ) 									# Model Directory

	NN          = SingleCNN(
						basedir  = basedir   , 
						trndir   = trndir    , 
						modeldir = modeldir  ,
						bsize    = batch_size,
						verbose  = True      ,
						target   = target    ,
						domain   = domain    )
	
	print('\nTensorflow is Using: \n    {} inter Threads\n    {} intra Threads\n'.format(
																	threading.get_inter_op_parallelism_threads(),
																	threading.get_intra_op_parallelism_threads()))

	print('\n')
	# print(NN.model.summary())
	print('Session     : {:3d}'.format(session    ))
	print('Epochs      : {:3d}'.format(epochs     ))
	print('Train Files : {:3d}'.format(Ntrainfiles))
	print('Repeats     : {:3d}'.format(Nrepeats   ))
	print('Batch Size  : {:3d}'.format(batch_size ))
	print('Train Steps : {:3d}'.format(train_step ))
	print('Valid Steps : {:3d}'.format(valid_step ))
	print('Target      : {}'.format(   target     ))
	print('Domain      : {}'.format(   domain     ))
	print('Starting New Training Session and New Logfile   ***')
	print('\n')

	data_idx       = list(np.arange(Ntrainfiles)) 		 									# List from 0-24											
	data_idx.extend(data_idx * (Nrepeats - 1)) 												# Repeat Nrepeats times (-1 since we already created 1)
	data_idx       = np.array(data_idx).astype(np.uint8) 									# Convert to Low Memory Array

	print('Data Idx    : ', data_idx.shape     ) 

	validname     = '{}/data_AG_035_{}.npz'.format(iodir, domain) 							# Validation Data
	x,y           = NN.tf_valid_time(validname, N=1800, target=target) 						# Take 1800 Unseen Examples
	data_valid    = Dataset.from_tensor_slices((x,y)) 					 					# Create TF Dataset
	data_valid    = data_valid.batch(300) 													# 6 Batches of 300 = 1800 Total

	logname       = '{}/{}_Log.csv'.format(modeldir, mname) 								# Log File Name

	if os.path.exists(logname) and override == False: 										# Log File Exists																	
		df        = pd.read_csv(logname) 													# Log File (csv)
		nepoch    = df.shape[0] 															# Keep Track of Current Epoch When Reopening
		Loss_Best = np.min(df.LossValid)													# Best MSE --> From Previous Runtime
		idx       = np.where(df.LossValid.values == Loss_Best)[0][0] 						# Best Index
		session   = df.Session[idx] + 1 													# Current Session

		bestname  = '{}_Epoch-{:04d}_Loss-{:010.8f}'.format(mname, df.Epoch[idx],Loss_Best) # Filename for Best Validation Loss

		print('Loading Best Model...'.format(bestname)) 									# Display Loading
		NN.load_weights(bestname) 															# Load Best Validation Model
		print('Loaded {}'.format(bestname)) 												# Display Loaded Model

	else: 																					# Log File Does Not Exist
		nepoch    = 0 																		# First Epoch - Best is 0
		Loss_Best = np.zeros([6]) 															# 6 Batches of 300
		for step, (x_,y_) in enumerate(data_valid): 										# Iterate over Dataset
			Loss_Best[step] = NN.valid_step_quick(x_,y_) 									# No Weight Updates

		Loss_Best = np.mean(Loss_Best) 														# NN Outputs Validation 

		# gpu_mem   = 1
		gpu_mem   = NN.gpu_memory() 														# GPU Usage
		gpu_mem   = 100*gpu_mem[0]/gpu_mem[2] 												# GPU Usage Percent
																							# Write to Log File
		# print('{:03d}  {:7.2f}s  |  {:10.8f}  |  {:10.8f}  |  {:10.8f}   |  {:6.3f}   |   {:5.2f}  |          '.format(nepoch, 0.0, NN.lrate(0), 0.0, Loss_Best, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, gpu_mem))
		print('{:03d}  {:7.2f}s  |  {:10.8f}  |  {:10.8f}  |  {:10.8f}   |  {:6.3f}   |   {:5.2f}  |          '.format(nepoch, 0.0, 0.0, 0.0, Loss_Best, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, gpu_mem))
		logfile   = open(logname, 'w')
		logfile.write('Session,Epoch,Time,LearningRate,LossTrain,LossValid,RAM,GPU,Load_Time,Train_Time,Valid_Time,GPU_Time\n')
		# logfile.write('{},{},999,{},999,{},{},{},999,999,999,999\n'.format(session, nepoch, NN.lrate(0), Loss_Best, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, gpu_mem))
		logfile.write('{},{},999,{},999,{},{},{},999,999,999,999\n'.format(session, nepoch, 3e-4, Loss_Best, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, gpu_mem))
		logfile.close()

	## Custom Train Loop
	Loss_Train     = np.zeros([epochs]) 													# Training Loss by Epoch
	Loss_Valid     = np.zeros([epochs]) 													# Validation Loss by Epoch
	for ii in range(data_idx.shape[0]): 													# Iterate over Epochs
		epoch_time     = t0.time() 															# Time per Epoch
		Loss_Train_    = np.zeros([steps]) 													# Loss Training by Step
		Loss_Valid_    = np.zeros([    6]) 													# Loss Validation by Step
		curr_step      = (nepoch+ii) * 7200

		trainname      = '{}/data_AG_{:03d}_{}.npz'.format(iodir, data_idx[ii], domain)		# Current Train File
		
		load_time      = t0.time() 															# Time to Load	
		x,y            = NN.tf_train_time(trainname, target=target) 						# Load Training Data
		load_time      = t0.time() - load_time												# Time  to Load

		data_train     = Dataset.from_tensor_slices((x,y)) 					 				# Convert Training Data to TF Dataset
		data_train     = data_train.batch(batch_size)  										# Create Batches

		## Training
		train_time      = t0.time() 														# Time to Train
		for step, (x_,y_) in enumerate(data_train): 										# Iterate over Train Steps
			Loss_Train_[step] = NN.train_step_quick(x_,y_) 									# Fast TrainStep (Operate Directly Tensorflow on Graph)
		Loss_Train[ii] = np.mean(Loss_Train_[-20:]) 										# Calculate Mean Loss from Final 20 Steps
		train_time      = t0.time() - train_time 											# Time to Train

		## Validation
		valid_time      = t0.time() 														# Time for Validation Steps
		for step, (x_,y_) in enumerate(data_valid): 										# Iterate over Valid Steps
			Loss_Valid_[step] = NN.valid_step_quick(x_,y_)									# Fast ValidStep (Operate Directly Tensorflow on Graph)
		Loss_Valid[ii] = np.mean(Loss_Valid_) 												# Mean over Entire Batch
		valid_time      = t0.time() - valid_time 											# Time for Validation Steps

		## GPU Memory
		gpu_time      = t0.time() 															# Time for Validation Steps
		# gpu_mem       = 1
		gpu_mem       = NN.gpu_memory() 													# GPU Usage
		gpu_mem       = 100*gpu_mem[0]/gpu_mem[2] 											# GPU Usage Percent
		gpu_time      = t0.time() - gpu_time 												# Time for Validation Steps

		## Save/Report
		if Loss_Valid[ii] < Loss_Best:	 													# Save Weight if Validation Loss Improves
			Loss_Best = Loss_Valid[ii] 														# Update Best Validation Loss
			bestname  = '{}_Epoch-{:04d}_Loss-{:010.8f}'.format(mname,nepoch+ii+1,Loss_Best)# Filename
			
			if Loss_Best  < .15: # Time = .12; Freq = .15 
				NN.save_model(bestname) 														# Save

			# print('{:03d}  {:7.2f}s  |  {:10.8f}  |  {:10.8f}  |  {:10.8f}  |  {:6.3f}  |  {:5.2f}  |  ** ** **'.format(nepoch+ii+1, (t0.time()-epoch_time), NN.lrate(curr_step), Loss_Train[ii],  Loss_Valid[ii], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, gpu_mem))
			print('{:03d}  {:7.2f}s  |  {:10.8f}  |  {:10.8f}  |  {:10.8f}  |  {:6.3f}  |  {:5.2f}  |  ** ** **'.format(nepoch+ii+1, (t0.time()-epoch_time), 3e-4, Loss_Train[ii],  Loss_Valid[ii], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, gpu_mem))
			logfile   = open(logname, 'a')
			# logfile.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(session, nepoch+ii+1, (t0.time()-epoch_time),  NN.lrate(curr_step), Loss_Train[ii], Loss_Valid[ii], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_mem, load_time, train_time, valid_time, gpu_time))
			logfile.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(session, nepoch+ii+1, (t0.time()-epoch_time),  3e-4, Loss_Train[ii], Loss_Valid[ii], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_mem, load_time, train_time, valid_time, gpu_time))
			logfile.close()

		else:
			# print('{:03d}  {:7.2f}s  |  {:10.8f}  |  {:10.8f}  |  {:10.8f}  |  {:6.3f}  |  {:5.2f}  |          '.format(nepoch+ii+1, (t0.time()-epoch_time),  NN.lrate(curr_step),  Loss_Train[ii], Loss_Valid[ii], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, gpu_mem))
			print('{:03d}  {:7.2f}s  |  {:10.8f}  |  {:10.8f}  |  {:10.8f}  |  {:6.3f}  |  {:5.2f}  |          '.format(nepoch+ii+1, (t0.time()-epoch_time),  3e-4,  Loss_Train[ii], Loss_Valid[ii], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, gpu_mem))
			logfile   = open(logname, 'a')
			# logfile.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(session, nepoch+ii+1, (t0.time()-epoch_time),  NN.lrate(curr_step), Loss_Train[ii], Loss_Valid[ii], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, gpu_mem, load_time, train_time, valid_time, gpu_time))
			logfile.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(session, nepoch+ii+1, (t0.time()-epoch_time),  3e-4, Loss_Train[ii], Loss_Valid[ii], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, gpu_mem, load_time, train_time, valid_time, gpu_time))
			logfile.close()

		gc.collect()
		K.clear_session()

		# mname_       = '{}/{}_{:03d}'.format(mname, mname, ii + nepoch)					# Save Every Model
		# NN.save_model(mname_) 															# Save Every Model