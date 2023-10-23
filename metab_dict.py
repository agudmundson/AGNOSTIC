__author__ = 'Aaron Gudmundson'
__email__  = 'agudmund@uci.edu'

'''
This python script contains a dictionary of N metabolites 
commonly seen in-vivo within the human brain.

Each metabolite is contained within a sub-dictionary 
with the following:
	- Name  :  Metabolite Name  										    							# 
	- Omega :  Chemical Shift of each Nucleus (ppm) 													# Size = (N x 1)
	- nH    :  Number of Protons 																		# Size = (N x 1)
	- T2    :  T2 Decay Rate  																			# Size = (N x 1)
			   These values are Relative to Creatine to account for variability in field strength. 		#
			   Values were determined by conducting a Systematic Review of T2 in Spectroscopy.    		#
			   Then a database was developed reflecting the T2 values from different field strengths,	#
			   pulse sequences, subjects, etc...  														#
			   Finally, creatine was selected to be a normalizing value. While this is not technically 	#
			   accurate, it provides a starting point. 	 												#													
			   (More Details to come) 						 											#
	- jcoup :  j-coupling matrix in (Hz)  													 			# Size = (N x N)
			   These values reflect the values found within the FID-A and NMR-ScopeB software. 			#
			   		FID-A     : https://doi.org/10.1002/mrm.26091  								 		#
			   		NMR-ScopeB: https://doi.org/10.1016/j.ab.2016.10.007 						 		#
'''

import numpy as np                                                                          			# Arrays

metab_dict = {
			  'Ace'  : {'name'  : 'Acetate',                                               		 		# Name
			  			'omega' : np.array([ 1.9040]),  												# Chemical Shift
				     	'nH'    : np.array([    3.0]),  												# Number of Protons
				     	'T2'    : np.array([    1.0]), 													# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0]])},  												# Coupling Matrix

			  'ATP1' : {'name'  : 'AdenosineTriphosphate1',                                    		 	# Name
			  			'omega' : np.array([ 6.1260, 6.1260, 6.1260, 6.1260, 6.1260, 6.1260, 0.0000] ),	# Chemical Shfit
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0] ), # Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,]), # Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,    5.7,    0.0,    0.0,    0.0,    0.0,    0.0], 	# Coupling Matrix
									        [   0.0,    0.0,    5.3,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    3.8,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    3.0,    3.1,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,  -11.8,    1.9],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    6.5],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    4.9]])},

			  'ATP2' : {'name'  : 'AdenosineTriphosphate2',                                    		 	# Name
			  			'omega' : np.array([ 8.2240, 8.5140, 6.7550] ), 					 			# Chemical Shfit
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0]  , 					 			# Coupling Matrix
									        [   0.0,    0.0,    0.0]  , 								
									        [   0.0,    0.0,    0.0]])},								
							        		        				 		
			  'Ala'  : {'name'  : 'Alanine', 															# Name
			  			'omega' : np.array([ 1.4667, 1.4667, 1.4667, 3.7746]),  						# Chemical Shift
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate								        		        				
				     	'jcoup' : np.array([[   0.0, -14.36, -14.36,   7.23], 							# Coupling Matrix
									        [   0.0,   0.0 , -14.36,   7.23],
									        [   0.0,   0.0 ,    0.0,    0.0],
									        [   0.0,   0.0 ,    0.0,    0.0]])},
			  
			  'Asc'  : {'name'  : 'Ascorbate',                                             		 		# Name
			  			'omega' : np.array([ 4.4920, 4.0020, 3.7430, 3.7160]),  						# Chemical Shift 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate		
				     	'jcoup' : np.array([[   0.0,   2.07,    0.0,    0.0], 							# Coupling Matrix
									        [   0.0,    0.0,    6.0,    4.6],
									        [   0.0,    0.0,    0.0,  -11.5],
									        [   0.0,    0.0,    0.0,    0.0]])},
 
			  'Asp'  : {'name'  : 'Aspartate',     														# Name
			  			'omega' : np.array([ 3.8914, 2.8011,  2.6533]),  								# Chemical Shift
				     	'nH'    : np.array([    1.0,    1.0,    1.0 ]),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0 ]), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,  3.647,   9.107], 									# Coupling Matrix
									        [   0.0,    0.0,-17.3426],
									        [   0.0,    0.0,     0.0]])},										            		
			  
			  'bHB'  : {'name'  : 'betaHydroxyButyrate', 												# Name
			  			'omega' : np.array([ 2.3880, 2.2940, 4.1330, 1.1860, 1.1860, 1.1860]),  		# Chemical Shift
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,  -14.5,   7.3,     0.0,    0.0,    0.0], 			# Coupling Matrix
									        [   0.0,    0.0,   6.3,     0.0,    0.0,    0.0],
									        [   0.0,    0.0,   0.0,     6.3,    6.3,    6.3],
									        [   0.0,    0.0,   0.0,     0.0,    0.0,    0.0],
									        [   0.0,    0.0,   0.0,     0.0,    0.0,    0.0],
									        [   0.0,    0.0,   0.0,     0.0,    0.0,    0.0]])},	

			  'bHG'  : {'name'  : 'betaHydroxyGlutarate', 												# Name
			  			'omega' : np.array([ 4.0220, 1.8250, 1.9770, 2.2210, 2.2720]),  				# Chemical Shift
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,    7.6,    4.1,    0.0,    0.0], 					# Coupling Matrix
									        [   0.0,    0.0,  -14.0,    5.3,   10.4],
									        [   0.0,    0.0,    0.0,   10.6,    6.0],
									        [   0.0,    0.0,    0.0,    0.0,  -15.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0]])},
			  
			  'Cit'  : {'name'  : 'Citrate', 															# Name
			  			'omega' : np.array([ 2.5400, 2.6500]),  										# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0] ),  										# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0] ), 										# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,  -15.1], 											# Coupling Matrix
									        [   0.0,   0.0 ]])},	

			  'GPC1' : {'name'  : 'GlyceroPhosphoCholine1', 											# Name
			  			'omega' : np.array([ 3.6050, 3.6720, 3.9030, 3.8710, 3.9460, 0.0000]),  		# Chemical Shift
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0, -14.78,   5.77,    0.0,    0.0,    0.0],  			# Coupling Matrix
									        [   0.0,    0.0,   4.53,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,   5.77,   4.53,    0.0],
									        [   0.0,    0.0,    0.0,    0.0, -14.78,   6.03],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,   6.03],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0]])},	

			  'GPC2' : {'name'  : 'GlyceroPhosphoCholine2', 											# Name
			  			'omega' : np.array([ 4.3120, 4.3120, 3.6590, 3.6590, 0.0000, 0.0000]),  		# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,  -9.32,   3.10,   5.90,   2.67,   6.03], 			# Coupling Matrix
									        [   0.0,    0.0,   5.90,   3.10,   2.67,   6.03],
									        [   0.0,    0.0,    0.0,  -9.32,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0]])},			 

			  'GPC3' : {'name'  : 'GlyceroPhosphoCholine3', 											# Name
			  			'omega' : np.array([ 3.2120, 3.2120, 3.2120, 0.0000]),  						# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0,   0.57], 							# Coupling Matrix
									        [   0.0,    0.0,    0.0,   0.57],
									        [   0.0,    0.0,    0.0,   0.57],
									        [   0.0,    0.0,    0.0,    0.0]])},

			  'Cho'  : {'name'  : 'Choline', 															# Name
			  			'omega' : np.array([ 3.1850, 4.0540, 4.0540, 3.5010, 3.5010, 0.0000]),  		# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0,    0.0,    0.0,   0.57],	 		# Coupling Matrix
									        [   0.0,    0.0, -14.10,   3.14,  6.979, 2.5720],
									        [   0.0,    0.0,    0.0,  7.011,  3.168, 2.6811],
									        [   0.0,    0.0,    0.0,    0.0, -14.07,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0]])},

			  'Cre1' : {'name'  : 'Creatine1',		 													# Name
			  			'omega' : np.array([ 6.6490, 3.9130, 3.9130]),  								# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0], 									# Coupling Matrix
									        [   0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0]])},		

			  'Cre2' : {'name'  : 'Creatine2', 															# Name
			  			'omega' : np.array([ 3.0270, 3.0270, 3.0270]),  								# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0],									# Coupling Matrix
									        [   0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0]])},	

			  'Cys'  : {'name'  : 'Cysteine', 														  	# Name
			  			'omega' : np.array([ 4.5608, 2.9264, 2.9747]),  		  						# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,   7.09,   4.71],									# Coupling Matrix
									        [   0.0,    0.0, -14.06],
									        [   0.0,    0.0,    0.0]])},	

			  'ETA'  : {'name' : 'EthanolAmine', 														# Name
			  			'omega' : np.array([ 3.8184, 3.8184, 3.1467, 3.1467, 0.0000]),  				# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,-10.640,  3.897,  6.794,  0.657],					# Coupling Matrix
									        [   0.0,    0.0,  6.694,  3.798,  0.142],
									        [   0.0,    0.0,    0.0,-11.710,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0]])},										        								        								        					   			  

			  'EtOH' : {'name' : 'EthylAlcohol', 														# Name
			  			'omega' : np.array([ 1.1900, 1.1900, 1.1900, 3.6700, 3.6700]),    				# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0,    7.1,    7.1],					# Coupling Matrix
									        [   0.0,    0.0,    0.0,    7.1,    7.1],
									        [   0.0,    0.0,    0.0,    7.1,    7.1],
									        [   0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0]])},										        								        								        					   
			  
			  'GABA' : {'name' : 'gammaAminoButyricAcid',												# Name
			  			'omega' : np.array([ 2.2840, 2.2840, 1.8880, 1.8880, 3.0130, 3.0130]),  		# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,-15.938,  7.678,  6.980,    0.0,    0.0],			# Coupling Matrix
									        [   0.0,    0.0,  6.980,  7.678,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,  -15.0,  8.510,  6.503],
									        [   0.0,    0.0,    0.0,    0.0,  6.503,  8.510],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,-14.062],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],])},
			  
			  'GABG' : {'name' : 'gammaAminoButyricAcid_Gov',											# Name
			  			'omega' : np.array([ 2.2840, 2.2840, 1.8890, 1.8890, 3.0128, 3.0128]),  		# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,    0.0, 5.3720, 7.1270,    0.0,    0.0],			# Coupling Matrix
									        [   0.0,    0.0, 10.578, 6.9820,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0, 7.7550, 7.4320],
									        [   0.0,    0.0,    0.0,    0.0, 6.1730, 7.9330],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],])},

			  'GlcA' : {'name'  : 'GlucoseAlpha', 														# Name
			  			'omega' : np.array([ 5.2160, 3.5190, 3.6980, 3.3950, 3.8220, 3.8260, 3.7490]),  # Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 	# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 	# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,    3.8,    0.0,    0.0,    0.0,    0.0,    0.0], 	# Coupling Matrix
									        [   0.0,    0.0,    9.6,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    9.4,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    9.9,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    1.5,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0, -12.10],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0]])},

			  'GlcB' : {'name'  : 'GlucoseBeta', 														# Name
			  			'omega' : np.array([ 4.6300, 3.2300, 3.4730, 3.3870, 3.4500, 3.8820, 3.7070]),  # Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 	# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 	# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,    8.0,    0.0,    0.0,    0.0,    0.0,    0.0], 	# Coupling Matrix
									        [   0.0,    0.0,    9.1,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    9.4,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    8.9,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    1.6,    5.4],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0, -12.30],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0]])},				   
			  
			  'Gln1' : {'name'  : 'Glutamine1', 														# Name
			  			'omega' : np.array([ 3.7530, 2.1290, 2.1090, 2.4320, 2.4540]),  				# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,  5.847,   6.50,    0.0,    0.0], 					# Coupling Matrix
									        [   0.0,    0.0,-14.504, 9.1650, 6.3470],
									        [   0.0,    0.0,    0.0, 6.3240, 9.2090],
									        [   0.0,    0.0,    0.0,    0.0,-15.371],
									        [   0.0,    0.0,    0.0,    0.0,    0.0]])},			     	

			  'Gln2' : {'name' : 'Glutamine2', 															# Name
			  			'omega' : np.array([ 6.8160, 7.5290]),  										# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0] ),  										# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0] ), 										# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0], 											# Couping Matrix
									        [   0.0,    0.0]])},	

			  'Glu'  : {'name'  : 'Glutamate', 															# Name
			  			'omega' : np.array([ 3.7433, 2.0375, 2.1200, 2.3378, 2.3520]),  				# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0, 7.3310, 4.6510,    0.0,    0.0], 					# Coupling Matrix
									        [   0.0,    0.0,-14.849, 6.4130, 8.4060],
									        [   0.0,    0.0,    0.0, 8.4780, 6.8750],
									        [   0.0,    0.0,    0.0,    0.0,-15.915],
									        [   0.0,    0.0,    0.0,    0.0,    0.0]])},

			  'GSH1' : {'name'  : 'Glutathione1', 														# Name
			  			'omega' : np.array([ 3.7690, 2.1590, 2.1460, 2.5100, 2.5600]),  				# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,   6.34,   6.36,    0.0,    0.0], 					# Coupling Matrix
									        [   0.0,    0.0, -15.48,    6.7,    7.6],
									        [   0.0,    0.0,    0.0,    7.6,    6.7],
									        [   0.0,    0.0,    0.0,    0.0, -15.92],
									        [   0.0,    0.0,    0.0,    0.0,    0.0]])},	

			  'GSH2' : {'name'  : 'Glutathione2', 														# Name
			  			'omega' : np.array([ 4.5608, 2.9264, 2.9747]),  								# Chemical Shift
				     	'nH'    : np.array([    1.0,    1.0,    1.0]),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0]), 									# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,   7.09,   4.71],
									        [   0.0,    0.0, -14.06],
									        [   0.0,    0.0,    0.0]])},	

			  'GSH3' : {'name'  : 'Glutathione3',
			  			'omega' : np.array([ 3.7690, 7.1540, 8.1770]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0]])},	
			
			  'Glyn' : {'name'  : 'Glycine',
			  			'omega' : np.array([ 3.5480, 3.5480]), 
				     	'nH'    : np.array([    1.0,    1.0] ),  										# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0] ), 										# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0],
									        [   0.0,    0.0]])},									        				     	

			  'Glyc' : {'name'  : 'Glycerol', 															# Name
			  			'omega' : np.array([ 3.5522, 3.6402, 3.7704, 3.5522, 3.6402]), 					# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,-11.715,  4.427,    0.0,    0.0],
									        [   0.0,    0.0,  6.485,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,  4.427,  6.485],
									        [   0.0,    0.0,    0.0,    0.0,-11.715],
									        [   0.0,    0.0,    0.0,    0.0,    0.0]])},	

			  'H2O'  : {'name'  : 'Water', 																# Name
			  			'omega' : np.array([ 4.7000, 4.7000]),  										# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0] ),  										# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0] ), 										# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0],
									        [   0.0,    0.0]])},	
			  
			  'Hist' : {'name'  : 'Histamine', 															# Name
			  			'omega' : np.array([ 2.9813, 2.9897, 3.2916, 3.2916, 7.8520, 7.0940, 7.5000]), 	# Chemical Shift
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 	# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 	# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0, -16.12,  6.868,  8.147,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,  7.001,  6.270,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,-14.145,    0.0,   0.73,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,   0.73,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0,   1.07],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0,   1.19],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0]])},	
			  
			  'Hisd' : {'name'  : 'Histidine',
			  			'omega' : np.array([ 3.9752, 3.1195, 3.2212]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0, -7.959,  4.812],
									        [   0.0,    0.0,-15.513],
									        [   0.0,    0.0,    0.0]])},

			  'HCr1' : {'name'  : 'HomoCarnosine1',
			  			'omega' : np.array([ 4.4720, 6.3970]), 
				     	'nH'    : np.array([    1.0,    1.0] ),  										# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0] ), 										# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,   6.88],
									        [   0.0,    0.0]])},	

			  'HCr2' : {'name'  : 'HomoCarnosine2',
			  			'omega' : np.array([ 3.1850, 3.0030, 7.7075, 8.0810]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0]])},	

			  'HCr3' : {'name'  : 'HomoCarnosine3',
			  			'omega' : np.array([ 2.9620, 1.8910, 2.3670, 7.8990]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0]])},					     	

			  'Lac'  : {'name'  : 'Lactate',
			  			'omega' : np.array([ 4.0974, 1.3142, 1.3142, 1.3142]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0, 6.9330, 6.9330, 6.9330],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0]])},	
			  
			  'Myo'  : {'name'  : 'MyoInositol', 														# Name
			  			'omega' : np.array([ 3.5217, 4.0538, 3.5217, 3.6144, 3.2690, 3.6144]),  		# Chemial Shift
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0, 2.8890,    0.0,    0.0,    0.0, 9.9980],
									        [   0.0,    0.0, 3.0060,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0, 9.9970,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0, 9.4850,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0, 9.4820],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0]])},             

			  'NAA1' : {'name'  : 'NAcetylAspartate1',
			  			'omega' : np.array([ 2.0080, 2.0080, 2.0080]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0]])},	

			  'NAA2' : {'name'  : 'NAcetylAspartate2',
			  			'omega' : np.array([ 7.8205, 4.3817, 2.6727, 2.4863]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,  6.400,    0.0,    0.0],
									        [   0.0,    0.0, 3.8610, 9.8210],
									        [   0.0,    0.0,    0.0,-15.592],
									        [   0.0,    0.0,    0.0,    0.0]])},					     	
				     
			  'NAG1' : {'name'  : 'NAcetylAspartateGlutamate1',
			  			'omega' : np.array([ 2.0420, 2.0420, 2.0420]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0]])},

			  'NAG2' : {'name'  : 'NAcetylAspartateGlutamate2',
			  			'omega' : np.array([ 4.6070, 2.7210, 2.5190, 8.2600]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0, 4.4120, 9.5150,  7.320],
									        [   0.0,    0.0,-15.910,    0.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0]])},
			  
			  'NAG3' : {'name'  : 'NAcetylAspartateGlutamate3',
			  			'omega' : np.array([ 4.1280, 2.0490, 1.8810, 2.1800, 2.1900, 7.9500]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]),			# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,  4.610,  8.420,    0.0,    0.0, 7.4600],
									        [   0.0,    0.0,-14.280, 10.560, 6.0900,    0.0],
									        [   0.0,    0.0,    0.0, 4.9000, 11.110,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,-15.280,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0]])},   

			  'PCh1' : {'name'  : 'PhosphoCholine1', 													# Name
			  			'omega' : np.array([ 4.2805, 4.2805, 3.6410, 3.6410, 0.0000, 0.0000]),  		# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]),			# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,-14.890,  2.284,  7.231,  2.860,  6.298],
									        [   0.0,    0.0,  7.326,  2.235,  2.772,  6.249],
									        [   0.0,    0.0,    0.0, -14.19,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0]])},  

			  'PCh2' : {'name'  : 'PhosphoCholine2', 													# Name
			  			'omega' : np.array([ 3.2080]),  												# Chemical Shift
				     	'nH'    : np.array([    9.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},	

			  'PCr1' : {'name'  : 'PhosphoCreatine2', 													# Name
			  			'omega' : np.array([ 3.9300, 3.9300, 6.5810, 7.2960]),  						# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0]])},		

			  'PCr2' : {'name'  : 'PhosphoCreatine2', 													# Name
			  			'omega' : np.array([ 3.0290]),  												# Chemical Shift (ppm)
				     	'nH'    : np.array([    3.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},	

			  'PhE'  : {'name'  : 'PhosphorylEthanolamine', 											# Name
			  			'omega' : np.array([ 3.9765, 3.9765, 3.2160, 3.2160, 0.0000, 0.0000]), 			# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]),			# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0,    1.0]), 			# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,-14.560, 3.1820, 6.7160, 0.4640, 7.2880],
									        [   0.0,    0.0, 7.2040,  2.980, 0.5880, 7.0880],
									        [   0.0,    0.0,    0.0,-14.710,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0]])},   

			  'Phy1' : {'name'  : 'PhenylAlanine1', 													# Name
			  			'omega' : np.array([ 7.3223, 7.4201, 7.3693, 7.4201, 7.3223]), 					# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]),					# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,  7.909,  1.592,  0.493,  1.419],
									        [   0.0,    0.0,  7.204,  0.994,  0.462],
									        [   0.0,    0.0,    0.0,  7.534,  0.970],
									        [   0.0,    0.0,    0.0,    0.0,  7.350],
									        [   0.0,    0.0,    0.0,    0.0,    0.0]])},	

			  'Phy2' : {'name'  : 'PhenylAlanine2', 													# Name
			  			'omega' : np.array([ 3.9753, 3.2734, 3.1049]),  								# Chemical Shift
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,  5.209,  8.013],
									        [   0.0,    0.0,-14.573],
									        [   0.0,    0.0,    0.0]])},										        								        	

			  'Scy'  : {'name'  : 'ScylloInositol',     												# Name
			  			'omega' : np.array([ 3.3400]),   												# Chemical Shift (ppm)
				     	'nH'    : np.array([    6.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},	 											# Coupling Matrix

			  'Ser'  : {'name'  : 'Serine',
			  			'omega' : np.array([ 3.8347, 3.9379, 3.9764]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0, 5.9790, 3.5610],
									        [   0.0,    0.0,-12.254],
									        [   0.0,    0.0,    0.0]])},

			  'Tau'  : {'name'  : 'Taurine',   															# Name
			  			'omega' : np.array([ 3.4206, 3.4206, 3.2459, 3.2459]), 							# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,    0.0, 6.7420, 6.4640],
									        [   0.0,    0.0, 6.4030, 6.7920],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0]])},				     	

			  'Thr'  : {'name'  : 'Threonine',
			  			'omega' : np.array([ 3.5785, 4.2464, 1.1358, 1.1358, 1.1358]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]),					# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0,    1.0]), 					# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,  4.917,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,   6.35,   6.35,   6.35],
									        [   0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0,    0.0]])},
			  
			  'Try1' : {'name'  : 'Tryptophan1',
			  			'omega' : np.array([ 4.0468, 3.4739, 3.2892, 7.3120]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,  4.851,  8.145,    0.0],
									        [   0.0,    0.0,-15.368,    0.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0]])},	
			  
			  'Try2' : {'name'  : 'Tryptophan2',
			  			'omega' : np.array([ 7.7260, 7.2788, 7.1970, 7.5360]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,   7.60,    1.0,  0.945],
									        [   0.0,    0.0,  7.507,   1.20],
									        [   0.0,    0.0,    0.0,  7.677],
									        [   0.0,    0.0,    0.0,    0.0]])},

			  'Tyr1' : {'name'  : 'Tyrosine1',
			  			'omega' : np.array([ 3.9281, 3.1908, 3.0370]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0] ),  								# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0] ), 								# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0, 5.1470, 7.8770],
									        [   0.0,    0.0,-14.726],
									        [   0.0,    0.0,    0.0]])},

			  'Tyr2'  : {'name' : 'Tyrosine2',
			  			'omega' : np.array([ 7.1852, 6.8895, 6.8895, 7.1852]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0, 7.9810, 0.3110, 2.5380],
									        [   0.0,    0.0, 2.4450, 0.4600],
									        [   0.0,    0.0,    0.0, 8.6490],
									        [   0.0,    0.0,    0.0,    0.0]])},

			  'Val'  : {'name'  : 'Valine',
			  			'omega' : np.array([ 3.5953, 2.2577, 1.0271, 0.9764]), 
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,  4.405,    0.0,    0.0],
									        [   0.0,    0.0,  6.971,  7.071],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0]])},

			  'MM092': {'name'  : 'MM_092',
			  			'omega' : np.array([ 0.9200]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM121': {'name'  : 'MM_121',
			  			'omega' : np.array([ 1.2100]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},


			  'MM139': {'name'  : 'MM_139',
			  			'omega' : np.array([ 1.3900]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM167': {'name'  : 'MM_167',
			  			'omega' : np.array([ 1.6700]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},


			  'MM204': {'name'  : 'MM_204',
			  			'omega' : np.array([ 2.0400]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM226': {'name'  : 'MM_226',
			  			'omega' : np.array([ 2.2600]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM256': {'name'  : 'MM_256',
			  			'omega' : np.array([ 2.5600]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM270': {'name'  : 'MM_270',
			  			'omega' : np.array([ 2.7000]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM299': {'name'  : 'MM_299',
			  			'omega' : np.array([ 2.9900]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM321': {'name'  : 'MM_321',
			  			'omega' : np.array([ 3.2100]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM362': {'name'  : 'MM_362',
			  			'omega' : np.array([ 3.6200]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM375': {'name'  : 'MM_375',
			  			'omega' : np.array([ 3.7500]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM386': {'name'  : 'MM_386',
			  			'omega' : np.array([ 3.8600]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'MM403': {'name'  : 'MM_403',
			  			'omega' : np.array([ 4.0300]), 
				     	'nH'    : np.array([    1.0] ),  												# Number of Protons
				     	'T2'    : np.array([    1.0] ), 												# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0]])},

			  'Doub' : {'name'  : 'Generic Doublet', 													# Name
			  			'omega' : np.array([ 4.0000, 2.0000]),  										# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0] ),  										# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0] ), 										# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0, 10.000],
									        [   0.0,    0.0]])},	

			  'Trip' : {'name'  : 'Generic Triplet', 													# Name
			  			'omega' : np.array([ 4.0000, 2.0000, 2.0000]),  								# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0]),									# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0]), 									# Relative T2 Decay Rate	
				     	'jcoup' : np.array([[   0.0,   10.0,   10.0],
									        [   0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0]])},	
 
			  'Quar' : {'name'  : 'Generic Quartet', 													# Name
			  			'omega' : np.array([ 4.0000, 2.0000, 2.0000, 2.0000]),  						# Chemical Shift (ppm)
				     	'nH'    : np.array([    1.0,    1.0,    1.0,    1.0]),							# Number of Protons
				     	'T2'    : np.array([    1.0,    1.0,    1.0,    1.0]), 							# Relative T2 Decay Rate
				     	'jcoup' : np.array([[   0.0,   10.0,   10.0,   10.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0],
									        [   0.0,    0.0,    0.0,    0.0]])},
			}