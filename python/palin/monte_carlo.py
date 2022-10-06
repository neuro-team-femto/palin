#!/usr/bin/env python
'''
PALIN toolbox v0.1
sep 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for Monte Carlo method for estimating internal noise
'''

import pandas as pd

def estimate_model(internal_noise_bounds, criteria_bounds): 
	'''This will estimate a model mapping input parameters to output parameters
	'''

	print('This will estimate a model mapping input parameters to output parameters')
	model = pd.DataFrame()

	return model

def invert_model(model, prob_agreement, prob_interval_1): 
	'''This will find best input parameters for a given output pair
	'''

	print('This will find best input parameters for a given output pair')
	internal_noise = 0
	criteria = 0

	return internal_noise, criteria


