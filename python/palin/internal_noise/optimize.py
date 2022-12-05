#!/usr/bin/env python
'''
PALIN toolbox v0.1
sep 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for optimization method for estimating internal noise
'''

import pandas as pd
from palin.utils import utils

def cost_function(target_prob_agreement,target_prob_interval_1,internal_noise_sigma, criteria,n_blocks = 5, n_trials = 1000): 
	'''This will estimate a model mapping input parameters to output parameters
	'''

	print('This will run a simulation with internal_noise and criteria as internal parameter, '+ 
		'and compute the distance between the estimated and target prob_agreement,prob_interval_1')

	est_prob_agreement,est_prob_interval1= utils.simulate_observer(internal_noise_sigma,criteria, n_blocks, n_trials)

	return (est_prob_agreement-target_prob_agreement)**2 + \
						(est_prob_interval1-target_prob_interval_1)**2

def optimize(target_prob_agreement, target_prob_interval_1,internal_noise_bounds,criteria_bounds): 
	'''This will find best input parameters for a given output pair
	'''

	print('This runs an optimization method (ex. simplex) to find the internal_noise and criteria values,'+
		' within internal_noise_bounds &criteria_bounds, which minimizes the cost_function re: prob_agreement, prob_interval_1')
	internal_noise = 0
	criteria = 0

	return internal_noise, criteria


