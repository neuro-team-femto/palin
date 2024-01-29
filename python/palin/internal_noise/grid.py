#!/usr/bin/env python
'''
PALIN toolbox v0.1
sep 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for Monte Carlo method for estimating internal noise
'''

import pandas as pd
import numpy as np
from palin.utils import utils


def estimate_model(internal_noise_bounds=[0,10,0.1],criteria_bounds=[-5,5,0.1],n_trials=10000,n_blocks=5):

	'''This will estimate a model mapping input parameters to output parameters
	'''
	criterias = np.round(np.arange(criteria_bounds[0],criteria_bounds[1]+criteria_bounds[2],criteria_bounds[2]),decimals=2)

	internal_noise_sigmas = np.round(np.arange(internal_noise_bounds[0],internal_noise_bounds[1]+internal_noise_bounds[2],internal_noise_bounds[2]),decimals=2)

	all_criteria = []
	all_internal_noise_sigma = []
	all_prob_interval1 = []
	all_prob_agreement = []

	for criteria in criterias:
		for internal_noise_sigma in internal_noise_sigmas:			

			# simulate observer with (criteria, internal_noise_sigma)
			prob_agreement,prob_interval1= utils.simulate_observer(internal_noise_sigma,criteria, n_blocks, n_trials)

			all_criteria.append(criteria)
			all_internal_noise_sigma.append(internal_noise_sigma)
			all_prob_interval1.append(prob_interval1)
			all_prob_agreement.append(prob_agreement)

	model = pd.DataFrame({'criteria':all_criteria,
						'internal_noise_sigma': all_internal_noise_sigma,
						'prob_interval1':all_prob_interval1,
						'prob_agreement': all_prob_agreement})

	return model

def invert_model(model, prob_agreement, prob_interval1): 
	'''This will find best input parameters for a given output pair
	'''

	model['distance'] = (model.prob_interval1-prob_interval1)**2 + \
						(model.prob_agreement-prob_agreement)**2

	[criteria,internal_noise_sigma,best_prob_interval1,best_prob_agreement,distance] = np.array(model[model.distance==model.distance.min()])[0]

	
	return criteria,internal_noise_sigma,best_prob_agreement,best_prob_interval1


