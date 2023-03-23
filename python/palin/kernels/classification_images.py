#!/usr/bin/env python
'''
PALIN toolbox v0.1
Decemberr 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

import pandas as pd
import numpy as np

def compute_kernel(data_df,trial_ids=['experimentor','type','subject','session'], dimension_ids=['segment'],response_id='response', value_id='pitch'):
	''' computes first-order temporal kernels for each participant using the classification image, ie. 
	mean(stimulus features classified as positive) - mean(stimulus features classified as negative)'''

	# for each participant, average stimulus features (e.g. mean pitch for each segment) separately for positive and negative responses, and subtract positives - negatives 
	dimension_mean_value = data_df.groupby(trial_ids+[dimension_ids]+[response_id])[value_id].mean().reset_index()
	positives = dimension_mean_value.loc[dimension_mean_value[response_id] == True].reset_index()
	negatives = dimension_mean_value.loc[dimension_mean_value[response_id] == False].reset_index()
	kernels = pd.merge(positives, negatives, on=trial_ids+[dimension_ids])
	kernels['kernel_value'] = kernels['%s_x'%value_id] - kernels['%s_y'%value_id]

	# Kernel are then normalized for each participant/session by dividing them 
	# by the square root of the sum of their squared values.
	kernels['square_value'] = kernels['kernel_value']**2
	for_norm = kernels.groupby(trial_ids)['square_value'].mean().reset_index()
	kernels = pd.merge(kernels, for_norm, on=trial_ids)
	kernels['norm_value'] = kernels['kernel_value']/np.sqrt(kernels['square_value_y'])
	kernels.drop(columns=['index_x','%s_x'%response_id,'%s_x'%value_id,'index_y','%s_y'%response_id,'%s_y'%value_id,'square_value_x', 'square_value_y'], inplace=True)
	
	return kernels

def compute_accuracy(data_df, control_kernel, session_identifiers = ['experimentor','type','subject','session'], trial_identifier = 'trial', stimulus_dimension='segment', stimulus_value = 'pitch', stimulus_response='response'): 
	''' Computes participant's accuracy on the task as a measure of how well the participant performs 
	compared to an ideal participant model having the control group as an internal representation and zero internal noise.
	Accuracy therefore combines both internal representation and noise in a single measure.''' 

	# for each participant, in each trial, compute the dot product of each stimulus with the control group kernel
	
	# create a df of positive and negative trial data for each participant
	positives = data_df.loc[data_df[stimulus_response] == 1].reset_index()[session_identifiers + [trial_identifier,stimulus_dimension,stimulus_value]]
	negatives = data_df.loc[data_df[stimulus_response] == 0].reset_index()[session_identifiers + [trial_identifier,stimulus_dimension,stimulus_value]]

	trial_data = positives.merge(negatives, 
                      on=session_identifiers + [trial_identifier,stimulus_dimension],
                     suffixes=('_pos','_neg'))

	# dot product of each positive and negative trial with the control group's kernel
	def dot_control(x):
		#print(list(x))
		#print(control_kernel)
		return np.dot(list(x), control_kernel)

	trial_data = trial_data.groupby(['experimentor','type','subject','session']+[trial_identifier]).agg({'%s_pos'%stimulus_value:dot_control,
																										'%s_neg'%stimulus_value:dot_control}).reset_index()

	# count hits as trials for which the positive stimuli is the one with higher dot product to control kernel
	trial_data['hit'] = trial_data['%s_pos'%stimulus_value] > trial_data['%s_neg'%stimulus_value] 

	# compute hit rate (average hit across trials) per participant
	hit_rate = trial_data.groupby(session_identifiers, as_index=False).hit.mean()

	return hit_rate



