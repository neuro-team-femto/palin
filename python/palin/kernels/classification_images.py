#!/usr/bin/env python
'''
PALIN toolbox v0.1
Decemberr 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

import pandas as pd
import numpy as np

def compute_kernel(data_df,trial_ids=['experimentor','type','subject','session'], dimension_ids=['segment'],response_id='response', value_id='pitch'):
	# we compute first-order temporal Kernel(5-points vector) for each subject as the
	# mean(Parameter classified as interrogative) - (mean P  classified as non-interrogative)
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



