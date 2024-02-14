#!/usr/bin/env python
'''
PALIN toolbox v0.1
Decemberr 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

import pandas as pd
import numpy as np
from .kernel_analyser import KernelAnalyser

class ClassificationImage(KernelAnalyser):

    @classmethod
    def extract_single_kernel(cls, data_df, feature_id = 'feature', value_id = 'value', response_id = 'response'):
        feature_average = data_df.groupby([feature_id,response_id])[value_id].mean().reset_index()
        positives = feature_average.loc[feature_average[response_id] == True].reset_index()
        negatives = feature_average.loc[feature_average[response_id] == False].reset_index()
        kernels = pd.merge(positives, negatives, on=feature_id, suffixes=('_true','_false'))
        kernels['kernel_value'] = kernels['%s_true'%value_id] - kernels['%s_false'%value_id]
        kernels = kernels[[feature_id,'kernel_value']].set_index(feature_id)
        kernels.index.names = ['feature']
        return kernels

        
# def compute_accuracy(data_df, control_kernel, session_identifiers = ['experimentor','type','subject','session'], trial_identifier = 'trial', stimulus_dimension='segment', stimulus_value = 'pitch', stimulus_response='response'): 
#   ''' Computes participant's accuracy on the task as a measure of how well the participant performs 
#   compared to an ideal participant model having the control group as an internal representation and zero internal noise.
#   Accuracy therefore combines both internal representation and noise in a single measure.''' 

#   # for each participant, in each trial, compute the dot product of each stimulus with the control group kernel
    
#   # create a df of positive and negative trial data for each participant
#   positives = data_df.loc[data_df[stimulus_response] == 1].reset_index()[session_identifiers + [trial_identifier,stimulus_dimension,stimulus_value]]
#   negatives = data_df.loc[data_df[stimulus_response] == 0].reset_index()[session_identifiers + [trial_identifier,stimulus_dimension,stimulus_value]]

#   trial_data = positives.merge(negatives, 
#                       on=session_identifiers + [trial_identifier,stimulus_dimension],
#                      suffixes=('_pos','_neg'))

#   # dot product of each positive and negative trial with the control group's kernel
#   def dot_control(x):
#       #print(list(x))
#       #print(control_kernel)
#       return np.dot(list(x), control_kernel)

#   trial_data = trial_data.groupby(session_identifiers+[trial_identifier]).agg({'%s_pos'%stimulus_value:dot_control,
#                                                                                                       '%s_neg'%stimulus_value:dot_control}).reset_index()

#   # count hits as trials for which the positive stimuli is the one with higher dot product to control kernel
#   trial_data['hit'] = trial_data['%s_pos'%stimulus_value] > trial_data['%s_neg'%stimulus_value] 

#   # compute hit rate (average hit across trials) per participant
#   hit_rate = trial_data.groupby(session_identifiers, as_index=False).hit.mean()

#   return hit_rate



