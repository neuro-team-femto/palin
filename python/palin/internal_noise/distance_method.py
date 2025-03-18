#!/usr/bin/env python

import pandas as pd
import numpy as np
import os.path
import warnings
import ast
from .agreement_method import AgreementMethod
from itertools import combinations
from abc import ABC,abstractmethod



class DistanceMethod(AgreementMethod):
    ''' 
    This class implements an AgreementMethod to estimate internal noise and criteria from reverse correlation data. 
    Contrary to the DoublePass method, it does not require that the data has double pass trials. 
    Prob_agree is estimated by a measure of how much actual responses are consistent with ideal responses based on the kernel
    Prob_first is estimated on the complete set of trials.
    '''

    def __str__(self): 
        return 'Distance method'

    @classmethod
    def compute_probabilities(cls,data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs):
        '''
        Compute probabilities over non-double pass trials
        '''
        # Keep only non double_pass trials, or the first occurrence of double_pass trials
        double_pass_id = 'double_pass_id' # column by which to identify double pass trials
        data_df = cls.index_double_pass_trials(data_df, trial_id=trial_id, value_id = value_id, double_pass_id = double_pass_id)
        single_pass_df = cls.keep_single_pass(data_df, trial_id=trial_id, double_pass_id = double_pass_id)
        
        # compute probability of agreement
        prob_agree = cls.compute_prob_agreement(single_pass_df, trial_id=trial_id, 
            response_id=response_id, feature_id= 'feature', value_id = 'value', **kwargs)
            # kernel_extractor=kwargs['kernel_extractor'], consistency_type = kwargs['consistency_type'])
        # compute probability of choosing first response option
        prob_first = cls.compute_prob_first(single_pass_df, trial_id=trial_id, response_id=response_id, stim_id=stim_id, **kwargs)
        return prob_agree, prob_first

    @classmethod
    def compute_accuracy(cls,data_df, trial_id='trial', stim_id= 'stim', feature_id= 'feature', value_id = 'value', response_id='response', **kwargs):
        '''
        Estimates prob agreement using the accuracy method, which computes the extent to which observer responses match those of an ideal observer with the same kernel and no internal noise
        '''
        if 'kernel_extractor' not in kwargs:
            raise TypeError('DistanceMethod missing required argument kernel_extractor')
        kernel_extractor = kwargs['kernel_extractor']

        if 'consistency_type' not in kwargs:
            raise TypeError('DistanceMethod missing required argument consistency_type')
        consistency_type = kwargs['consistency_type']

        if 'weight_by_activation' not in kwargs: 
            raise TypeError('DistanceMethod missing required argument weight_by_activation')
        weight_by_activation = kwargs['weight_by_activation']

        # pivot data to have one entry per trials (instead of n_features entries) 
        trials_df = data_df.groupby([trial_id,stim_id]).agg({value_id:list, response_id:'first'}).reset_index()
        trials_df[value_id] = trials_df[value_id].apply(lambda x: np.array(x))  

        # compute stimulus difference in each trial, and response = 0 or 1 (first or second)
        trials_df = trials_df.groupby([trial_id]).agg({value_id:lambda x:x.diff().iloc[1], #diff produces 2 lines, the first is nan
                           response_id: lambda x: 0 if x.iloc[0] else 1}).reset_index()

        # compute scalar product with kernel
        kernel = kernel_extractor.extract_single_kernel(data_df, trial_id,stim_id, feature_id, value_id, response_id)
        trials_df['activation']= trials_df[value_id].apply(lambda x: x.dot(list(kernel.kernel_value)))   

        # accurate if response consistent with kernel product. hit if correctly responds second, cr if correcty responds first
        if consistency_type == 'accuracy':
            trials_df['consistency'] = (trials_df.activation > 0) == (trials_df.response==1)
        elif consistency_type == 'hit':
            trials_df['consistency'] = (trials_df.activation > 0) & (trials_df.response==1)
            trials_df.loc[trials_df.activation < 0, 'consistency'] = np.nan
        elif consistency_type== 'cr':
            trials_df['consistency'] = (trials_df.activation < 0) & (trials_df.response==0)
            trials_df.loc[trials_df.activation > 0, 'consistency'] = np.nan
        else:
            raise ValueError('Unrecognized consistency type: %s'%consistency_type)
   
        if weight_by_activation: 
            # weight each trial status (accurate/hit/cr) by quantity of activation in that trial
            def minmax(df_col): 
                return (df_col - df_col.min())/(df_col.max()-df_col.min())
            trials_df['activation_weight'] = minmax(trials_df.activation.abs())
            trials_df.consistency = trials_df.consistency * trials_df.activation_weight

        return trials_df.consistency.mean()

    @classmethod
    def compute_prob_agreement(cls,data_df, trial_id='trial', stim_id= 'stim', feature_id= 'feature', value_id = 'value', response_id='response', **kwargs):
        '''
        Estimates the probability of giving the same response over repeated trials
        by computing the intercept of the probability over pairs of trials ranked by distance
        has the option to compute distance as raw stimulus difference, or difference projected on kernel (provide a kernel_extractor other than None)
        '''

        if 'method' not in kwargs:
            raise TypeError('DistanceMethod missing required argument method')
        method = kwargs['method']

        if method == 'accuracy':
            return cls.compute_accuracy(data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs)
        elif method == 'distance':
            return cls.compute_distance(data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs)
        else:
            raise ValueError('Unrecognized method: %s'%method)