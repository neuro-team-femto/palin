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
    def minmax(cls, df_col): 
        '''
        Utility method to minmaxscale a pd column
        '''
        return (df_col - df_col.min())/(df_col.max()-df_col.min())

    @classmethod
    def compute_accuracy(cls,data_df, trial_id='trial', stim_id= 'stim', feature_id= 'feature', value_id = 'value', response_id='response', **kwargs):
        '''
        Estimates prob agreement using the accuracy method, which computes the extent to which observer responses match those of an ideal observer with the same kernel and no internal noise
        '''
        if 'kernel_extractor' not in kwargs:
            raise TypeError('DistanceMethod missing required argument kernel_extractor')
        kernel_extractor = kwargs['kernel_extractor']

        if 'trial_mask' not in kwargs:
            raise TypeError('DistanceMethod missing required argument trial_mask')
        trial_mask = kwargs['trial_mask']

        if 'weight_trials' not in kwargs: 
            raise TypeError('DistanceMethod missing required argument weight_trials')
        weight_trials = kwargs['weight_trials']

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
        if trial_mask == 'all':
            trials_df['consistency'] = (trials_df.activation > 0) == (trials_df.response==1)
        elif trial_mask == 'hit':
            trials_df['consistency'] = (trials_df.activation > 0) & (trials_df.response==1)
            trials_df.loc[trials_df.activation < 0, 'consistency'] = np.nan
        elif trial_mask== 'cr':
            trials_df['consistency'] = (trials_df.activation < 0) & (trials_df.response==0)
            trials_df.loc[trials_df.activation > 0, 'consistency'] = np.nan
        else:
            raise ValueError('Unrecognized trial_mask: %s'%trial_mask)
   
        if weight_trials: 
            # weight each trial status (accurate/hit/cr) by quantity of activation in that trial
            trials_df['trial_weight'] = cls.minmax(trials_df.activation.abs())
            trials_df.consistency = trials_df.consistency * trials_df.trial_weight

        return trials_df.consistency.mean()

    @classmethod
    def compute_distance(cls,data_df, trial_id='trial', stim_id= 'stim', feature_id= 'feature', value_id = 'value', response_id='response', **kwargs):
        '''
        Estimates prob agreement using the distance method, which checks that chosen stimuli have smaller distance to the chosen average than to the non-chosen average
        '''
        
        if 'trial_mask' not in kwargs:
            raise TypeError('DistanceMethod missing required argument trial_mask')
        trial_mask = kwargs['trial_mask']

        if 'weight_trials' not in kwargs: 
            raise TypeError('DistanceMethod missing required argument weight_trials')
        weight_trials = kwargs['weight_trials']

        # pivot data to have one entry per trials (instead of n_features entries) 
        trials_df = data_df.groupby([trial_id,stim_id]).agg({value_id:list, response_id:'first'}).reset_index()
        trials_df[value_id] = trials_df[value_id].apply(lambda x: np.array(x))  

        # compute chosen (positive) and non-chosen (negative) template
        positive_template = trials_df[trials_df[response_id]==True][value_id].mean()
        negative_template = trials_df[trials_df[response_id]==False][value_id].mean()

        # compute distance to templates
        trials_df['distance_positive'] = trials_df[value_id].apply(lambda x: np.sqrt(np.mean((x - positive_template)**2)))
        trials_df['distance_negative'] = trials_df[value_id].apply(lambda x: np.sqrt(np.mean((x - negative_template)**2)))
        trials_df['expected_chosen'] = trials_df.distance_positive < trials_df.distance_negative
       
        # compute consistency
        trials_df.loc[trials_df[response_id]==True, 'consistency'] = trials_df.expected_chosen
        trials_df.loc[trials_df[response_id]==False, 'consistency'] = ~trials_df.expected_chosen

        if trial_mask == 'all':
            trials_df = trials_df.groupby([trial_id, response_id, 'distance_positive','distance_negative'], as_index=False).consistency.all()
        elif trial_mask == 'hit':
            trials_df = trials_df[trials_df[response_id]==True]
        elif trial_mask== 'cr':
            trials_df = trials_df[trials_df[response_id]==False]
        else:
            raise ValueError('Unrecognized trial_mask: %s'%trial_mask)

        #return trials_df

        if weight_trials: 
            trials_df['distance_ratio'] = trials_df.distance_negative/trials_df.distance_positive
            # weight chosen trials by distance_negative / distance_positive
            trials_df.loc[trials_df[response_id]==True, 'trial_weight'] = trials_df.loc[trials_df[response_id]==True, 'distance_ratio']
            # weight non-chosen trials by distance_positive / distance_negative
            trials_df.loc[trials_df[response_id]==False, 'trial_weight'] = 1 / trials_df.loc[trials_df[response_id]==False, 'distance_ratio']
            # normalize weights to 0-1
            trials_df.trial_weight = cls.minmax(trials_df.trial_weight)
            # or normalize weights to unit sum
            #trials_df.trial_weight = trials_df.trial_weight / trials_df.trial_weight.sum()
            # weight consistency
            trials_df.consistency = trials_df.consistency * trials_df.trial_weight

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