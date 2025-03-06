#!/usr/bin/env python
'''
PALIN toolbox v0.1
December 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

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

        if 'kernel_extractor' not in kwargs:
            raise TypeError('DistanceMethod missing required argument kernel_extractor')

        if 'consistency_type' not in kwargs:
            raise TypeError('DistanceMethod missing required argument consistency_type')

        # Keep only non double_pass trials, or the first occurrence of double_pass trials
        double_pass_id = 'double_pass_id' # column by which to identify double pass trials
        data_df = cls.index_double_pass_trials(data_df, trial_id=trial_id, value_id = value_id, double_pass_id = double_pass_id)
        single_pass_df = cls.keep_single_pass(data_df, trial_id=trial_id, double_pass_id = double_pass_id)
        
        # compute probability of agreement
        prob_agree = cls.compute_prob_agreement(single_pass_df, trial_id=trial_id, 
            response_id=response_id, feature_id= 'feature', value_id = 'value', kernel_extractor=kwargs['kernel_extractor'], consistency_type = kwargs['consistency_type'])
        # compute probability of choosing first response option
        prob_first = cls.compute_prob_first(single_pass_df, trial_id=trial_id, response_id=response_id, stim_id=stim_id)
        return prob_agree, prob_first

    @classmethod
    def compute_prob_agreement(cls,data_df, trial_id='trial', stim_id= 'stim', feature_id= 'feature', value_id = 'value', response_id='response', kernel_extractor=None, consistency_type=None):
        '''
        Estimates the probability of giving the same response over repeated trials
        by computing the intercept of the probability over pairs of trials ranked by distance
        has the option to compute distance as raw stimulus difference, or difference projected on kernel (provide a kernel_extractor other than None)
        '''

        # pivot data to have one entry per trials (instead of n_features entries) 
        trials_df = data_df.groupby([trial_id,stim_id]).agg({value_id:list, response_id:'first'}).reset_index()
        trials_df[value_id] = trials_df[value_id].apply(lambda x: np.array(x))  

        # compute stimulus difference in each trial
        trials_df = trials_df.groupby([trial_id]).agg({value_id:lambda x:x.diff().iloc[1], #diff produces 2 lines, the first is nan
                           response_id: lambda x: 0 if x.iloc[0] else 1}).reset_index()

        # compute scalar product with kernel
        kernel = kernel_extractor.extract_single_kernel(data_df, trial_id,stim_id, feature_id, value_id, response_id)
        trials_df['activation']= trials_df[value_id].apply(lambda x: x.dot(list(kernel.kernel_value)))   

        # accurate if response consistent with kernel product
        trials_df['accurate'] = (trials_df.activation > 0) == (trials_df.response==1)
        trials_df['hit'] = (trials_df.activation > 0) & (trials_df.response==1)
        trials_df.loc[trials_df.activation < 0, 'hit'] = np.nan
        trials_df['cr'] = (trials_df.activation < 0) & (trials_df.response==0)
        trials_df.loc[trials_df.activation > 0, 'cr'] = np.nan

        # weight by quantity of activatio
        trials_df['accurate_weight'] = trials_df.activation.abs()
        trials_df.accurate_weight = trials_df.accurate_weight/trials_df.accurate_weight.max()

        trials_df['hit_weight'] = (trials_df.activation - trials_df.activation.min())/(trials_df.activation.max()-trials_df.activation.min())
        trials_df['cr_weight'] = 1-(trials_df.activation - trials_df.activation.min())/(trials_df.activation.max()-trials_df.activation.min())

        trials_df['weighted_accurate'] = trials_df.accurate * trials_df.accurate_weight
        trials_df['weighted_hit'] = trials_df.hit * trials_df.hit_weight
        trials_df['weighted_cr'] = trials_df.cr * trials_df.cr_weight

        
        trials_df[['accurate','hit','cr','weighted_accurate',"weighted_hit","weighted_cr"]].mean()


        if consistency_type == 'accuracy':
            return trials_df.accurate.mean()
        elif consistency_type == 'weighted_accuracy':
            return trials_df.weighted_accurate.mean()
        elif consistency_type == 'hit':
            return trials_df.hit.mean()
        elif consistency_type == 'weighted_hit':
            return trials_df.weighted_hit.mean()
        elif consistency_type== 'cr':
            return trials_df.cr.mean()
        elif consistency_type == 'weighted_cr':
            return trials_df.weighted_cr.mean()
        else:
            raise ValueError('Unrecognized consistency type: %s'%consistency_type)