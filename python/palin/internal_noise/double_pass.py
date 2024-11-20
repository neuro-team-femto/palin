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


class DoublePass(AgreementMethod):
    ''' 
    This class implements an AgreementMethod to estimate internal noise and criteria from reverse correlation data. 
    This requires that the data has double pass trials, from which it computes prob_agree and prob_first from double-pass data. 
    '''

    def __str__(self): 
        return 'Double-Pass method'

    @classmethod
    def compute_probabilities(cls,data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs):
        '''
        Compute probabilities over double pass trials
        '''
        double_pass_id = 'double_pass_id' # column by which to identify double pass trials
        # index double pass trials
        double_pass_df = cls.index_double_pass_trials(data_df, trial_id=trial_id, value_id = value_id, double_pass_id = double_pass_id)
        double_pass_df = double_pass_df[double_pass_df[double_pass_id].notna()]

        # compute probability of agreement over double pass
        prob_agree = cls.compute_prob_agreement(double_pass_df, trial_id=trial_id, response_id=response_id, double_pass_id='double_pass_id')
        # compute probability of choosing first response option
        prob_first = cls.compute_prob_first(double_pass_df, trial_id=trial_id, response_id=response_id, stim_id=stim_id, **kwargs)
        return prob_agree, prob_first

    @classmethod
    def compute_prob_agreement(cls,data_df, trial_id='trial', response_id='response', double_pass_id='double_pass_id'):
        '''
        Computes the probability of giving the same response over all pairs of repeated trials (as identified by double_pass_id, see index_double_pass_trials)
        ''' 

        # compute agreements for each double_pass trial
        def same_answer(group, trial_id, response_id):    
            d = group.groupby(trial_id).agg({response_id: lambda group: tuple(group)}).reset_index()
            return d.response.nunique()==1
        agrees = data_df.groupby(double_pass_id).apply(lambda group: same_answer(group, trial_id, response_id))
    
        # return agreement probability
        return agrees.sum()/len(agrees)
    
    