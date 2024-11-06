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
    def compute_probabilities(cls,data_df, trial_id, stim_id, feature_id, value_id, response_id):
        '''
        Compute probabilities over double pass trials
        '''
        double_pass_id = 'double_pass_id' # column by which to identify double pass trials
        # index double pass trials
        data_df = cls.index_double_pass_trials(data_df, trial_id=trial_id, value_id = value_id, double_pass_id = double_pass_id)
        # compute probability of agreement over double pass
        prob_agree = cls.compute_prob_agreement(data_df, trial_id=trial_id, response_id=response_id, double_pass_id=double_pass_id)
        # compute probability of choosing first response option
        prob_first = cls.compute_prob_first(data_df, trial_id=trial_id, response_id=response_id, stim_id=stim_id, double_pass_id=double_pass_id)
        return prob_agree, prob_first

    @classmethod
    def index_double_pass_trials(cls, data_df, trial_id='trial',double_pass_id='double_pass_id',value_id='value'):
        '''
        Runs over data by a single user, identifies any repeated trials based on the set of their values, and tags them with a column called double_pass_id.
        At the end of the procedure, data-df[double_pass_id].max() is the total number of repeated trials found in the data.  
        '''
        # represent the several values of a given trial (ex. 6 features for interval 1, 6 features for interval 2) as a tuple 
        set_df = data_df.groupby(trial_id).agg({value_id: lambda group: tuple(group)}).reset_index()

        # count how many trials have each unique pair of stimuli
        pass_count_df = set_df.groupby(value_id).agg({trial_id: ['nunique','first','last']})
        pass_count_df.columns = ["_".join(x) for x in pass_count_df.columns]
        pass_count_df = pass_count_df.reset_index()

        # identify pairs of stimuli that have 2 trials (i.e. for which there has been a double pass)
        double_pass_df = pass_count_df[pass_count_df['%s_nunique'%trial_id]==2].reset_index(drop=True)
        
        # assign unique id
        double_pass_df[double_pass_id] = double_pass_df.index

        # join to base dataset
        double_pass_df = double_pass_df.melt(id_vars=double_pass_id, 
                                             value_vars=['%s_first'%trial_id,'%s_last'%trial_id], 
                                             var_name='%s_type'%trial_id, 
                                             value_name=trial_id)
        data_df= pd.merge(data_df, double_pass_df[[trial_id, double_pass_id]], 
                          how="left", on=trial_id)
        return data_df  


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
    
    @classmethod
    def compute_prob_first(cls, data_df, trial_id='trial', response_id='response', stim_id='stim_order', double_pass_id='double_pass_id'):
        '''
        Computes probability to choose the first response option (i.e. a measure of response bias) across the subset of double_pass trials 
        ''' 
    
        # compute first response for each double_pass trial
        def first_option(group, stim_id, response_id):    
            resp = group.sort_values(by=stim_id)[response_id].iloc[0]
            return resp==1
        firsts = data_df[data_df[double_pass_id].notna()].groupby(trial_id).apply(lambda group: first_option(group, stim_id, response_id))
    
        return firsts.sum()/len(firsts)