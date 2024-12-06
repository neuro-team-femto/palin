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

class InterceptMethod(AgreementMethod):
    ''' 
    This class implements an AgreementMethod to estimate internal noise and criteria from reverse correlation data. 
    Contrary to the DoublePass method, it does not require that the data has double pass trials. 
    Prob_agree is estimated by the intercept of a polynomial fitted to the probability of agreement of all pairs of trials, as a function of trial distance
    Prob_first is estimated on the complete set of trials.
    '''

    def __str__(self): 
        return 'Intercept method'

    @classmethod
    def compute_probabilities(cls,data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs):
        '''
        Compute probabilities over non-double pass trials
        '''

        if 'kernel_extractor' not in kwargs:
            raise TypeError('InterceptMethod missing required argument kernel_extractor')

        # Keep only non double_pass trials, or the first occurrence of double_pass trials
        double_pass_id = 'double_pass_id' # column by which to identify double pass trials
        data_df = cls.index_double_pass_trials(data_df, trial_id=trial_id, value_id = value_id, double_pass_id = double_pass_id)
        single_pass_df = cls.keep_single_pass(data_df, trial_id=trial_id, double_pass_id = double_pass_id)
        
        # compute probability of agreement
        prob_agree = cls.compute_prob_agreement(single_pass_df, trial_id=trial_id, stim_id=stim_id,
            response_id=response_id, feature_id= feature_id, value_id = value_id, kernel_extractor=kwargs['kernel_extractor'])
        # compute probability of choosing first response option
        prob_first = cls.compute_prob_first(single_pass_df, trial_id=trial_id, response_id=response_id, stim_id=stim_id)
        return prob_agree, prob_first

    @classmethod
    def compute_prob_agreement(cls,data_df, trial_id='trial', stim_id= 'stim', feature_id= 'feature', value_id = 'value', response_id='response', kernel_extractor=None):
        '''
        Estimates the probability of giving the same response over repeated trials
        by computing the intercept of the probability over pairs of trials ranked by distance
        has the option to compute distance as raw stimulus difference, or difference projected on kernel (provide a kernel_extractor other than None)
        '''
        
        # pivot data to have one entry per trials (instead of n_features entries) 
        trials_df = data_df.groupby([trial_id,stim_id]).agg({value_id:list, response_id:'first'}).reset_index()
        trials_df[value_id] = trials_df[value_id].apply(lambda x: np.array(x))
        
        # for each trial, compute value difference and chosen stim (0 or 1)
        # WARNING: this won't work for 1AFC data
        trials_df = trials_df.groupby([trial_id]).agg({value_id:lambda x:x.diff().iloc[1], #diff produces 2 lines, the first is nan
                           response_id: lambda x: 0 if x.iloc[0] else 1}).reset_index()

        # generate all combinations of trials 
        # to do: optimize compute time - the bottleneck is not to generate combinations (merge faster, but bigger df), 
        # rather the subsequent operations on the large df        
        a, b = map(list, zip(*combinations(trials_df.index, 2)))
        combinations_df = pd.concat([trials_df.loc[a].add_suffix('_1').reset_index(), 
            trials_df.loc[b].add_suffix('_2').reset_index()], axis=1).drop(columns=['index'])
        #combinations_df = trials_df.merge(trials_df, how='cross', suffixes=('_1','_2'))
        #combinations_df = combinations_df[combinations_df.trial_1 != combinations_df.trial_2]

        # computed difference between each trial pair (as an option: projected on the kernel)
        combinations_df[value_id]=(combinations_df[value_id+'_1'] - combinations_df[value_id+'_2'])
        if kernel_extractor: 
            kernel = kernel_extractor.extract_single_kernel(data_df, trial_id,stim_id, feature_id, value_id, response_id)
            # projected difference on the kernel (if 2 trials are different in dimensions for which the kernel is null, then that difference doesn't matter)
            combinations_df[value_id]= combinations_df[value_id].apply(lambda x: np.abs(x.dot(list(kernel.kernel_value))))
        else: 
            # RMS of trial difference (i.e. trial difference of stim difference)
            combinations_df[value_id] = combinations_df[value_id].apply(lambda x: np.sqrt(np.sum(x**2)))
        combinations_df['agree']=(combinations_df[response_id+'_1']==combinations_df[response_id+'_2']).astype(int) 
        #comb_df = comb_df.drop(columns=['value_1','value_2','response_1','response_2'])

        # bin combinations
        min_non_null = combinations_df[combinations_df[value_id]>0][value_id].min() # the minimum non null distance (to exclude double pass trials from the estimate)
        nbins = 50  # rule of thumb
        bins = pd.cut(combinations_df[value_id],
            bins=np.linspace(min_non_null,
                combinations_df[value_id].max()+1,
                nbins),
            labels=False)
        bins = bins + 1 # increment all bins by 1
        bins = bins.fillna(0) # and give bin 0 to all that are < min_non_null 
        combinations_df = combinations_df.groupby(bins).agree.mean().reset_index()

        # return intercept of polynomial fit
        try:
            poly = np.poly1d(np.polyfit(combinations_df[value_id][1:], combinations_df.agree[1:], 3))
            return poly(0) # return intercept of polynomial fit 
        except RuntimeError: 
            print('error fitting polynomial')
            return np.nan
