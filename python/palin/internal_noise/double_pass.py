#!/usr/bin/env python
'''
PALIN toolbox v0.1
December 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

import pandas as pd
import numpy as np
from .internal_noise_extractor import InternalNoiseExtractor
from ..simulation.linear_observer import LinearObserver
from ..simulation.simple_experiment import SimpleExperiment
from ..simulation.trial import Int2Trial, Int1Trial 

class DoublePass(InternalNoiseExtractor):

    @classmethod
    def extract_single_internal_noise(cls,data_df, trial_id, stim_id = 'stim', feature_id = 'feature', value_id = 'value', response_id = 'response'):

        double_pass_id = 'double_pass_id' # column by which to identify double pass trials

        # index double pass trials
        data_df = cls.index_double_pass_trials(data_df, trial_id=trial_id, value_id = value_id, double_pass_id = double_pass_id)
        # compute probability of agreement over double pass
        prob_agree = compute_prob_agreement(data_df, trial_id=trial_id, response_id=response_id, double_pass_id=double_pass_id)
        # compute probability of choosing first response option
        prob_first = compute_prob_first(data_df, trial_id=trial_id, response_id=response_id, stim_id=stim_id, double_pass_id=double_pass_id)

        return 0

    def __str__(self): 
        return 'Double-Pass method'

    @classmethod
    def index_double_pass_trials(cls, data_df, trial_id='trial',double_pass_id='double_pass_id',value_id='stim_parameter_id'):
    ''' identify repeated trials in experimental sessions (i.e. 'double pass trials'), and tag them with a unique id stored in a new column.
    '''
        # represent the several values of a given trial (ex. 6 features for interval 1, 6 features for interval 2) as a tuple 
        frozen_set_df = data_df.groupby(trial_id).agg({value_id: lambda group: tuple(group)}).reset_index()

        # count how many trials have each unique pair of stimuli
        pass_count_df = frozen_set_df.groupby(value_id).agg({trial_id: ['nunique','first','last']})
        pass_count_df.columns = ["_".join(x) for x in pass_count_df.columns.ravel()]
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
    # computes the probability of agreement between two responses to a repeated stimuli on the double pass trials 

        # compute agreements for each double_pass trial
        def same_answer(group, trial_id, response_id):    
            d = group.groupby(trial_id).agg({response_id: lambda group: tuple(group)}).reset_index()
            return d.response.nunique()==1
        agrees = data_df.groupby(double_pass_id).apply(lambda group: same_answer(group, trial_id, response_id))
    
        # return agreement probability
        return agrees.sum()/len(agrees)
    
    @classmethod
    def compute_prob_first(cls, data_df, trial_id='trial', response_id='response', stim_id='stim_order', double_pass_id='double_pass_id'):
    # Computes probability that responds true to the first interval across the subset of double_pass trialslumn (e.g. double_pass_id) identifying repeated trials. Use utils.index_double_pass_trials to create that column if doesn't exist. 
    
        # compute first response for each double_pass trial
        def first_option(group, stim_id, response_id):    
            resp = group.sort_values(by=stim_id)[response_id].iloc[0]
            return resp==1
        firsts = data_df[data_df[double_pass_id].notna()].groupby(trial_id).apply(lambda group: first_option(group, stim_id, response_id))
    
        return firsts.sum()/len(firsts)

    @classmethod
    def simulate_observer(cls,internal_noise_std,criteria, n_trials, n_blocks=1): 

        # simulate observer with (criteria, internal_noise_sigma)
        # in the midterm, this should be done with a Simulation object, for which we need a prob_a, prob_first analyser

        obs = LinearObserver(kernel=[1], internal_noise_std=internal_noise_std, criteria=criteria)
        exp = SimpleExperiment(n_trials=n_trials, trial_type=Int2Trial, n_features=1, external_noise_std=1)

        responses_pass_1 = obs.respond_to_experiment(exp)
        responses_pass_2 = obs.respond_to_experiment(exp)
    
        # probability interval 1 (average of prob in both pass)
        prob_first = (np.mean(responses_pass_1) + np.mean(responses_pass_2))/2
    
        #probability of agreement between pass
        prob_agree = np.mean(responses_pass_1==responses_pass_2) 
    
        return prob_agree,prob_first

       