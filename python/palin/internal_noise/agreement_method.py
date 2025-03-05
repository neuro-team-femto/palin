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
from .internal_noise_extractor import InternalNoiseExtractor
from abc import ABC,abstractmethod

class AgreementMethod(InternalNoiseExtractor):
    ''' 
    This abstract class subserves all InternalNoiseExtractors that rely on simulating ideal observers with a range of internal noise and criteria values to match an observed measure of probability of agreement
    The class implements the simulation method, but does not implement methods to compute prob_agreement and prob_first. 
    Options to do that are in child classes DoublePass (which relies on the availability of exactly repeated trials, aka 'Ponsot/Neri' method), InterceptMethod (which estimates agreement based on closely resembling trials) and others
    '''

    def __str__(self): 
        return 'Agreement method'

    @classmethod
    def extract_single_internal_noise(cls,data_df, trial_id = 'trial', stim_id = 'stim', feature_id = 'feature', value_id = 'value', response_id = 'response', **kwargs):
        '''
        Extracts internal noise and criteria for a single observer/session. 
        To extract for several users/sessions, use the superclass's method extract_internal_noise
        '''

        if 'agreement_model_file' not in kwargs:
            raise ValueError('no model file provided for AgreementMethod') 

        prob_agree, prob_first = cls.compute_probabilities(data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs)

        internal_noise, criteria = cls.estimate_noise_criteria(prob_agree, prob_first, agreement_model_file=kwargs['agreement_model_file'])

        return internal_noise

    @classmethod
    def extract_single_criteria(cls,data_df, trial_id = 'trial', stim_id = 'stim', feature_id = 'feature', value_id = 'value', response_id = 'response', **kwargs):
        '''
        Extracts internal noise and criteria for a single observer/session. 
        To extract for several users/sessions, use the superclass's method extract_internal_noise
        '''

        if 'agreement_model_file' not in kwargs:
            raise ValueError('no model file provided for AgreementMethod') 

        prob_agree, prob_first = cls.compute_probabilities(data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs)

        internal_noise, criteria = cls.estimate_noise_criteria(prob_agree, prob_first, kwargs['agreement_model_file'])

        return criteria

    @classmethod
    @abstractmethod
    def compute_probabilities(cls,data_df, trial_id='trial', stim_id='stim', feature_id='feature', value_id='value', response_id='reponse', **kwargs):
        raise NotImplementedError()
        
    @classmethod
    @abstractmethod
    def compute_prob_agreement(cls,data_df, trial_id='trial', stim_id= 'stim', feature_id= 'feature', value_id = 'value', response_id='response', **kwargs):
        raise NotImplementedError()

    @classmethod
    def compute_prob_first(cls, data_df, trial_id='trial', response_id='response', stim_id='stim', **kwargs):
        '''
        Computes probability to choose the first response option (i.e. a measure of response bias) across the subset of trials 
        ''' 
        # compute first response for each trial
        def first_option(group, stim_id, response_id):    
            resp = group.sort_values(by=stim_id)[response_id].iloc[0]
            return resp==1
        firsts = data_df.groupby(trial_id).apply(lambda group: first_option(group, stim_id, response_id))
    
        return firsts.sum()/len(firsts)

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
    def keep_single_pass(cls, data_df, trial_id='trial', double_pass_id = 'double_pass_id'): 
        '''
        Utility to keep only non double_pass trials, or the first occurrence of double_pass trials, in a df already indexed by index_double_pass_trials
        '''
        if double_pass_id not in data_df: 
            return data_df
        first_pass_trials = data_df.groupby([double_pass_id])[trial_id].unique().reset_index()
        first_pass_trials[trial_id] = first_pass_trials[trial_id].apply(lambda x: np.min(x)) # first occurrence of all non Nan double_pass
        single_pass_df = data_df[(data_df[double_pass_id].isna())
                  | (data_df[trial_id].isin(first_pass_trials[trial_id].to_list()))]
        return single_pass_df
    
    @classmethod
    def estimate_noise_criteria(cls,prob_agree, prob_first, agreement_model_file=None):
        '''
        Estimates internal noise and criteria given a measure of prob_agree and prob_first. 
        Uses a model (a dataframe previously generated by @build_model and stored as a .csv file), previously built with build_model(). 
        Searches through a range of possible internal noise and criteria values
        '''    
        # load model or rebuild
        if (not agreement_model_file) or (not os.path.isfile(agreement_model_file)): 
            raise ValueError('no model file provided for AgreementMethod: use AgreementMethod.build_model() first') 
        else: 
            model_df = pd.read_csv(agreement_model_file)
                
        # find internal_noise & criteria settings that minimizes distance to prob_agree and prob_first 
        model_df['dist'] = model_df.apply(lambda row: (row.prob_agree-prob_agree)**2 + (row.prob_first-prob_first)**2, axis=1)

        # todo: consider putting a prior on small internal_noise and small criteria
        best_match = model_df[model_df.dist==model_df.dist.min()]

        return best_match.internal_noise_std.iloc[0], best_match.criteria.iloc[0]

    @classmethod
    def build_model(cls,agreement_model_file = './double_pass_model.csv', internal_noise_range=np.arange(0,5,.1),criteria_range=np.arange(-5,5,1), n_repeated_trials=100, n_runs=10): 
        '''
        Build a model that associates a range of internal noise and criteria values with their corresponding (simulated) prob_agree and prob_first.
        This uses a simulated LinearObserver, and returns the model as a dataframe 
        '''
        print('Building double-pass model')

        # deferred imports of the simulation modules, to avoid circular imports
        from ..simulation.observers.linear_observer import LinearObserver
        from ..simulation.trial import Int2Trial 
        from ..simulation.experiments.double_pass_experiment import DoublePassExperiment
        from ..simulation.trial import Int2Trial, Int1Trial 
        from ..simulation.analysers.agreement_statistics import AgreementStatistics
        from ..simulation.simulation import Simulation as Sim
        from .double_pass import DoublePass
        

        observer_params = {'kernel':[[1]],
                   'internal_noise_std':internal_noise_range, 
                  'criteria':criteria_range}
        experiment_params = {'n_trials':[n_repeated_trials],
                     'n_repeated':[n_repeated_trials],
                     'trial_type': [Int2Trial],
                     'n_features': [1],
                     'external_noise_std': [1]}
        analyser_params = {'internal_noise_extractor':[DoublePass]}

        ## TODO CHECK HOW to use AgreementStatistics instead of DoublePassStatistics here

        sim = Sim(DoublePassExperiment, experiment_params,
              LinearObserver, observer_params, 
              AgreementStatistics, analyser_params)

        sim_df = sim.run_all(n_runs=n_runs)
        
        # average measures over all runs
        sim_df.groupby(['internal_noise_std','criteria'])[AgreementStatistics.get_metric_names()].mean()
        
        sim_df.to_csv(agreement_model_file)

        return sim_df

       