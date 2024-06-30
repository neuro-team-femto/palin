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
from ..simulation.observers.linear_observer import LinearObserver
from ..simulation.experiments.simple_experiment import SimpleExperiment
from ..simulation.trial import Int2Trial, Int1Trial 
from ..simulation.experiments.double_pass_experiment import DoublePassExperiment
from ..simulation.trial import Int2Trial, Int1Trial 
from ..simulation.analysers import double_pass_statistics as dps #import module instead of class to avoid circular import
from ..simulation.simulation import Simulation as Sim


class DoublePass(InternalNoiseExtractor):
    ''' 
    This class provides methods to estimate internal noise and criteria from reverse correlation data, provided the data has double pass trials. 
    Internal noise and criteria are computed using the Ponsot/Neri double-pass simulation method, by first computing prob_agree and prob_first from double-pass data, 
    and then simulating an ideal observer with a range of internal noise and criteria values to find what duplet of values generates prob_agree and prob_first that are closest to those observed in the actual data. 
    '''

    @classmethod
    def extract_single_internal_noise(cls,data_df, trial_id, stim_id, feature_id, value_id, response_id, model_file, rebuild_model=False, internal_noise_range=np.arange(0,5,.1),criteria_range=np.arange(-5,5,1), n_repeated_trials=100, n_runs=10):
        '''
        Extracts internal noise and criteria for a single observer/session. 
        To extract for several users/sessions, use the superclass's method extract_internal_noise
        '''
        double_pass_id = 'double_pass_id' # column by which to identify double pass trials
        # index double pass trials
        data_df = cls.index_double_pass_trials(data_df, trial_id=trial_id, value_id = value_id, double_pass_id = double_pass_id)
        # compute probability of agreement over double pass
        prob_agree = cls.compute_prob_agreement(data_df, trial_id=trial_id, response_id=response_id, double_pass_id=double_pass_id)
        # compute probability of choosing first response option
        prob_first = cls.compute_prob_first(data_df, trial_id=trial_id, response_id=response_id, stim_id=stim_id, double_pass_id=double_pass_id)

        internal_noise, criteria = cls.estimate_noise_criteria(prob_agree, prob_first, model_file, rebuild_model, internal_noise_range,criteria_range, n_repeated_trials, n_runs)

        return internal_noise,criteria

    def __str__(self): 
        return 'Double-Pass method'

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

    @classmethod
    def estimate_noise_criteria(cls,prob_agree, prob_first, model_file,rebuild_model=False, internal_noise_range=np.arange(0,5,.1),criteria_range=np.arange(-5,5,1), n_repeated_trials=100, n_runs=10): 
        '''
        Estimates internal noise and criteria given a measure of prob_agree and prob_first. 
        Either uses a prebuilt model (a dataframe previously generated by @build_model and stored as a .csv file), or rebuild a new model. 
        Searches through a range of possible internal noise and criteria values
        '''    
        # load model or rebuild
        if os.path.isfile(model_file) & ~rebuild_model: 
            model_df = pd.read_csv(model_file, index_col=0)
        else:
            model_df = cls.build_model(internal_noise_range, criteria_range, n_repeated_trials, n_runs)
            model_df.to_csv(model_file)

        # find internal_noise & criteria settings that minimizes distance to prob_agree and prob_first 
        model_df['dist'] = model_df.apply(lambda row: (row.prob_agree-prob_agree)**2 + (row.prob_first-prob_first)**2, axis=1)

        best_match = model_df[model_df.dist==model_df.dist.min()]

        return best_match.internal_noise_std.iloc[0], best_match.criteria.iloc[0]

    @classmethod
    def build_model(cls,internal_noise_range=np.arange(0,5,.1),criteria_range=np.arange(-5,5,1), n_repeated_trials=100, n_runs=10): 
        '''
        Build a model that associates a range of internal noise and criteria values with their corresponding (simulated) prob_agree and prob_first.
        This uses a simulated LinearObserver, and returns the model as a dataframe 
        '''
        print('Rebuilding double-pass model')

        observer_params = {'kernel':[[1]],
                   'internal_noise_std':internal_noise_range, 
                  'criteria':criteria_range}
        experiment_params = {'n_trials':[n_repeated_trials],
                     'n_repeated':[n_repeated_trials],
                     'trial_type': [Int2Trial],
                     'n_features': [1],
                     'external_noise_std': [1]}
        analyser_params = {}

        sim = Sim(DoublePassExperiment, experiment_params,
              LinearObserver, observer_params, 
              dps.DoublePassStatistics, analyser_params)

        sim_df = sim.run_all(n_runs=n_runs, verbose=True)

        # average measures over all runs
        sim_df.groupby(['internal_noise_std','criteria'])[dps.DoublePassStatistics.get_metric_names()].mean()
        return sim_df

       