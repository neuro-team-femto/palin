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
from itertools import combinations

class InterceptMethod(InternalNoiseExtractor):
    ''' 
    This class provides methods to estimate internal noise and criteria from reverse correlation data. 
    Contrary to the DoublePass Extractor, it does not require that the data has double pass trials. 
    Internal noise and criteria are computed using the Ponsot/Neri double-pass simulation method, by first computing prob_agree and prob_first.
    Prob_agree (normally: across exactly repeated trials) in estimated in the absence of double pass trials by the intercept of a polynomial fitted to the probability of agreement of all pairs of trials, as a function of trial distance
    Prob_first is estimated on the complete set of trials. 
    Internal noise and criteria are then estimated by simulating an ideal observer with a range of internal noise and criteria values to find what duplet of values generates prob_agree and prob_first that are closest to those observed in the actual data. 
    '''

    @classmethod
    def extract_single_internal_noise(cls,data_df, trial_id, stim_id, feature_id, value_id, response_id, model_file, rebuild_model=False, internal_noise_range=np.arange(0,5,.1),criteria_range=np.arange(-5,5,1), n_repeated_trials=100, n_runs=10):
        '''
        Extracts internal noise and criteria for a single observer/session. 
        To extract for several users/sessions, use the superclass's method extract_internal_noise
        '''
        # compute probability of agreement, using the intercept method over all trials (regardless of double pass)
        prob_agree = cls.compute_prob_agreement(data_df, trial_id=trial_id, stim_id= stim_id, value_id= value_id, response_id=response_id)
        # compute probability of choosing first response option over all trials (regardless of double pass)
        prob_first = cls.compute_prob_first(data_df, trial_id=trial_id, response_id=response_id, stim_id=stim_id)

        internal_noise, criteria = cls.estimate_noise_criteria(prob_agree, prob_first, model_file, rebuild_model, internal_noise_range,criteria_range, n_repeated_trials, n_runs)

        return internal_noise,criteria

    def __str__(self): 
        return 'Intercept method'

    @classmethod
    def compute_prob_agreement(cls,data_df, trial_id='trial', stim_id= 'stim', feature_id= 'segment', value_id = 'value', response_id='response', kernel_extractor=None):
        '''
        Estimates the probability of giving the same response over repeated trials
        by computing the intercept of the probability over pairs of trials ranked by distance
        '''
        
        # pivot data to have one entry per trials (instead of n_features entries) 
        trials_df = data_df.groupby([trial_id,stim_id]).agg({value_id:list, response_id:'first'}).reset_index()
        trials_df[value_id] = trials_df[value_id].apply(lambda x: np.array(x))
        
        # for each trial, compute value difference and chosen stim (0 or 1)
        # WARNING: this won't work for 1AFC data
        trials_df = trials_df.groupby([trial_id]).agg({value_id:lambda x:x.diff().iloc[1], #diff produces 2 lines, the first is nan
                           response_id: lambda x: 0 if x.iloc[0] else 1}).reset_index()

        # generate all combinations of trials 
        # to do: optimize compute time
        a, b = map(list, zip(*combinations(trials_df.index, 2)))
        combinations_df = pd.concat([trials_df.loc[a].add_suffix('_1').reset_index(), 
            trials_df.loc[b].add_suffix('_2').reset_index()], axis=1).drop(columns=['index'])

        # computed difference between each trial pair (as an option: projected on the kernel)
        combinations_df[value_id]=(combinations_df[value_id+'_1'] - combinations_df[value_id+'_2'])
        if kernel_extractor: 
            kernel = kernel_extractor.extract_single_kernel(data_df, feature_id, value_id, response_id)
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
        combinations_df.groupby(bins).agree.mean().reset_index()
   
        # return intercept of polynomial fit
        try:
            poly = np.poly1d(np.polyfit(combinations_df[value_id][1:], combinations_df.agree[1:], 3))
            return poly(0) # return intercept of polynomial fit 
        except RuntimeError: 
            print('error fitting polynomial')
            return np.nan
    
    @classmethod
    def compute_prob_first(cls, data_df, trial_id='trial', response_id='response', stim_id='stim_order'):
        '''
        Computes probability to choose the first response option (i.e. a measure of response bias) across all trials
        ''' 
    
        # compute first response for each double_pass trial
        def first_option(group, stim_id, response_id):    
            resp = group.sort_values(by=stim_id)[response_id].iloc[0]
            return resp==1
        firsts = data_df.groupby(trial_id).apply(lambda group: first_option(group, stim_id, response_id))
    
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

        # deferred imports of the simulation modules, to avoid circular imports
        from ..simulation.observers.linear_observer import LinearObserver
        from ..simulation.trial import Int2Trial 
        from ..simulation.experiments.double_pass_experiment import DoublePassExperiment
        from ..simulation.trial import Int2Trial, Int1Trial 
        from ..simulation.analysers import DouplePassStatistics
        from ..simulation.simulation import Simulation as Sim

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
              DoublePassStatistics, analyser_params)

        sim_df = sim.run_all(n_runs=n_runs, verbose=True)

        # average measures over all runs
        sim_df.groupby(['internal_noise_std','criteria'])[DoublePassStatistics.get_metric_names()].mean()
        return sim_df

       