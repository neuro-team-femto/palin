#!/usr/bin/env python
'''
PALIN toolbox v0.1
sep 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Class to simulate participant in revcor experiment simulations
'''

import numpy as np
import pandas as pd

class Participant:
    def __init__(self, kernel,internal_noise_std,criteria):
        self.kernel = kernel
        self.criteria = criteria
        self.internal_noise_std = internal_noise_std

    @classmethod
    def with_random_kernel(cls, n_features, internal_noise_std,criteria):
        return cls(np.random.uniform(-1,1,n_features),internal_noise_std,criteria)

    def respond_to_stim(self, stim):
        return np.dot(stim, self.kernel)
    
    def generate_internal_noise(self, external_noise_std): 
        return np.random.normal(loc=0, scale=self.internal_noise_std)*external_noise_std
        
    def respond_to_trial(self, trial, external_noise_std): 

        activity = self.respond_to_stim(trial.stims[0])
        if trial.type == '2afc': 
            activity -= self.respond_to_stim(trial.stims[1])
        internal_noise = self.generate_internal_noise(external_noise_std)
        response = 0 if (activity + internal_noise >= (self.criteria*external_noise_std)) else 1
        return response

    def respond_to_experiment(self,experiment):         
        responses = []
        for trial in experiment.trials: 
            responses.append(self.respond_to_trial(trial, experiment.external_noise_std))
        return responses

class Trial: 
    TYPES = ['1afc','2afc']
    def __init__(self,stims):
        self.stims = stims
        self.type = Trial.TYPES[len(stims)-1]

class Experiment: 
    def __init__(self, n_trials, exp_type, n_features, external_noise_std): 
        self.n_trials = n_trials
        self.type = exp_type
        self.n_features = n_features
        self.external_noise_std = external_noise_std   

    def create_stim_noise(self):
        stim_noise = np.random.normal(loc=0, scale=self.external_noise_std, size=self.n_features)
        return range(self.n_features),stim_noise

    def generate_trials(self):
        self.trials = []
        for trial_number in range(self.n_trials):
            stims = []
            num_stim = (self.type == '2afc') + 1
            for stim_order in range(num_stim):
                stim_feature, stim_values = self.create_stim_noise()
                stims.append(list(stim_values))
            self.trials.append(Trial(stims))


class Analyzer: 

    
    def set_kernel_method(self,kernel_method):
        self.kernel_method = kernel_method

    #self.noise_method = noise_method

    @classmethod
    def to_df(csl, experiment, responses): 

        trial_ids = []
        stim_orders = []
        features = []
        values = []
        resps  = []
    
        for num_trial, trial in enumerate(experiment.trials):
            response = responses[num_trial]
            for num_stim, stim in enumerate(trial.stims): 
                for num_feature, value in enumerate(stim):
                    trial_ids.append(num_trial)
                    stim_orders.append(num_stim)
                    features.append(num_feature)
                    values.append(value)
                    resps.append(True if response == num_stim else False)
    
        return pd.DataFrame.from_dict({'trial_id': trial_ids,
                                   'stim_order': stim_orders,
                                   'feature': features,
                                   'value': values, 
                                   'response': resps})

    def estimate_kernel(self, experiment, responses): 

        if self.kernel_method == None: 
            print('no kernel method',  file=sys.stderr)

        responses_df = Analyzer.to_df(experiment, responses)

        kernels_df = self.kernel_method(responses_df,
            trial_ids=[],
            dimension_ids=['feature'],
            response_id='response', 
            value_id='value',
            normalize=True)

        return list(kernels_df.kernel_value)








        #response_df = trials_df.groupby('trial_id').apply(
            #    lambda group: self.respond_to_trial(new Trial(type='2afc',
            #                                                  stims=[group[group['stim_order'] == 0]['value'].values,
            #                                                         group[group['stim_order'] == 1]['value'].values]),
            #                                        experiment.external_noise_std))
            # call this "response"
            #response_df = response_df.reset_index().rename({0:'response'}, axis=1)
            # merge with trial_data
            #response_df = response_df.merge(trials_df, on='trial_id')
            # and convert to True/false for the two stimuli of each trial
            #response_df['response'] = response_df.apply(lambda row: True if row['response'] == row['stim_order'] else False, axis=1)
        