
from .analyser import Analyser
import numpy as np

class InternalNoiseValue(Analyser): 

    def __init__(self, internal_noise_extractor, model_file, rebuild_model = False, internal_noise_range=np.arange(0,5,.1),criteria_range=np.arange(-5,5,1), n_repeated_trials=100, n_runs=10):
        self.internal_noise_extractor = internal_noise_extractor
        self.model_file = model_file
        self.rebuild_model = rebuild_model
        self.internal_noise_range = internal_noise_range
        self.criteria_range = criteria_range
        self.n_repeated_trials = n_repeated_trials
        self.n_runs = n_runs
        
    def analyse(self, experiment, participant, participant_responses): 

        #true_internal_noise = participant.internal_noise_std

        return self.estimate_internal_noise(experiment, participant_responses)[0]

    def estimate_internal_noise(self, experiment, participant_responses): 

        responses_df = self.to_df(experiment, participant_responses)

        internal_noise = self.internal_noise_extractor.extract_single_internal_noise(data_df = responses_df,
         trial_id = 'trial', stim_id = 'stim', feature_id = 'feature', value_id = 'value', response_id = 'response', model_file = self.model_file,
         internal_noise_range=self.internal_noise_range, criteria_range=self.criteria_range, n_repeated_trials=self.n_repeated_trials, n_runs=self.n_runs)

        return internal_noise
        
        