
from ..analyser import Analyser
import numpy as np

class InternalNoiseValue(Analyser): 

    def __init__(self, internal_noise_extractor, **kwargs):
        self.internal_noise_extractor = internal_noise_extractor
        self.kwargs = kwargs

    @classmethod
    def get_metric_names(self):
        return ['estimated_internal_noise','estimated_criteria']
        
    def analyse(self, experiment, participant, participant_responses): 

        return self.estimate_internal_noise(experiment, participant_responses)

    def estimate_internal_noise(self, experiment, participant_responses): 

        responses_df = self.to_df(experiment, participant_responses)

        internal_noise, criteria = self.internal_noise_extractor.extract_single_internal_noise(data_df = responses_df,
         trial_id = 'trial', stim_id = 'stim', feature_id = 'feature', value_id = 'value', response_id = 'response', **self.kwargs)

        return internal_noise, criteria
        
        