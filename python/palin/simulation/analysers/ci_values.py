
from ..analyser import Analyser
from ...internal_noise.glm_method import GLMMethod
import numpy as np

class CIValues(Analyser): 

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    @classmethod
    def get_metric_names(self):
        return ['confidence_interval', 'confidence_interval_unscaled']
        
    def analyse(self, experiment, participant, participant_responses): 

        data_df =  self.to_df(experiment, participant_responses)

        scaled_agg_norm_ci, agg_norm_ci =  GLMMethod.extract_norm_ci_value(data_df, trial_id='trial', feature_id='feature', value_id='value', response_id='response', **self.kwargs)
        
        return scaled_agg_norm_ci, agg_norm_ci
        
        