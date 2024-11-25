
from ..analyser import Analyser
from ...internal_noise.glm_method import GLMMethod
import numpy as np

class CIValue(Analyser): 
        
    @classmethod
    def get_metric_names(self):
        return ['confidence_interval']
        
    def analyse(self, experiment, participant, participant_responses): 

        data_df =  self.to_df(experiment, participant_responses)
        ci =  GLMMethod.extract_norm_ci_value(data_df, trial_id='trial', feature_id='feature', value_id='value', response_id='response')
        return [ci]
        
        