
from ..analyser import Analyser
from ...internal_noise.glm_method import GLMMethod


class ConfIntMetrics(Analyser): 

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def get_metric_names(cls):
        return ['norm_max_ci', 'max_ci']

    def analyse(self, experiment, participant, participant_responses): 

        responses_df = self.to_df(experiment, participant_responses)

        norm_max_feature_ci,max_feature_ci=GLMMethod.extract_norm_ci_value(data_df = responses_df, trial_id='trial',feature_id='feature',value_id='value', response_id='response')

        return norm_max_feature_ci, max_feature_ci