
from ..analyser import Analyser
#from ...internal_noise.agreement_method import AgreementMethod


class AgreementStatistics(Analyser): 

    def __init__(self, internal_noise_extractor, **kwargs):
        self.internal_noise_extractor = internal_noise_extractor
        self.kwargs = kwargs

    @classmethod
    def get_metric_names(cls):
        return ['prob_agree', 'prob_first']

    def analyse(self, experiment, participant, participant_responses): 

        responses_df = self.to_df(experiment, participant_responses)

        prob_agree, prob_first = self.internal_noise_extractor.compute_probabilities(data_df = responses_df, 
            trial_id='trial',stim_id='stim',feature_id='feature',value_id='value', response_id='response', **self.kwargs)

        return prob_agree, prob_first