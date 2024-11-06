
from ..analyser import Analyser
#from ...internal_noise.agreement_method import AgreementMethod


class AgreementStatistics(Analyser): 

    def __init__(self, internal_noise_extractor, kernel_extractor = None):
        self.internal_noise_extractor = internal_noise_extractor
        self.kernel_extractor = kernel_extractor

    def get_metric_names(self):
        return ['prob_agree', 'prob_first']

    def analyse(self, experiment, participant, participant_responses): 

        responses_df = self.to_df(experiment, participant_responses)

        prob_agree, prob_first = self.internal_noise_extractor.compute_probabilities(data_df = responses_df, 
            trial_id='trial',stim_id='stim',feature_id='feature',value_id='value', response_id='response', kernel_extractor=self.kernel_extractor)

        return prob_agree, prob_first