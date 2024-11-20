
from ..analyser import Analyser
from palin.internal_noise.double_pass import DoublePass

class DoublePassStatistics(Analyser): 

    @classmethod
    def get_metric_names(cls):
        return ['prob_agree', 'prob_first']

    def analyse(self, experiment, participant, participant_responses): 

        responses_df = self.to_df(experiment, participant_responses)

        # index double_pass
        responses_df = DoublePass.index_double_pass_trials(data_df = responses_df, 
            trial_id='trial',value_id='value', double_pass_id='double_pass_id')

        # compute probability of agreement over double pass
        prob_agree = DoublePass.compute_prob_agreement(responses_df, trial_id='trial', response_id='response', double_pass_id='double_pass_id')
        # compute probability of choosing first response option
        prob_first = DoublePass.compute_prob_first(responses_df, trial_id='trial', response_id='response', stim_id='stim', double_pass_id='double_pass_id')

        return prob_agree, prob_first