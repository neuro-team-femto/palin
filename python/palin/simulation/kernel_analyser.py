
from .analyser import Analyser
from palin.metrics import metrics as me

class KernelAnalyser(Analyser): 

    def __init__(self, kernel_analyser):
        self.kernel_analyser = kernel_analyser
        
    def analyse(self, experiment, participant, participant_responses): 

        true_kernel = self.kernel_analyser.normalize_kernel(participant.kernel)

        estimated_kernel = self.estimate_kernel(experiment, participant_responses)

        return self.kernel_correlation(estimated_kernel, true_kernel)

    def estimate_kernel(self, experiment, participant_responses, normalize=True): 

        responses_df = self.to_df(experiment, participant_responses)

        kernel_df = self.kernel_analyser.extract_single_kernel(data_df = responses_df,
        feature_id = 'feature', value_id = 'value', response_id = 'response')

        if normalize: 
            kernel_df = self.kernel_analyser.normalize_kernel(kernel_df)

        return list(kernel_df.kernel_value)

    def normalize_kernel(self, kernel): 

        return self.kernel_analyser.normalize_kernel(kernel)

    def kernel_correlation(self, kernel_1, kernel_2): 
        
        return me.kernel_distance(kernel_1, kernel_2, type='CORR')
