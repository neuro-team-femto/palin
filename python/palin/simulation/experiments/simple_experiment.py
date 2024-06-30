import numpy as np

from ..experiment import Experiment

class SimpleExperiment(Experiment): 

    def __init__(self, n_trials, trial_type, n_features, external_noise_std): 
        # init parameters
        self.n_trials = n_trials
        self.trial_type = trial_type
        self.n_features = n_features
        self.external_noise_std = external_noise_std  
        self.trials = None
        # generate trials
        self.generate_trials() 

    def create_stim_noise(self):
        stim_noise = np.random.normal(loc=0, scale=self.external_noise_std, size=self.n_features)
        return range(self.n_features),stim_noise

    def generate_trials(self):
        self.trials = []
        for trial_number in range(self.n_trials):
            stims = []
            num_stim = self.trial_type.n_stims()
            for stim_order in range(num_stim):
                stim_feature, stim_values = self.create_stim_noise()
                stims.append(list(stim_values))
            self.trials.append(self.trial_type(stims))