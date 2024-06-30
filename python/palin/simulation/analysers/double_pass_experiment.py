import numpy as np

from .simple_experiment import SimpleExperiment


class DoublePassExperiment(SimpleExperiment): 

    def __init__(self, n_trials, n_repeated, trial_type, n_features, external_noise_std): 
        # init parameters
        self.n_repeated = n_repeated
        super().__init__(n_trials, trial_type, n_features, external_noise_std)
        # generate trials
        self.generate_trials() 

    def generate_trials(self):
        
        # generate n_trials
        super().generate_trials()

        # and then an extra n_repeated trials copied from the original trials (first n_repeated trials) 
        if self.n_repeated > self.n_trials: 
            self.n_repeated = self.n_trials
        self.trials += self.trials[:self.n_repeated]