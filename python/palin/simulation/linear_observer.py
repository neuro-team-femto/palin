
import numpy as np
from .observer import Observer

class LinearObserver(Observer):

    def __init__(self, kernel,internal_noise_std,criteria):
        self.kernel = kernel
        self.criteria = criteria
        self.internal_noise_std = internal_noise_std

    @classmethod
    def with_random_kernel(cls, n_features, internal_noise_std,criteria):
        return cls(np.random.uniform(-1,1,n_features),internal_noise_std,criteria)

    def respond_to_stim(self, stim):
        return np.dot(stim, self.kernel)
    
    def generate_internal_noise(self, external_noise_std): 
        return np.random.normal(loc=0, scale=self.internal_noise_std)*external_noise_std
        
    def respond_to_trial(self, trial, experiment): 

        trial_activity = trial.activate(self)
        internal_noise = self.generate_internal_noise(experiment.external_noise_std)
        response = 0 if (trial_activity + internal_noise >= (self.criteria*experiment.external_noise_std)) else 1
        return response

