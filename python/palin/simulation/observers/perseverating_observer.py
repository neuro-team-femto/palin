import numpy as np
from ..observer import Observer
#from .linear_observer import LinearObserver

class PerseveratingObserver(Observer):

    def __init__(self, kernel, internal_noise_std, criteria, transition_matrix=None):
        #super().__init__(kernel, internal_noise_std, criteria)
        if transition_matrix is None or not (len(transition_matrix) == 2 and len(transition_matrix[0]) == 2):
            raise ValueError("transition_matrix must be a 2x2 matrix with probabilities.")
        self.kernel = kernel
        self.internal_noise_std = internal_noise_std
        self.criteria = criteria
        self.transition_matrix = transition_matrix
        self.state = "ENG"  # Initial state
        self.last_response = None
        self.last_state_sequence = []

    @classmethod
    def with_random_kernel(cls, n_features, internal_noise_std, criteria, transition_matrix, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        return cls(np.random.uniform(-1, 1, n_features), internal_noise_std, criteria, transition_matrix)
    
    def respond_to_stim(self, stim):
        # Calculate a linear response using the kernel
        if isinstance(self.kernel, str) and self.kernel == 'random':
            # Create a random kernel if specified
            self.kernel = np.random.uniform(-1, 1, len(stim))
        return np.dot(stim, self.kernel)

    def generate_internal_noise(self, external_noise_std):
        # Generate internal noise based on the standard deviation and external noise level
        return np.random.normal(loc=0, scale=self.internal_noise_std) * external_noise_std

    def respond_to_trial(self, trial, experiment):
        if hasattr(trial, 'stims'):
            stim = trial.stims[0]
        # Check the current state and generate the appropriate response
        if self.state == "ENG":
            trial_activity = self.respond_to_stim(stim)
            internal_noise = self.generate_internal_noise(experiment.external_noise_std)
            response = 0 if (trial_activity + internal_noise >= (self.criteria * experiment.external_noise_std)) else 1
        else:  # Perseverative state
            # In the perseverative state, repeat the last response if available
            response = self.last_response if self.last_response is not None else (
                0 if (self.respond_to_stim(stim) + self.generate_internal_noise(experiment.external_noise_std) >= (self.criteria * experiment.external_noise_std)) else 1
            )
        
        # Update the last response and the current state
        self.last_response = response
        self.update_state()  # Update the state using the transition matrix
        self.last_state_sequence.append(self.state)  # Store the current state in the sequence

        return response

    def update_state(self):
        current_state_index = 0 if self.state == "ENG" else 1
        next_state_probabilities = self.transition_matrix[current_state_index]
        if not isinstance(next_state_probabilities, (list, np.ndarray)):
            raise ValueError("The transition matrix is incorrectly formatted. It must be a 2D list or array.")
        self.state = "ENG" if np.random.rand() < next_state_probabilities[0] else "PER"

    def respond_to_experiment(self, experiment):
        self.last_state_sequence = []  # Reset state sequence for new experiment
        responses = []
        for trial in experiment.trials:
            responses.append(self.respond_to_trial(trial, experiment))
        return responses
