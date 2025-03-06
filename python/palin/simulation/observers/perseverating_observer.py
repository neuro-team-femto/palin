import numpy as np
from ..observer import Observer

class PerseveratingObserver(Observer):

    def __init__(self, kernel, internal_noise_std, criteria, transition_matrix):
        if transition_matrix is None or not (
            len(transition_matrix) == 2 and len(transition_matrix[0]) == 2
        ):
            raise ValueError("transition_matrix must be a 2x2 matrix with probabilities.")
        
        self.kernel = kernel
        self.internal_noise_std = internal_noise_std
        self.criteria = criteria
        self.transition_matrix = transition_matrix
        self.state = "ENG"  # Initial state (ENG = engaged, PER = perseverative)
        self.last_response = None
        self.last_state_sequence = []

    @classmethod
    def with_random_kernel(cls, n_features, internal_noise_std, criteria, transition_matrix, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        random_kernel = np.random.uniform(-1, 1, n_features)
        return cls(random_kernel, internal_noise_std, criteria, transition_matrix)

    def respond_to_stim(self, stim):
    
        # If kernel is 'random', initialize it
        if isinstance(self.kernel, str) and self.kernel == 'random':
            self.kernel = np.random.uniform(-1, 1, len(stim))

        # Ensure kernel and stim are numpy arrays of type float
        if not isinstance(self.kernel, np.ndarray):
            self.kernel = np.array(self.kernel, dtype=float)
        if not isinstance(stim, np.ndarray):
            stim = np.array(stim, dtype=float)

        return np.dot(stim, self.kernel)
    # Debug: Print the types of stim and kernel
    

    def generate_internal_noise(self, external_noise_std):
        # Generate internal noise based on the standard deviation and external noise level
        return np.random.normal(loc=0, scale=self.internal_noise_std) * external_noise_std

    def respond_to_trial(self, trial, experiment):
        stim = trial.stims[0] if hasattr(trial, 'stims') else None
        if stim is None:
            raise ValueError("Trial does not contain stimuli")
        
        if self.state == "ENG":
            # Respond based on kernel and internal noise
            trial_activity = self.respond_to_stim(stim)
            internal_noise = self.generate_internal_noise(experiment.external_noise_std)
            response = 0 if (trial_activity + internal_noise >= self.criteria * experiment.external_noise_std) else 1
        else:  # Perseverative state
            # Repeat the last response, or default to 0 if no prior response exists
            response = self.last_response if self.last_response is not None else 0

        # Update state and store response
        self.last_response = response
        self.update_state()  # Determine the next state
        self.last_state_sequence.append(self.state)

        return response

    def update_state(self):
        # Transition to the next state based on the transition matrix
        current_state_index = 0 if self.state == "ENG" else 1
        next_state_probabilities = self.transition_matrix[current_state_index]
        self.state = "ENG" if np.random.rand() < next_state_probabilities[0] else "PER"

    def respond_to_experiment(self, experiment):
        # Ensure the state sequence resets for each experiment
        self.last_state_sequence = []
        responses = []
        for trial in experiment.trials:
            responses.append(self.respond_to_trial(trial, experiment))
        return responses

    def get_latest_states(self):
        """
        Returns the sequence of states (ENG/PER) from the last experiment.
        """
        return self.last_state_sequence
