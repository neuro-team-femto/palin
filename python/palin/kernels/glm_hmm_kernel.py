#!/usr/bin/env python
'''
PALIN toolbox v0.1
December 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

'''

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from .kernel_extractor import KernelExtractor
import ssm  

# class GLMHMMKernel(KernelExtractor):
#     """
#     This class extracts kernel weights using a Generalized Linear Model (GLM).
#     """

#     @classmethod
#     def extract_single_kernel(cls, data_df, trial_id='trial',stim_id='stim', feature_id='feature', value_id='value', response_id='response', **kwargs):
#         """
#         Extracts a single kernel by fitting a GLM to the given data.
#         Returns a DataFrame with feature IDs and kernel values.
#         """

#         # train GLM HMM on data (possibly give priors as parameters)
#         model = cls.train_HMM_from_data(data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs)

#         # take kernel from engaged state       
#         kernel = cls.convert_model_to_kernel(model, feature_id)
        
#         return kernel
    
#     @classmethod
#     def convert_model_to_kernel(csl,model, feature_id='feature'):
#         '''
#         Take kernel from engaged state of a trained HMM model'''
        
#         raise NotImplementedError()


#     @classmethod
#     def train_HMM_from_data(cls, data_df, trial_id='trial',stim_id='stim', feature_id='feature', value_id='value', response_id='response', **kwargs):

#         # Warning: this won't work for 1-int data (and no real way to know whether it's the case)

#         # use ssm library
#         # possibly pass priors as argument (or by default use best priors)        

#         raise NotImplementedError()
        
#     def __str__(self):
#         return 'GLM-HMM Kernel'
class GLMHMMKernel(KernelExtractor):
    """
    This class extracts kernel weights using a Generalized Linear Model-Hidden Markov Model (GLM-HMM).
    """

    @classmethod
    def extract_single_kernel(cls, data_df, trial_id='trial', stim_id='stim', feature_id='feature', 
                              value_id='value', response_id='response', best_priors=None, **kwargs):
        """
        Extracts a single kernel by fitting a GLM-HMM to the given data.
        Returns a DataFrame with feature IDs and kernel values.

        Args:
            data_df (pd.DataFrame): Input data.
            trial_id (str): Column name for trial identifiers.
            stim_id (str): Column name for stimulus identifiers.
            feature_id (str): Column name for feature identifiers.
            value_id (str): Column name for stimulus values.
            response_id (str): Column name for response values.
            best_priors (dict): Optional dictionary of best priors.
            **kwargs: Additional arguments for training.

        Returns:
            tuple:
                - np.ndarray: Extracted kernel weights for each feature.
                - np.ndarray: Posterior probabilities for each trial.
        """
        # Preprocess inputs and responses
        inputs, responses = cls._preprocess_inputs_and_responses(
            data_df, trial_id, stim_id, feature_id, value_id, response_id
        )

        # Train the GLM-HMM model
        model = cls._train_glmhmm(inputs, responses, best_priors, **kwargs)

        # Extract kernel weights
        kernel = cls._extract_kernel_from_model(model)

        # Extract posterior probabilities and predicted states
        posterior_probs, predicted_states = cls._extract_posterior_probabilities(model, responses, inputs)

        return kernel, posterior_probs, predicted_states

    @staticmethod
    def _preprocess_inputs_and_responses(data_df, trial_id, stim_id, feature_id, value_id, response_id):
        """
        Prepares inputs (stimulus differences and choice history) and responses.

        Args:
            data_df (pd.DataFrame): Input data.
            trial_id (str): Column name for trial identifiers.
            stim_id (str): Column name for stimulus identifiers.
            feature_id (str): Column name for feature identifiers.
            value_id (str): Column name for stimulus values.
            response_id (str): Column name for response values.

        Returns:
            tuple: Inputs array and responses array.
        """
        # Process responses
        responses = (
            data_df[data_df[stim_id] == 0]
            .groupby(trial_id)[response_id]
            .first()
            .replace({True: 0, False: 1})
            .tolist()
        )

        # Process stimulus values
        stimulus_values = (
            data_df.groupby([trial_id, stim_id])[value_id]
            .apply(list)
            .unstack()
            .apply(lambda x: [x[0], x[1]], axis=1)
            .tolist()
        )

        # Calculate stimulus differences for each trial
        inputs = []
        for trial in stimulus_values:
            stim_1 = np.array(trial[0])  # First stimulus
            stim_2 = np.array(trial[1])  # Second stimulus
            trial_diffs = stim_2 - stim_1  # Calculate differences between the two stimuli
            inputs.append(trial_diffs)

        # Convert inputs to a numpy array
        inputs_array = np.array(inputs)

        # Shift responses to create choice history
        choice_history_array = pd.Series(responses).shift(1).fillna(0).values.reshape(-1, 1)

        # Combine stimulus differences and choice history into a single input array
        inputs_array_no_noise = np.hstack((inputs_array, choice_history_array))

        # Convert responses to a numpy array
        responses_no_noise_array = np.array(responses).reshape(-1, 1)

        return inputs_array_no_noise, responses_no_noise_array

    @staticmethod
    def _train_glmhmm(inputs, responses, best_priors=None, **kwargs):
        """
        Trains a GLM-HMM model on the provided inputs and responses.

        Args:
            inputs (np.ndarray): Combined inputs (stimulus differences + choice history).
            responses (np.ndarray): Participant responses.
            best_priors (dict): Optional dictionary of best priors.
            **kwargs: Additional arguments for training.

        Returns:
            ssm.HMM: Trained GLM-HMM model.
        """
        num_states = 2
        obs_dim = 1
        input_dim = inputs.shape[1]
        num_categories = 2

        # Use best priors if provided, else default priors
        if best_priors is None:
            best_priors = {
                'mean_value_1': 0.5, 'mean_value_2': 1.0, 'sigma_value_1': 1.0,
                'sigma_value_2': 1.0, 'alpha_value': 2.0
            }

        prior_means = [(0, best_priors['mean_value_1']), (best_priors['mean_value_2'], 0)]
        prior_sigmas = [(0.01, best_priors['sigma_value_1']), (best_priors['sigma_value_2'], 0.01)]
        prior_alpha = best_priors['alpha_value']

        # Instantiate the GLM-HMM model
        glm_hmm = ssm.HMM(
            num_states, obs_dim, input_dim,
            observations="ind_input_driven_obs",
            observation_kwargs=dict(C=num_categories, prior_means=prior_means, prior_sigmas=prior_sigmas),
            transitions="sticky", transition_kwargs=dict(alpha=prior_alpha, kappa=0)
        )

        # Fit the model
        glm_hmm.fit(responses, inputs=inputs, method="em", num_iters=200, tolerance=1e-3)

        return glm_hmm

    @staticmethod
    def _extract_kernel_from_model(model):
        """
        Extracts kernel weights from the trained GLM-HMM model.

        Args:
            model (ssm.HMM): Trained GLM-HMM model.

        Returns:
            np.ndarray: Kernel weights for each state.
        """
        return model.observations.params

    @staticmethod
    def _extract_posterior_probabilities(model, responses, inputs):
        """
        Extracts posterior probabilities and predicted states from the trained GLM-HMM model.

        Args:
            model (ssm.HMM): Trained GLM-HMM model.
            responses (np.ndarray): Participant responses.
            inputs (np.ndarray): Combined inputs (stimulus differences + choice history).

        Returns:
            tuple: 
                - np.ndarray: Posterior probabilities for each trial.
                - np.ndarray: Predicted states for each trial.
        """
        # Extract posterior probabilities using the GLM-HMM model
        posterior_probs = model.expected_states(data=responses, input=inputs)[0]

        # Determine the most likely state for each trial
        predicted_states = np.argmax(posterior_probs, axis=1)

        return posterior_probs, predicted_states

    def __str__(self):
        return "GLM-HMM Kernel"