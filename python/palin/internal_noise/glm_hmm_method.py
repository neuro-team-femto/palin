#!/usr/bin/env python
'''
PALIN toolbox v0.1
December 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

'''

import pandas as pd
import numpy as np
import os.path
import warnings
import ast
import pickle
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
from ..kernels.glm_hmm_kernel import GLMHMMKernel
from ..kernels.glm_kernel import GLMKernel
from .internal_noise_extractor import InternalNoiseExtractor

from abc import ABC,abstractmethod

# class GLMHMMMethod(InternalNoiseExtractor):
#     """
#     Implements a method to estimate internal noise based on the confidence intervals from a GLM fit.
#     """

#     def __str__(self):
#         return "GLM HMM Method"

#     @classmethod
#     def extract_single_internal_noise(cls, data_df, trial_id='trial', feature_id='feature', value_id='value', response_id='response', **kwargs):
#         """
#         Extracts internal noise for a single observer/session using the GLM fit.

#         Parameters:
#         - data_df (pd.DataFrame): DataFrame containing trial data.
#         - trial_id (str): Column name for trial IDs.
#         - feature_id (str): Column name for feature IDs.
#         - value_id (str): Column name for feature values.
#         - response_id (str): Column name for response values.

#         Returns:
#         - float: Estimated internal noise.
#         """
        
#         # train GLM HMM on data (possibly give priors as parameters)
#         model = GLMHMMKernel.train_HMM_from_data(data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs)

#         # decode original data with model to keep only engaged trials
#         engaged_data_df = csl.decode_engaged_data(data_df, model)

#         # run GLMMethod on engaged data
#         if 'internal_noise_extractor' not in kwargs:
#             raise ValueError('no internal_noise_extractor provided for GLM-HMM Method.') 
        
#         internal_noise = kwargs['internal_noise_extractor'].extract_single_internal_noise(engaged_data_df, trial_id, feature_id, value_id, response_id, **kwargs)
        
#         return internal_noise
    
#     @classmethod
#     def decode_engaged_data(cls, data_df, model):
#         '''
#         Decode original data with model to keep only engaged trials
#         '''
#         raise NotImplementedError()
# class GLMHMMMethod(InternalNoiseExtractor):
#     """
#     Implements a method to estimate internal noise using a specified extractor.
#     Dynamically supports multiple internal noise extractors.
#     """

#     def __str__(self):
#         return "GLM-HMM Method"

#     @classmethod
#     def extract_single_internal_noise(cls, data_df, trial_id='trial', feature_id='feature', value_id='value',
#                                        response_id='response', internal_noise_extractor=None, **kwargs):
#         """
#         Extracts internal noise for a single observer/session using a specified extractor.

#         Parameters:
#         - data_df (pd.DataFrame): DataFrame containing trial data.
#         - trial_id (str): Column name for trial IDs.
#         - feature_id (str): Column name for feature IDs.
#         - value_id (str): Column name for feature values.
#         - response_id (str): Column name for response values.
#         - internal_noise_extractor (class or method): The internal noise extractor to use.
#         - **kwargs: Additional arguments specific to the extractor (e.g., model files).

#         Returns:
#         - float: Estimated internal noise.
#         """
#         if internal_noise_extractor is None:
#             raise ValueError("An `internal_noise_extractor` must be provided.")

#         # Train the GLM-HMM model (if needed)
#         if hasattr(internal_noise_extractor, 'train_HMM_from_data'):
#             model = internal_noise_extractor.train_HMM_from_data(
#                 data_df=data_df,
#                 trial_id=trial_id,
#                 feature_id=feature_id,
#                 value_id=value_id,
#                 response_id=response_id,
#                 **kwargs
#             )
#             # Decode engaged data
#             engaged_data_df = cls.decode_engaged_data(data_df, model)
#         else:
#             # If extractor doesn't train HMM, pass the raw data directly
#             engaged_data_df = data_df

#         # Compute internal noise using the specified extractor
#         internal_noise = internal_noise_extractor.extract_single_internal_noise(
#             engaged_data_df, trial_id=trial_id, feature_id=feature_id,
#             value_id=value_id, response_id=response_id, **kwargs
#         )

#         return internal_noise

#     @classmethod
#     def decode_engaged_data(cls, data_df, model):
#         """
#         Decode original data using the GLM-HMM model to keep only engaged trials.

#         Args:
#             data_df (pd.DataFrame): Original dataset.
#             model: Trained GLM-HMM model.

#         Returns:
#             pd.DataFrame: Filtered dataset containing only engaged trials.
#         """
#         # Extract posterior probabilities
#         posterior_probs, _ = model.predict(data_df)

#         # Filter for engaged trials based on posterior probabilities (state 1 by default)
#         engaged_trials = np.where(np.argmax(posterior_probs, axis=1) == 1)[0]
#         engaged_data_df = data_df[data_df['trial'].isin(engaged_trials)].reset_index(drop=True)

#         return engaged_data_df
class GLMHMMMethod(InternalNoiseExtractor):
    """
    Implements a method to estimate internal noise using GLM-HMM.
    Filters the data to keep only trials in the specified state and
    applies the internal noise extractor to compute noise for the filtered data.
    """

    def __str__(self):
        return "GLM-HMM Method"

    @classmethod
    def extract_single_internal_noise(cls, data_df, trial_id='trial', feature_id='feature', value_id='value',
                                       response_id='response', **kwargs): 
        """
        Extracts internal noise for a single observer/session using the GLM-HMM.

        Parameters:
        - data_df (pd.DataFrame): DataFrame containing trial data.
        - trial_id (str): Column name for trial IDs.
        - feature_id (str): Column name for feature IDs.
        - value_id (str): Column name for feature values.
        - response_id (str): Column name for response values.
        - internal_noise_extractor (class or method): The noise extractor to use.
            - glm_model_file (str): optional Path to the GLM model file (used by some internal_noise_extractor).
            - agreement_model_file (str): optional Path to the agreement model file (used by some internal_noise_extractor).
            - kernel_extractor (class or method): Optional kernel extractor (used by some internal_noise_extractor).
        - state_to_filter (int): State index to filter (default is 1).        
        - **kwargs: Additional arguments (e.g., priors).

        Returns:
        - float: Estimated internal noise.
        """
        if not 'internal_noise_extractor' in kwargs:
            raise ValueError("An `internal_noise_extractor` must be provided.")

        if not 'state_to_filter' in kwargs:
            kwargs['state_to_filter'] = 1
            
        # Train GLM-HMM and extract kernel, posterior probabilities, and predicted states
        kernel, posterior_probs, predicted_states, _ = GLMHMMKernel.extract_single_kernel(
            data_df=data_df,
            trial_id=trial_id,
            feature_id=feature_id,
            value_id=value_id,
            response_id=response_id,
            **kwargs
        )

        # Filter data for the specified state
        filtered_data_df = cls._filter_data_by_state(data_df, posterior_probs, kwargs['state_to_filter'], trial_id)

        # Compute internal noise using the specified internal noise extractor
        internal_noise = kwargs['internal_noise_extractor'].extract_single_internal_noise(
            filtered_data_df, trial_id=trial_id, feature_id=feature_id, value_id=value_id, response_id=response_id, **kwargs)

        return internal_noise

    @staticmethod
    def _filter_data_by_state(data_df, posterior_probs, state_to_filter, trial_id):
        """
        Filters the data to retain only the trials that belong to the specified state.

        Args:
            data_df (pd.DataFrame): Original dataset.
            posterior_probs (np.ndarray): Posterior probabilities for each trial.
            state_to_filter (int): Index of the state to filter (default is 1).
            trial_id (str): Column name for trial identifiers in the dataset.

        Returns:
            pd.DataFrame: Filtered dataset containing only trials in the specified state.
        """
        # Identify trials that belong to the specified state
        state_trials = np.where(np.argmax(posterior_probs, axis=1) == state_to_filter)[0]

        # Filter the DataFrame based on these trials
        filtered_data_df = data_df[data_df[trial_id].isin(state_trials)].reset_index(drop=True)

        return filtered_data_df

# class GLMHMMMethod(InternalNoiseExtractor):
#     """
#     Implements a method to estimate internal noise based on confidence intervals from a GLM fit,
#     focusing on trials identified using a GLM-HMM model.
#     """

#     def __str__(self):
#         return "GLM-HMM Method"

#     @classmethod
#     def extract_single_internal_noise(cls, data_df, trial_id='trial', feature_id='feature', value_id='value',
#                                        response_id='response', **kwargs):
#         """
#         Extracts internal noise for a single observer/session using the GLM fit.

#         Parameters:
#         - data_df (pd.DataFrame): DataFrame containing trial data.
#         - trial_id (str): Column name for trial IDs.
#         - feature_id (str): Column name for feature IDs.
#         - value_id (str): Column name for feature values.
#         - response_id (str): Column name for response values.

#         Returns:
#         - float: Estimated internal noise.
#         """
#         # Check if glm_model_file is provided
#         if 'glm_model_file' not in kwargs:
#             raise ValueError('No model file provided for GLM Method. Use GLMMethod.build_model() before calling.')

#         # Extract confidence interval (CI) on weights from a GLM fit
#         norm_max_feature_ci = cls.extract_norm_ci_value(data_df, trial_id, feature_id, value_id, response_id)

#         # Validate and load the model file
#         model_file = kwargs['glm_model_file']
#         if not os.path.isfile(model_file):
#             raise ValueError('Invalid model file provided for GLM Method.')
#         else:
#             model = sm.load(model_file)
#             ci_df = pd.DataFrame({
#                 'confidence_interval': [norm_max_feature_ci],
#                 'n_trials': [data_df[trial_id].nunique()]
#             })
#             ci_df = sm.add_constant(ci_df)
#             internal_noise = model.predict(ci_df).iloc[0]

#         return internal_noise

#     @classmethod
#     def decode_engaged_data(cls, data_df, posterior_probs, state_to_filter=1, trial_id='trial'):
#         """
#         Filters the data to retain only the trials that belong to the specified state.

#         Args:
#             data_df (pd.DataFrame): Original dataset.
#             posterior_probs (np.ndarray): Posterior probabilities for each trial.
#             state_to_filter (int): Index of the state to filter (default is state 1).
#             trial_id (str): Column name for trial identifiers in the dataset.

#         Returns:
#             pd.DataFrame: Filtered dataset containing only trials in the specified state.
#         """
#         # Identify trials that belong to the specified state
#         state_trials = np.where(np.argmax(posterior_probs, axis=1) == state_to_filter)[0]

#         # Filter the DataFrame based on these trials
#         filtered_data_df = data_df[data_df[trial_id].isin(state_trials)].reset_index(drop=True)

#         return filtered_data_df

#     @classmethod
#     def extract_norm_ci_value(cls, data_df, trial_id='trial', feature_id='feature', value_id='value', response_id='response'):
#         """
#         Extracts the normalized confidence interval value for a GLM fit.

#         Parameters:
#         - data_df (pd.DataFrame): Input DataFrame.
#         - trial_id (str): Column name for trial IDs.
#         - feature_id (str): Column name for feature IDs.
#         - value_id (str): Column name for feature values.
#         - response_id (str): Column name for response values.

#         Returns:
#         - float: Normalized confidence interval value.
#         """
#         # Use GLMKernel to fit GLM and extract kernel and confidence intervals
#         model = GLMKernel.train_GLM_from_data(
#             data_df=data_df,
#             feature_id=feature_id,
#             value_id=value_id,
#             response_id=response_id
#         )

#         # Extract confidence intervals
#         ci = model.conf_int()
#         ci.columns = ['lower_bound', 'upper_bound']
#         ci_df = ci.iloc[1:]  # Exclude the intercept
#         ci_df = ci_df.reset_index(drop=True)

#         kernel_df = GLMKernel.convert_model_to_kernel(model, feature_id)

#         # Calculate confidence interval size
#         kernel_df['conf_int'] = ci_df['upper_bound'] - ci_df['lower_bound']

#         # Normalize confidence intervals
#         kernel_df['norm_ci'] = kernel_df['conf_int'] / kernel_df['kernel_value'].abs()

#         # Calculate normalized maximum feature confidence interval
#         max_feature_ci = kernel_df['norm_ci'].iloc[np.argmax(kernel_df['kernel_value'].abs())]

#         # Scale by the number of trials
#         norm_max_feature_ci = max_feature_ci * np.sqrt(data_df[trial_id].nunique())

#         return norm_max_feature_ci