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


class GLMHMMMethod(InternalNoiseExtractor):
    """
    Implements a method to estimate internal noise using GLM-HMM.
    Filters the data to keep only trials in the specified state and
    applies the internal noise extractor to compute noise for the filtered data.
    """

    def __str__(self):
        return "GLM-HMM Method"

    @classmethod
    def extract_single_internal_noise(cls, data_df, trial_id='trial', stim_id = 'stim', feature_id='feature', value_id='value',
                                       response_id='response', **kwargs): 
        """
        Extracts internal noise for a single observer/session using the GLM-HMM.

        Parameters:
        - data_df (pd.DataFrame): DataFrame containing trial data.
        - trial_id (str): Column name for trial IDs.
        - feature_id (str): Column name for feature IDs.
        - value_id (str): Column name for feature values.
        - response_id (str): Column name for response values.
        - glmhmm_internal_noise_extractor (class or method): The noise extractor to use.
            - glm_model_file (str): optional Path to the GLM model file (used by some internal_noise_extractor).
            - agreement_model_file (str): optional Path to the agreement model file (used by some internal_noise_extractor).
            - kernel_extractor (class or method): Optional kernel extractor (used by some internal_noise_extractor).
        - state_to_filter (int): State index to filter (default is 1).        
        - **kwargs: Additional arguments (e.g., priors).

        Returns:
        - float: Estimated internal noise.
        """
        if not 'glmhmm_internal_noise_extractor' in kwargs:
            raise ValueError("An `glmhmm_internal_noise_extractor` must be provided.")
       

        if not 'state_to_filter' in kwargs:
            kwargs['state_to_filter'] = 1
            
        # Train GLM-HMM and extract kernel, posterior probabilities, and predicted states
        kernel, posterior_probs, predicted_states, _ = GLMHMMKernel.extract_single_kernel(
            data_df=data_df,
            trial_id=trial_id, 
            stim_id = stim_id, 
            feature_id=feature_id,
            value_id=value_id,
            response_id=response_id,
            return_probs= True,
            **kwargs
        )

        # Filter data for the specified state
        filtered_data_df = cls._filter_data_by_state(data_df, posterior_probs, kwargs['state_to_filter'], trial_id)

        if filtered_data_df[trial_id].nunique() == 0:
            print("Warning: No trials in engaged state. Returning internal noise = nan")
            return np.nan
        
        print('engaged trials:', filtered_data_df[trial_id].nunique())
        # Compute internal noise using the specified internal noise extractor
        internal_noise = kwargs['glmhmm_internal_noise_extractor'].extract_single_internal_noise(
            filtered_data_df, trial_id=trial_id, stim_id = stim_id, feature_id=feature_id, value_id=value_id, response_id=response_id, **kwargs)

        print(internal_noise)
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

