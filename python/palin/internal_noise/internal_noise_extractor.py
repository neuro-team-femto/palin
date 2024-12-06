#!/usr/bin/env python
'''
PALIN toolbox v0.1
Decemberr 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class InternalNoiseExtractor(ABC):

    @classmethod
    @abstractmethod
    def extract_single_internal_noise(cls,data_df, trial_id = 'trial', stim_id = 'stim',  feature_id = 'feature', value_id = 'value', response_id = 'response', **kwargs): 
        raise NotImplementedError()

    @classmethod
    def extract_internal_noise(cls,data_df, group_ids, trial_id, stim_id, feature_id, value_id, response_id, **kwargs):
        
        # for each level in group, extract internal_noise
        noises_df = data_df.groupby(group_ids).apply(lambda group: cls.extract_single_internal_noise(group, 
            trial_id, stim_id, feature_id, value_id, response_id, **kwargs)).reset_index()
        noises_df.columns = group_ids + ['internal_noise']
        return noises_df

    