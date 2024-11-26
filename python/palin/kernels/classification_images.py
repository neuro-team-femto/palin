#!/usr/bin/env python
'''
PALIN toolbox v0.1
Decemberr 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

import pandas as pd
import numpy as np
from .kernel_extractor import KernelExtractor

class ClassificationImage(KernelExtractor):

    @classmethod
    def extract_single_kernel(cls, data_df, trial_id='trial',stim_id='stim', feature_id='feature', value_id='value', response_id='response', **kwargs):

        ## note this doesn't work for 1-int data

        feature_average = data_df.groupby([feature_id,response_id])[value_id].mean().reset_index()
        positives = feature_average.loc[feature_average[response_id] == True].reset_index()
        negatives = feature_average.loc[feature_average[response_id] == False].reset_index()
        kernels = pd.merge(positives, negatives, on=feature_id, suffixes=('_true','_false'))
        kernels['kernel_value'] = kernels['%s_true'%value_id] - kernels['%s_false'%value_id]
        kernels = kernels[[feature_id,'kernel_value']].set_index(feature_id)
        kernels.index.names = ['feature']
        return kernels

    def __str__(self): 
        return 'Classification Image'


     