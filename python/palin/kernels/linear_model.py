#!/usr/bin/env python
'''
PALIN toolbox v0.1
Decemberr 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

import pandas as pd
import numpy as np
from .kernels import KernelAnalyser

class LinearModel(KernelExtractor):

    @classmethod
    def extract_single_kernel(cls, data_df, feature_id = 'feature', value_id = 'value', response_id = 'response'):
        
        raise NotImplementedError()

        
 