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

# class GLMKernel(KernelExtractor):

#     @classmethod
#     def extract_single_kernel(cls, data_df, feature_id = 'feature', value_id = 'value', response_id = 'response', **kwargs):
#     '''
#     Extracts kernel by fitting a GLM to data_df and returning the GLM weights. 
#     Returns a single kernel as a dataframe with feature and kernel_value
#     '''

#         #kernels['kernel_value'] = kernels['%s_true'%value_id] - kernels['%s_false'%value_id]
#         #kernels = kernels[[feature_id,'kernel_value']].set_index(feature_id)
#         #kernels.index.names = ['feature']
#         #return kernels

#     def __str__(self): 
#         return 'GLM Kernel'

class GLMKernel(KernelExtractor):
    """
    This class extracts kernel weights using a Generalized Linear Model (GLM).
    """

    @classmethod
    def extract_single_kernel(cls, data_df, feature_id='feature', value_id='value', response_id='response', **kwargs):
        """
        Extracts a single kernel by fitting a GLM to the given data.
        Returns a DataFrame with feature IDs and kernel values.
        """
        
        model = cls.train_GLM_from_data(data_df, feature_id, value_id, response_id, **kwargs)

        kernel = cls.convert_model_to_kernel(model, feature_id)
        
        return kernel
    
    @classmethod
    def convert_model_to_kernel(csl,model, feature_id='feature'):
        # Extract coefficients
        coefs = model.params[1:]  # Exclude the intercept
        kernel = pd.DataFrame({feature_id: range(len(coefs)), 'kernel_value': coefs.values}).set_index(feature_id)
        return kernel


    @classmethod
    def train_GLM_from_data(cls, data_df, feature_id='feature', value_id='value', response_id='response', **kwargs):
        
        # Preprocess data
        pivoted_df = data_df.pivot_table(index=['trial', 'stim', 'response'],
                                         columns='feature',
                                         values='value',
                                         aggfunc='first').reset_index()
        pivoted_df.columns = ['trial', 'stim_order', 'response'] + [f'value{col}' for col in range(data_df[feature_id].nunique())]
        pivoted_df['response'] = pivoted_df['response'].astype(int)

        df_stim0 = pivoted_df[pivoted_df['stim_order'] == 0].set_index('trial')
        df_stim1 = pivoted_df[pivoted_df['stim_order'] == 1].set_index('trial')
        df_diff = df_stim0.filter(like='value').subtract(df_stim1.filter(like='value')).add_prefix('diff_')
        df_diff['response'] = df_stim0['response'].values
        preprocessed_data = df_diff.reset_index()

        # TODO: add jitter to cover for simulated data with obs with 0 internal noise

        # Fit the GLM
        formula = f'{response_id} ~ {" + ".join(preprocessed_data.filter(like="diff_").columns)}'
        model = smf.glm(formula=formula, data=preprocessed_data, family=Binomial()).fit()

        return model
        

    

    def __str__(self):
        return 'GLM Kernel'


     