#!/usr/bin/env python
'''
PALIN toolbox v0.1
Decemberr 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class InverseModel(ABC):

    @classmethod
    @abstractmethod
    def build(cls,**kwargs): 
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def run(cls, prob_agree, prob_first, **kwargs):
        raise NotImplementedError()

    