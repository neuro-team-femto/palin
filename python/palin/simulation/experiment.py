from abc import ABC, abstractmethod

class Experiment(ABC): 
    '''
        Abstract class that represents a simulated experimental paradigm, with trials that are then submitted to a simulated observer. 
        See e.g. @SimpleExperiment for a example of implementation
    '''

    # possibly yield next_trial(), if we want more abstraction ?
    
    @abstractmethod
    def generate_trials(self):
        '''
            Generates the experiment's trials, to be called upon __init__()
        '''
        raise NotImplementedError()