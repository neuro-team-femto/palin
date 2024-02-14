from abc import ABC, abstractmethod

class Experiment(ABC): 

    # possibly yield next_trial(), if we want more abstraction ?
    
    @abstractmethod
    def generate_trials(self):
        raise NotImplementedError()