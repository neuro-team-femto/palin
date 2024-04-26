from abc import ABC, abstractmethod


class Observer(ABC):

    @abstractmethod
    def respond_to_stim(self, stim):
        raise NotImplementedError()
    
    @abstractmethod
    def respond_to_trial(self, trial, experiment): 
        raise NotImplementedError()

    def respond_to_experiment(self,experiment): 
        responses = []
        for trial in experiment.trials: 
            responses.append(self.respond_to_trial(trial, experiment))
        return responses       
