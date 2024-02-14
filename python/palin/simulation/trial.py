from abc import ABC, abstractmethod

class Trial(ABC):

    @abstractmethod
    def activate(self,participant): 
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def n_stims(self):
        raise NotImplementedError()

class Int1Trial(Trial): 

    def __init__(self,stims):
        self.stims = stims

    def activate(self,obs): 
        return obs.respond_to_stim(self.stims[0])

    @classmethod
    def n_stims(self):
        return 1

class Int2Trial(Trial): 

    def __init__(self,stims):
        self.stims = stims

    def activate(self,obs): 
        return obs.respond_to_stim(self.stims[0]) - obs.respond_to_stim(self.stims[1])

    @classmethod
    def n_stims(self):
        return 2

        

