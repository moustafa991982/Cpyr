import torch
import torch.optim


class BaseOptimizer:
    '''
    Represent base class for optimizers
    provides same interface of an optimizer
    Provides save and load methods
    '''
    def __init__(self,opt,name = None):
        self.opt = opt
        self.name = name
    def __name__(self):
        return self.name
    def zero_grad(self):
        self.opt.zero_grad()
        
    def step(self):
        self.opt.step()
    
    def save(self,PATH):
        '''Path to where optmizer is saved'''
        torch.save(self.opt.state_dict(),PATH)
    def load(self,PATH):
        '''Path from where the optimizer is loaded'''
        self.opt.load_state_dict(torch.load(PATH))
        
    