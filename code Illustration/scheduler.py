import torch
from callbacks import *


class scheduler(Callbacks):
    '''
    Represent a wrapper to torch schedulers, it implements the callback class.

    '''
    def __init__(self,sched,name='BaseSched'):
        self.sched = sched
        self.attr = self.extract() 
        self.name = name
    def __name__(self):
        '''Used in tag generation'''
        return self.name
    def step(self):
        '''perform scheduler step'''
        self.sched.step()
    def get_lr(self):
        return self.sched.get_lr()
    
    def on_epoch_begin(self,**kwargs):
        '''Set the scheduler as the attribute of fit this helps in passing sched in kwargs'''
        learn = kwargs.get('learn', None)
        if learn !=None:
            learn.sched = self
    def on_train_batch_end(self,**kwargs):
        self.step()
    
    def on_epoch_end(self,**kwargs):
        self.reset()
    
    def extract(self):
        '''
        Extracts numeric parameters of scheduler to reset it when needed 
        '''
        attr = dict()
        for i in dir(self.sched):
            
            if f'{getattr(self.sched,i)}'.replace('-',"").isnumeric():
                attr[i] = getattr(self.sched,i)
        return attr
    
    def reset(self):
        '''
        Gurantee that at the end of the epoch the lr will be reset
        This is important when using SGDR  
        '''
        for i in self.attr:
            setattr(self.sched,i,self.attr[i])
        
    def save(self,PATH):
        '''Path to save the scheduler'''
        torch.save(self.sched.state_dict(),PATH)
    def load(self,PATH):
        '''path from where the scheduler is loaded'''
        self.sched.load_state_dict(torch.load(PATH))