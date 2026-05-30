import torch
import torch.nn as nn
import torch.nn.functional as f

class BaseModel(nn.Module):
    '''
    Base model for all customized modules
    It provides the model with basic features as saving and loading its state dict
    
    
    Note:Inputs provided from the dataset are in the type and shape needed for the loss function
    So Models must:
    1- Ensure consistency of Input to its components regarding type and dimensions
    2- Return the output in the shape and type needed for the loss function
    
    '''
    def __init__(self,name='BaseModel'):
        super().__init__()
        self.name = name
    def __name__(self):
        return self.name
    def record(self,*args):
        pass
    def save(self,PATH):
        '''Path to save the model'''
        #print(os.listdir())
        torch.save(self.state_dict(),PATH)
    def load(self,PATH):
        '''path from where the model is loaded'''
        self.load_state_dict(torch.load(PATH))
        self.eval()