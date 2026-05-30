import sklearn.metrics 
import numpy as np
class BaseMeter:
    '''
    Base Class for metrics
    Provides the meters with name method to help in presenting result.
    '''

    def __init__(self,name=None):
        self.name = name
    def __name__(self):
        return self.name

    def shape(self,pred,true):
        '''
        Method check any inconsistency in type or shape of predicted and true values.
        Inconsistency occurs due to different representation needed among loss functions and meters
        Inputs:
        pred(tensor): represent the predictions of the model
        true(tensor): represent the true values of the model
        Output:
        pred: 1D numpy array
        true: 1D numpy array
        '''
        pred = pred.cpu().data.numpy()
        true = true.cpu().data.numpy()
        
        if pred.shape != true.shape:
    
            pred = np.argmax(pred,axis=1)
            
        true = true.reshape(-1)
        pred = pred.reshape(-1)
        return pred,true
    

class accuracy(BaseMeter):
    '''Implementation for accuracy'''
    def __init__(self,name='acc'):
        super().__init__(name)
        self.meter = sklearn.metrics.accuracy_score
    
    def __call__(self,pred,true):
        
        pred,true = super().shape(pred,true)
        return self.meter(true,pred)
    
class f1(BaseMeter):
    '''Implmentation for f1 score'''
    def __init__(self,name='f1'):
        super().__init__(name)
        self.meter = sklearn.metrics.f1_score
        
    def __call__(self,pred,true):
        pred,true = super().shape(pred,true)
        return self.meter(true,pred,average='macro')
    
    
def calc_metrics(pred,true,meters=[]):
    '''Method used to loop over multiple meters provided to fit function'''
    k=[]
    for i in meters:
        k.append(i(pred,true))
    return k