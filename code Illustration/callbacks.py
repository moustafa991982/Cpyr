class Callbacks:
    '''
    Base class class for all components that will use the callbacks concept
    It contains different functions that represent different stages through the training and validation phase
    Note: **Kwargs attribute must be added to any implemented method from the callbacks
    '''
    def on_epoch_begin(self,**kwargs):
        '''on start of fitting'''
        pass
    def on_train_begin(self,**kwargs):
        '''Before starting the training loop'''
        pass
    def on_train_batch_begin(self,**kwargs):
        '''on the beginning of training loop'''
        pass
    def on_train_batch_end(self,**kwargs):
        '''on the end of training loop'''
        pass
    
    
    def on_val_begin(self,**kwargs):
        '''Before startig the validation loop'''
        pass
    def on_val_batch_begin(self,**kwargs):
        '''on the beginning of validation loop'''
        pass
    def on_val_batch_end(self,**kwargs):
        '''on the end of validation loop'''
        pass
    
    def on_val_end(self,**kwargs):
        '''on the end of validation loop'''
        pass
    
    def on_epoch_end(self,**kwargs):
        '''on start of fitting'''
        pass