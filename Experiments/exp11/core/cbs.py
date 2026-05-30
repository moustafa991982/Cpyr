from core.callbacks import *
from core.utils import *
import numpy as np

class Is_best(Callbacks):
    '''
    Search for the best model each epoch and save it if needed, implements callback class.
    '''
    def __init__(self):
        self.meter = np.inf
    def on_val_end(self,**kwargs):
        
        learn = kwargs.get('learn', '')
        special_tag = kwargs.get('tag','')
        
        if self.meter> float(learn.val_loss / learn.epoch):
            self.meter = float(learn.val_loss / learn.epoch)
            tag = gen_tag(**kwargs)
            tag = 'Models/'+special_tag+'is_best_'+tag+'/'
            
            create_dir(tag)
            save_all(tag,**kwargs)
            print('Is Best')
        
