from callbacks import *
from utils import *
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

            

class Checkpoint(Callbacks):
    '''Checkpoint class create a saving point for scheduler, optimizer and model each certain number of cycles
Input:
save_freq: each "save_freq" epochs a checkpoint is created
kwargs [tag]:keywprd added for naming the created dirs 
    '''
    def __init__(self,save_freq):
        self.count = 0
        self.dir = None
        self.save_freq = save_freq
    def on_epoch_begin(self,**kwargs):
        kwargs['include_time'] = False
        
        special_tag = kwargs.get('tag','')
        dir = gen_tag(**kwargs)
        dir = 'checkpoints/'+dir+'_'+special_tag+'/'
        self.dir = dir
        create_dir(dir)
    def on_val_end(self,**kwargs):
        self.count +=1
        if self.count % self.save_freq ==0:
            
            dir = self.dir + f'{self.count}/'
            print(dir)
            create_dir(dir)
            save_all(dir,**kwargs)
    def on_epoch_end(self,**kwargs):
        dir = self.dir + f'{self.count}/'
        print(dir)
        create_dir(dir)
        save_all(dir,**kwargs)        

#
class tensorboard(Callbacks):
    
    def __init__(self,path):
        self.w = tb.SummaryWriter(path)
        self.train_iter = 0
        self.val_iter = 0
        
    def on_train_batch_end(self,**kwargs):
        model = kwargs.get('model',None)
        learn = kwargs.get('learn',None)
        
        
        layer_len = len(model.state_dict())
        k = 0
        
        self.w.add_scalar('Training loss',learn.train_loss/learn.epoch,self.train_iter)
        
        # Create 
        for j,i in model.named_parameters():
            self.w.add_histogram(f'{model.__name__()}: {j}',i,self.train_iter)
            self.w.add_histogram(f'{model.__name__()}: gradients of {j}',i.grad,self.train_iter)
            k+=1
        self.train_iter+=1   
    def on_val_batch_end(self,**kwargs):
        
        learn = kwargs.get('learn',None)
        self.w.add_scalar('Validation loss',learn.val_loss/learn.epoch,self.val_iter)
        self.val_iter +=1
    def on_epoch_end(self,**kwargs):
        self.w.close()


    
