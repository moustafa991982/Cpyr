import torch
from core.callbacks import *
from tqdm import tqdm
class Validate(Callbacks):
    '''Evaluate a Dataset after training ends
    '''
    def __init__(self,dl):
        self.dl = dl
        self.test_loss = []
        
        self.meter_loss = 0
    def on_epoch_end(self,**kwargs):
        model = kwargs.get('model',False)
        crit = kwargs.get('crit',False)
        
        self.test_loss = []
        with torch.no_grad():
            model.eval()
            t=tqdm(iter(self.dl), leave=False, total=len(self.dl), miniters=0)
            print('testing...')
    
            for data in t:
                if next(model.parameters()).is_cuda:
                    data = data.cuda() if type(data) not in [list,tuple] else [i.cuda() for i in data]
                out = model(*data)
                loss = crit(out,data)
                
                self.test_loss.append(loss)
                
                t.close()
    def evaluate(self,**kwargs):
        self.on_epoch_end(**kwargs)
