import torch
from torch.autograd import Variable as V



from tqdm import tqdm
import time
from tqdm import tnrange
import tqdm as tq

from core.meter import *


class learner:
    def __init__(self):
        self.sched = None
        self.train_loss = None
        self.val_loss = None
        self.epoch = None
    def fit(self,model,opt,epoch,dl,crit,callbacks=[],metrics=[],**kwargs):
        # on_epoch_begin
        for i in callbacks:
            i.on_epoch_begin(learn = self, model = model , opt = opt , crit = crit  ,dl = dl,**kwargs)

        for j in tnrange(epoch,desc="Epoch",position=1,mininterval=0):
            self.train_loss = 0
            self.val_loss = 0
            meters = []
            # on_train_begin
            for i in callbacks:
                i.on_train_begin(learn = self, model = model , opt = opt , crit = crit  ,dl = dl,**kwargs)


            t=tqdm(iter(dl.trn_dl), leave=False, total=len(dl.trn_dl), miniters=0 ,postfix = f'Loss: {self.train_loss}')
            model.train()
            self.epoch = 0
            for data in t:
                # on_train_batch_begin
                for i in callbacks:
                    i.on_train_batch_begin(learn = self,model = model , opt = opt , crit = crit ,dl = dl,**kwargs)

                self.epoch += 1
                t.close()
                t.set_postfix_str(f'loss : {float(self.train_loss)/self.epoch}',refresh = True)

                opt.zero_grad()

                data = data.cuda()
                out = model(V(data))

                loss = crit(out,data)
                loss.backward()
                opt.step()

                # on_train_batch_end
                self.train_loss += loss.data

                for i in callbacks:
                    i.on_train_batch_end(learn = self,model = model , opt = opt , crit = crit ,dl = dl,**kwargs)


            # on_val_begin
            for i in callbacks:
                i.on_val_begin(learn = self,model = model , opt = opt , crit = crit  ,dl = dl,**kwargs)

            self.epoch = 0
            with torch.no_grad():


                model.eval()
                t=tqdm(iter(dl.val_dl), leave=False, total=len(dl.val_dl), miniters=0,postfix = f'Loss: {self.val_loss}')
                for  data in t:
                    #on val_batch_begin
                    for i in callbacks:
                        i.on_val_batch_begin(learn = self,model = model , opt = opt , crit = crit  ,dl = dl,**kwargs)
                    self.epoch += 1

                    data = data.cuda() 
                    out = model(V(data))

                    loss = crit(out,data)


                    #on val_batch_end
                    self.val_loss += loss
                    meters.append(calc_metrics(out,data,metrics))
                    for i in callbacks:
                        i.on_val_batch_end(learn = self,model = model , opt = opt , crit = crit  ,dl = dl,meters = meters,**kwargs)

                    t.close()
                    t.set_postfix_str(f'loss : {float(self.val_loss)/self.epoch}',refresh = True)
                
                
                
            
            # on_val_end
            
            for i in callbacks:
                i.on_val_end(learn=self , model = model , opt = opt , crit = crit  ,dl = dl,**kwargs)
            
            self.train_loss /= len(dl.trn_dl)
            self.val_loss /= len(dl.val_dl)
            model.record(self.val_loss)
            print('train Loss: ', self.train_loss.tolist(),'  validation loss: ',
                  self.val_loss.tolist())
            # Needs editing
            print(torch.tensor(meters).mean())
        #on epoch end
        for i in callbacks:
                i.on_epoch_end(learn=self ,model = model , opt = opt , crit = crit  ,dl = dl,meters = meters,**kwargs)
                
def tqdm(*args, **kwargs):
    '''
    Responsible for the gui used to represent the progress of training
    '''
    clear_tqdm()
    return tq.tqdm(*args, **kwargs)

def clear_tqdm():
    '''
    Responsible for the gui used to represent the progress of training
    '''
    inst = getattr(tq.tqdm, '_instances', None)
    if not inst: return
    try:
        for i in range(len(inst)): inst.pop().close()
    except Exception:
        pass
   
            
