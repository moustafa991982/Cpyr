import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable as V

from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
import time
from tqdm import tnrange
import tqdm as tq

def fit(model,opt,epoch,dl,crit):
    for j in tnrange(epoch,desc="Epoch",position=1,mininterval=0):
        train_loss = 0
        val_loss = 0
        
        t=tqdm(iter(dl.trn_dl), leave=False, total=len(dl.trn_dl), miniters=0 ,postfix = f'Loss: {train_loss}')
        model.train()
        k = 0
        for data in t:
            k+=1
            t.close()
            t.set_postfix_str(f'loss : {float(train_loss)/k}',refresh = True)
            
            opt.zero_grad()
            
            data = data.type(torch.float).cuda()
            out = model(V(data))
            
            loss = crit(out,data)
            loss.backward()
            opt.step()
            
            train_loss += loss.data
            
        k = 0    
        with torch.no_grad():
            model.eval()
            t=tqdm(iter(dl.val_dl), leave=False, total=len(dl.val_dl), miniters=0,postfix = f'Loss: {val_loss}')
            for  data in t:
                k+=1
                
                data = data.type(torch.float).cuda() 
                out = model(V(data))
                loss = crit(out,data)
                val_loss += loss
                
                t.close()
                t.set_postfix_str(f'loss : {float(val_loss)/k}',refresh = True)
            model.record(loss)
            train_loss /= len(dl.trn_dl)
            val_loss /= len(dl.val_dl)
            print('train Loss: ', train_loss.tolist(),'  validation loss: ',val_loss.tolist())
            
        
        
class CombineDataLoader:
    def __init__(self,trn_dl,val_dl = None):
        self.trn_dl = trn_dl
        self.val_dl = val_dl 

class dataset(Dataset):
    def __init__(self,base_dir , index ):
        super().__init__()
        
        files = os.listdir(base_dir)
        files = files
        
        names = [files[i] for i in index] 
        
        self.data  = [ base_dir +i for i in  names]
        
    def __getitem__(self, index):
        
        example = np.load(self.data[index]).astype(np.float).reshape(1,112,112)
        return example
        
    def __len__(self):
        return len(self.data)
        raise NotImplementedError

def split(base_dir,prc=0.2 , dataset_pct = 1):
    
    total_size = int(len(os.listdir(base_dir)) * dataset_pct)
    
    val_size = int(total_size*prc)
    
    trn_index = np.arange(total_size - val_size)
    val_index = np.arange(trn_index[-1]+1,total_size)
    
    val = dataset(base_dir , val_index)
    trn = dataset(base_dir , trn_index)
    return trn , val

def tqdm(*args, **kwargs):
        clear_tqdm()
        return tq.tqdm(*args, **kwargs)

def clear_tqdm():
    inst = getattr(tq.tqdm, '_instances', None)
    if not inst: return
    try:
        for i in range(len(inst)): inst.pop().close()
    except Exception:
        pass
   
    
    
   