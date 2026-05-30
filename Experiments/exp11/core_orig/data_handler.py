import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class dataset(Dataset):

    def __init__(self,base_dir , index ,dim = None,**kwargs):
        super().__init__()
        a = lambda x: int(x[5:-4])
        
        self.mode = kwargs.get('mode', 'right') 
        self.word_type = kwargs.get('word_type', float)
        
        self.dim = dim        
        files = os.listdir(base_dir)
        files = sorted(files,key=a) 
        self.data = [base_dir+files[i] for i in index]
        
    def __getitem__(self, index):
        
        word = np.load(self.data[index])
        if self.dim !=None:
            word = self.pad(word)    
        word = torch.from_numpy(word.reshape(self.dim))
        return word.type(self.word_type)
        
    def __len__(self):
        return len(self.data)
        
    def pad(self,word):
        
        total_len = np.array(self.dim).prod()
        #Get the differnce between target length and given word length
        word_dim = len(word)
        
        diff = total_len - word_dim
       
        #Check the model center or right
        if self.mode == 'right':
            return np.pad(word,(0,diff),'constant')
        else:
            left_pad = diff//2
            return np.pad(word,(left_pad , diff-left_pad),'constant')

#
def split(base_dir, prc=0.2 ,test_pcg = None ,dataset_pct = 1,**kwargs):
    '''
    Function: splits dataset to training and validation
    Input: 
    base_dir(str): path to the dataset
    prc(float) : ratio between validation set size and dataset size
    dataset_pct(float) :percentage of dataset to be used
    Output:
    train dataset
    Val dataset
    Example:
    
    split('dataset/binary dataset/',dataset_pct=0.12,dim = (1,112,112),  mode= 'center' ,word_type = torch.LongTensor)
    '''
    total_size = int(len(os.listdir(base_dir)) * dataset_pct)
    
    if test_pcg:
        
        val_size = int(total_size*prc)
        test_size = int(total_size*test_pcg)
        trn_index = np.arange(total_size - val_size - test_size)
        val_index = np.arange(trn_index[-1]+1,trn_index[-1]+ val_size +1 )
        test_index = np.arange(trn_index[-1]+ val_size +2,total_size )
        
        
        
    
    else:
        val_size = int(total_size*prc)
        trn_index = np.arange(total_size - val_size)
        val_index = np.arange(trn_index[-1]+1,total_size)
    
    
    trn = dataset(base_dir , trn_index,**kwargs)
    val = dataset(base_dir , val_index,**kwargs)
    if test_pcg:
        
        test = dataset(base_dir , test_index,**kwargs)
        return trn,val,test
    return trn,val
#
class CombineDataLoader:
    '''
    Takes train dataloader and validation dataloader and combine the in one object
    '''
    def __init__(self,trn_dl,val_dl = None):
        self.trn_dl = trn_dl
        self.val_dl = val_dl
