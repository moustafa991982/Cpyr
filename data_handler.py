import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import *
import numpy as np
import os

class dataset(Dataset):
    '''
    Function: splits dataset to training and validation
    Input: 
    base_dir(str): path to the dataset
    mode(str):define the type of padding required (right,left) default is 'right'
    index(list): list of interger numbers that represent the indicies of the data from base dir.
    Output:
    torch Dataset
    '''
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
        try:
            word_dim = len(word)
        except:
            
            word = word.reshape(1,)
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

#

class OverlappingSampler(Sampler):
    r"""Samples elements sequentially, applies the concept of overlapping.
    Some data will be repeated in the next mini-batch as desired
    if overlap is set to be 0 the sampler acts as a normal sequential sampler
    Arguments:
        data_source (Dataset): dataset to sample from
        seq_len(int): indicates the length of the single example (length of words or packets)
        overlap(int): Defines the number of words to be repeated and it must be less than the seq_len
    """
    def __init__(self, data_source,seq_len,overlap,bs):
        assert overlap < seq_len , 'Overlap must be smaller than the sequence length'
        self.data_source = data_source
        self.seq_len = seq_len
        self.overlap = overlap
        self.bs=bs
        self.idx = []
        self.corrector = 0
        self.get_list()
    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)
    
    def get_list(self):
        i=0
        j =-1
        self.idx.append(i)
        while i != len(self.data_source)-1:
            j+=1
            i=j
            i -=self.corrector   
            self.idx.append(i)
            if len(self.idx)%self.seq_len == 0:
                self.corrector+=self.overlap
            if len(self.idx)%(self.bs*self.seq_len) == 1:
                self.idx.append(i)

#

class Sequence_dl:
    '''
    Class that returns an iterator over the desired dataset with the ability to achieve overlaps
    Some data will be repeated in the next mini-batch as desired
    if overlap is set to be 0 the sampler acts as a normal sequential sampler
    args:
    dataset:torch dataset object
    bs(int): batch size
    seq_len(int): indicates the length of the single example (length of words or packets)
    overlap(int): Defines the number of words to be repeated and it must be less than the seq_len
    dim(list):Indicates the dimension of a single example
    '''
    def __init__(self,dataset,bs,seq_len, overlap ,dim=[112,112] , drop_last=True,num_workers=4):
        assert type(dim) == list , 'dim must be a list'
        self.dim = dim
        self.bs = bs
        self.seq_len = seq_len
        sam = BatchSampler(OverlappingSampler(dataset,seq_len,overlap=overlap,bs=bs), batch_size=seq_len*bs, drop_last=drop_last)
        self.dl = DataLoader( dataset, batch_sampler=sam,num_workers=num_workers,collate_fn=self.helper)
    def helper(self,x):
        dim = self.dim.copy()       
        dim.reverse()
        dim.extend([self.seq_len,self.bs])
        #self.dim.append(self.bs)
        dim.reverse()
        sizes = [len(i) for i in x]
        return torch.stack(x).view(dim)
        #return torch.tensor(x).view(self.bs,self.seq_len)
    def __call__(self):
        return self.dl
    def __len__(self):
        return self.dl.__len__()
    def __iter__(self):
        return iter(self.dl)
