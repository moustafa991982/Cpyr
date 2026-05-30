import torch
from torch.utils.data import Dataset, DataLoader
from core.callbacks import *
import numpy as np
import os
from torch.utils.data import BatchSampler,SequentialSampler

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

class LKADataset(Dataset):
    '''
    Function: splits dataset to training and validation
    Input: 
    base_dir(str): path to the dataset
    mode(str):define the type of padding required (right,left) default is 'right'
    index(list): list of interger numbers that represent the indicies of the data from base dir.
    Output:
    torch Dataset
    '''
    def __init__(self,base_dir , index=None,**kwargs):
        super().__init__()
        a = lambda x: int(x[5:-4])
        self.word_type = kwargs.get('word_type', float)
        files = os.listdir(base_dir)
        files = sorted(files,key=a)
        
        index = range(len(files)) if index is None else index
        self.data = [base_dir+files[i] for i in index]
        
        generation_type = kwargs.get('generation_type','eval')
        print('generation_type',generation_type)
        self.lka=None
        self.randomize_lka(generation_type)
        
    def __getitem__(self, index):
        word = np.load(self.data[index])  
        word = torch.from_numpy(word)
        lka = self.lka[index]
        return torch.tensor([word,lka]).type(self.word_type)
        
    def __len__(self):
        return len(self.data)
    def randomize_lka(self,generation_type):
        lka = self.generate_lka(generation_type)
        self.lka = lka.repeat(20,1).T.reshape(-1)[:len(self.data)]
        
    def generate_lka(self,generation_type):
        ''' generate lka signal based on hard coded durations.
        Inputs:
        ------
        generation_type str: could be train eval ones zeros exp1-exp5
        '''
        if generation_type == 'train':
            str1 =torch.randint(0,2,(55,))
            steer1 = str1[-1].repeat(48)
            str2 = torch.randint(0,2,(12,))#31
            switch1 = torch.zeros(88)
            str3 = torch.randint(0,2,(13,))
            steer2 = str2[-1].repeat(42)

            trn_lka = torch.cat([str1,steer1,str2,switch1,str3,steer2]).type(torch.float)
            return trn_lka
        
        elif generation_type == 'eval':
            #SEQ_len * length of addded tensor
            repetition = (len(self.data) // (20*16))+1
            #return train of lka -> get the size from the length of the data
            return torch.clone(torch.cat([torch.zeros(8),torch.ones(8)])).repeat(repetition).type(torch.float)
        elif generation_type == 'ones':
            #SEQ_len * length of addded tensor
            repetition = (len(self.data) // (20*8))+1
            #return train of lka -> get the size from the length of the data
            return torch.clone(torch.cat([torch.ones(8)])).repeat(repetition).type(torch.float)
        elif generation_type == 'zeros':
            #SEQ_len * length of addded tensor
            repetition = (len(self.data) // (20*8))+1
            #return train of lka -> get the size from the length of the data
            return torch.clone(torch.cat([torch.zeros(8)])).repeat(repetition).type(torch.float)
        elif generation_type == 'exp1':
            repetition = (len(self.data) // 20)+1
            zeros = torch.zeros(16)
            ones = torch.ones(repetition-16)
            return torch.clone(torch.cat([zeros,ones])).type(torch.float)
        elif generation_type == 'exp2':
            repetition = (len(self.data) // 20)+1
            z1 = torch.zeros(16)#4
            o1 = torch.ones(8)#6
            z2 = torch.zeros(16)#10
            o2 = torch.ones(80)# 30
            z3 = torch.zeros(repetition-30)
            return torch.clone(torch.cat([z1,o1,z2,o2,z3])).type(torch.float)
        elif generation_type == 'exp3':
            repetition = (len(self.data) // 20)+1
            z1 = torch.zeros(40)#10
            o1 = torch.ones(40)#20
            z2 = torch.zeros(repetition-20)
            return torch.clone(torch.cat([z1,o1,z2])).type(torch.float)
        elif generation_type == 'exp4':
            repetition = (len(self.data) // 20)+1
            z1 = torch.zeros(80)#20
            o1 = torch.ones(repetition-20)
            return torch.clone(torch.cat([z1,o1])).type(torch.float)
        elif generation_type == 'exp5':
            repetition = (len(self.data) // 20)+1
            z1 = torch.zeros(120)#30
            o1 = torch.ones(repetition-20)
            return torch.clone(torch.cat([z1,o1])).type(torch.float)


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
    
    
    trn = LKADataset(base_dir , trn_index,generation_type='train',**kwargs)
    val = LKADataset(base_dir , val_index,generation_type='eval',**kwargs)
    if test_pcg:
        
        test = LKADataset(base_dir , test_index,**kwargs)
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
    def __init__(self,dataset,bs,seq_len, overlap ,dim=[112,112],drop_last=True,num_workers=4,generation_type='eval'):
        assert type(dim) == list , 'dim must be a list'
        self.dim = [bs,seq_len]
        self.dim.extend(dim)
        self.bs = bs
        self.seq_len = seq_len
        sam = BatchSampler(SequentialSampler(range(len(dataset))), batch_size=seq_len*bs, drop_last=drop_last)
        self.dl = DataLoader(dataset, batch_sampler=sam,num_workers=num_workers,collate_fn=self.helper)
    def helper(self,x):
        x = torch.stack(x).view(self.dim)
        x,lka = x[:,:,0],x[:,:,1]
        lka = lka.mean(dim=1)
        return x,lka
        #return torch.tensor(x).view(self.bs,self.seq_len)
    def __call__(self):
        return self.dl
    def __len__(self):
        return self.dl.__len__()
    def __iter__(self):
        return iter(self.dl)

class Random_LKA(Callbacks):
    def __init__(self,dataset,generation_type='train'):
        self.ds = dataset
        self.generation_type = generation_type
    def on_train_batch_end(self,**kwargs):
        '''on the end of training loop'''
        self.ds.randomize_lka(self.generation_type)

