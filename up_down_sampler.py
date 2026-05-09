import modified_transformer.Models as m
import modified_transformer.SubLayers as sub
import torch
import numpy as np
import torch.nn as nn
from basemodel import *

class down_sample(BaseModel):
    '''
        Model is responsible of downsampling an image by decreasing its filters numbers from 512 to 128
        bs: batch size
        ts: number of time stamps
        Input:
        x(tensor): array of dimensions (bs x timestamp) x filters(512) x height(7) x width(7)
        in_ch(int):Desired number of channels to the Conv Layer
        out_ch(int):Number of desired channels to be output from Conv layer
        kernel_size(tuple):the size of kernel in convolutional layer
        Output:
        res(tensor): out put of dimension bs x timestamp x (filters(128) , height(7) , width(7))
    
        '''
    def __init__(self,bs,ts,in_ch=512,out_ch=128,kernel_size = (1,1),stride=(1,1),name='down sample'):
        super().__init__(name)
        self.shrink = nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride)
        self.bs = bs
        self.ts = ts
    def forward(self,x,bs = None):
        if not bs:
            bs = self.bs
        res = self.shrink(x)
        res = res.view(bs,self.ts,-1)
        return res

#
class up_sample(BaseModel):
    '''
        Model is responsible of upSampling an image by reconstructing its filters numbers from 512 to 128
        bs: batch size
        ts: number of time stamps
        Input:
        x(tensor): array of dimensions (bs x timestamp) x filters(512) x height(7) x width(7)
        in_ch(int):Desired number of channels to the Conv Layer
        out_ch(int):Number of desired channels to be output from Conv layer
        kernel_size(tuple):the size of kernel in convolutional layer
        view(tuple): define the dimension of the output and must be consistent with the expected 
            convolutional layer output 
        Output:
        res(tensor): out put of dimension (bs , timestamp) x filters(128) , height(7) , width(7))
        '''
    def __init__(self,bs,ts,in_ch=128,out_ch=512,kernel_size = (1,1),stride=(1,1),name = 'up_sample',view = (7,7)):
        super().__init__(name)
        self.expand = nn.ConvTranspose2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride)
        self.view = view
        self.bs = bs
        self.ts = ts
        self.in_ch = in_ch
    def forward(self,x,bs=None):
        if not bs:
            bs = self.bs
        dim = [bs*self.ts , self.in_ch]
        dim.extend(self.view)
        x = x.view(dim)
        res = self.expand(x)
        return res

#
class up_down_sample(BaseModel):
    '''
        Model is responsible of upSampling an image by reconstructing its filters numbers from 512 to 128
        bs: batch size
        ts: number of time stamps
        Input:
        x(tensor): array of dimensions (bs x timestamp) x filters(512) x height(7) x width(7)
        in_ch(int):Desired number of channels to the Conv Layer
        out_ch(int):Number of desired channels to be output from Conv layer
        kernel_size(tuple):the size of kernel in convolutional layer
        view(tuple): define the dimension of the output and must be consistent with the expected 
            convolutional layer output of up
        Output:
        res(tensor): out put of dimension (bs , timestamp) x filters(128) , height(7) , width(7))
        '''
    def __init__(self,bs,ts,in_ch=512,out_ch=128,kernel_size=(1,1),stride=(1,1),view=(7,7),name = 'up_sample'):
        assert type(kernel_size) == tuple , 'kernel size must be a tuple' 
        assert type(view) == tuple , 'view must be a tuple' 
        assert type(stride) == tuple , 'stride must be a tuple'
        super().__init__(name)
        self.down = down_sample(bs=bs,ts=ts,in_ch=in_ch,out_ch=out_ch,kernel_size=kernel_size,stride=stride)
        self.up = up_sample(bs=bs,ts=ts,in_ch=out_ch,out_ch=in_ch,kernel_size=kernel_size,stride=stride, view = view)
    def forward(self,x):
        res = self.down(x)
        return self.up(res)


