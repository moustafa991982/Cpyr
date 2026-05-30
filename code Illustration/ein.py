import modified_transformer.Models as m
import modified_transformer.SubLayers as sub
import torch
import numpy as np
import torch.nn as nn
from modified_transformer.Models import *
from modified_transformer.Layers import DecoderLayer
from up_down_sampler import *
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

#

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, out_ch = 128 ,dropout=0.0):

        super().__init__()
        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        self.lin = nn.Linear(out_ch*7*7 , d_model)
    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq[:,:,0])

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq[:,:,0])
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq[:,:,0], seq_q=tgt_seq[:,:,0])
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq[:,:,0], seq_q=tgt_seq[:,:,0])

        # -- Forward
    
        dec_output = self.lin(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]
        
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

##

class Transformer_Dec(nn.Module):
    ''' A sequence to sequence model with attention mechanism. 
    
    len_max_seq(int): Max time stamps used (important for variable length input)
    d_word_vec(int): Defines the size of position embedding
    d_model(int): Defines the shape of data that will be outputed from each block
    n_head:(int): Represent the number of heads used in multi-Head attention block
    out_ch(int): The number of filters to input and output to transformer
    '''

    def __init__(
            self,
            len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            out_ch = 128):

        super().__init__()
 
        self.decoder = Decoder(
            len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout,out_ch = out_ch)

        self.reconstruct = nn.Linear(d_model, out_ch*7*7, bias=False)
        nn.init.xavier_normal_(self.reconstruct.weight)
        
        ## Dimensions (must change on changing bs)
        self.src_seq_dim = torch.rand(4,1,512)
        
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'


    def forward(self, src_seq, enc_output, tgt_seq, tgt_pos):

        dec_output , *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        out = self.reconstruct(dec_output)
        return out

##

class UneTrans(BaseModel):
    '''
    UneTrans is considered the mother model that collects all the componets of Unet model combined with Transformer.
    Input:
    bs(int): batch size
    len_max_seq(int): Max time stamps used (important for variable length input)
    d_word_vec(int): Defines the size of position embedding
    d_model(int): Defines the shape of data that will be outputed from each block
    n_head:(int): Represent the number of heads used in multi-Head attention block
    out_ch(int): The number of filters to input and output to transformer
    Input:
    x(tensor): Must be of shape (bs,ts) x 112 x 112
    output:
    unet_decoded(tensor): has shape of (bs,ts) x 112 x 112
    '''
    def __init__(self,bs,len_max_seq,
            d_word_vec=2048, d_model=2048, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            out_ch = 128,device = 'cuda',name = 'Unet with Transformers'):
        super().__init__(name)
        
        self.uds = up_down_sample(bs = bs ,ts = len_max_seq , out_ch=out_ch)
        
        self.uds_enc = up_down_sample(bs = bs ,ts = 1 , out_ch=out_ch)
        
        self.transformer = Transformer_Dec(len_max_seq =len_max_seq ,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
            out_ch = out_ch)
        
        self.lin = nn.Linear(out_ch*7*7,d_model)
        
    def forward(self,*ip):
        
        enc = ip[1]
        x = ip[0]
        bs = enc.size(0)
        ts = x.size(0) // bs
        
        #print('bottle_neck',unet_encoding[-1].shape)
        unet_bottleneck_downSampled = self.uds.down(x,bs)
        enc = self.uds_enc.down(enc,bs)
        
        enc = self.lin(enc)
        #print('bottle_neck',unet_bottleneck_downSampled.shape)
        # Preparing position
        
        b = [torch.arange(1,ts+1).view(1,-1) for i in range(bs)]
        b = torch.cat(b,dim=0).to(x.device)
        
        transformer_output = self.transformer(enc,enc,unet_bottleneck_downSampled,b)
        
        transformer_output_upSampled = self.uds.up(transformer_output,bs)
        
        return transformer_output_upSampled

##

class Ein(BaseModel):
    def __init__(self,bs,seq_len,in_ch=512,out_ch=128,dim=7,d_word_vec=2048, d_model=2048, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,device='cuda',name='einsum'):
        super().__init__(name)
        #self.Conv = nn.Conv2d(in_ch,out_ch,kernel_size=(1,1))
        
        self.par = nn.Parameter(torch.rand(in_ch,dim,dim))
        
        self.trans = UneTrans(bs=bs,len_max_seq=seq_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
            out_ch = out_ch,device='cuda')
    def forward(self,x):
        res = torch.einsum('bschw,chw->bchw',x,self.par)
        temp = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        res = self.trans(temp,res)
        res = 2*torch.sigmoid(res)
        return res.view(x.size(0),x.size(1),x.size(2),x.size(3),x.size(4))    




