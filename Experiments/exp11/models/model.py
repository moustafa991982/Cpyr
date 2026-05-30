import torch.nn as nn
from core.basemodel import *

class LKAPredictor(BaseModel):    
    def __init__(self,bs,seq_len,rcn_w = 10,compress_hs=10,pred_hs=128,dp=0.4,enhncd=True):
        super().__init__()
        
        self.bs,self.seq_len,self.enhncd = bs,seq_len,enhncd

        
        self.lka_pred = nn.Sequential(nn.Linear(1+seq_len+compress_hs,pred_hs),
                                            nn.ReLU(),
                                            nn.Dropout(dp),
                                            nn.Linear(pred_hs,pred_hs),
                                            nn.ReLU(),
                                            nn.Dropout(dp),
                                            nn.Linear(pred_hs,1),
                                            nn.Sigmoid())
        
        self.rld_cmp = nn.Sequential(nn.Linear(seq_len+compress_hs,compress_hs),
                                            nn.PReLU())
        
        if enhncd:
            self.rld_rcn = nn.Sequential(nn.Linear(compress_hs,rcn_w))
        
        self.drop = nn.Dropout(dp)
        
        #### LKA - Training - RLD Variables ###
        self.lka,self.max_count,self.rcn_w  = None , 0,rcn_w
        self.register_buffer('p_lka',nn.Parameter(torch.zeros(self.bs,1).type(torch.float)))
        self.register_buffer('rld',nn.Parameter(torch.tensor([0.5]*compress_hs).repeat(self.bs).view(self.bs,-1)))
        #### Storage Elements####
        self.RLD_hist = []
        self.LKA_hist = []
    def forward(self,x,lka):
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        x=x*4
        #########RLD COMPRESSION##########
        self.rld = self.rld_cmp(torch.cat([x.squeeze(dim=-1),self.rld],dim=-1))
        ########END OF RLD COMPRESSION####

        #Get the history of vis from previous vis and all the past frames
        x_drop = self.drop(x).squeeze(dim=-1)
        cat = torch.cat([self.p_lka,x_drop,self.rld],dim=-1)
        
        predicted_lka = self.lka_pred(cat).squeeze()
        
        #### Updates for LKA - RLD history - Minibatch counter ####
        self.rld = self.rld.detach()
        self.p_lka = lka.view(-1,1).detach().to(device)
        self.RLD_hist.extend(x.view(-1).detach().cpu().numpy())
        self.LKA_hist.extend(lka.view(-1).repeat(self.seq_len).detach().cpu().numpy())
        #### Reconstruction ####
        if self.enhncd:
            if len(self.RLD_hist) >= self.rcn_w*self.bs:
            
                rcn_rld = self.rld_rcn(self.rld)
                tgt_rld = torch.tensor(self.RLD_hist[-self.rcn_w*self.bs:]).view(self.bs,-1).clone().to(device)
            else:
                
                rcn_rld = self.rld_rcn(self.rld)[:,0:self.seq_len]
                tgt_rld = torch.tensor(self.RLD_hist).view(self.bs,self.seq_len).clone().to(device)
            return self.p_lka.squeeze(dim=-1),predicted_lka,(rcn_rld,tgt_rld)
        else:
            return self.p_lka,predicted_lka
