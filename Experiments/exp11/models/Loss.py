import torch.nn as nn

class mse_bce_loss:
    def __init__(self,enhncd,lam=0.5,reduction='mean'):
        self.enhncd,self.lam = enhncd,lam
        self.reduction=reduction
    def __call__(self,ip,tgt):
        if self.enhncd:
            tgt_lka,op_lka,(rcn_rld,tgt_rld) = ip
            rcn_loss = nn.functional.mse_loss(rcn_rld,tgt_rld)
            lka_loss = nn.functional.mse_loss(op_lka,tgt_lka)
            return self.lam*lka_loss + (1-self.lam)*rcn_loss
        #### LKA PART ####
        tgt_lka,op_lka = ip        
        lka_loss = nn.functional.binary_cross_entropy(op_lka,tgt_lka.squeeze(),reduction=self.reduction)
        return (lka_loss)
