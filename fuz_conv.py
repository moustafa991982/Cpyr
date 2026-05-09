import torch 
import torch.nn as nn

class conv_enocder(BaseModel):    
    def __init__(self):
        super().__init__()

        self.encode = nn.Conv2d(seq_len,1,(1,1))
        self.decode = nn.ConvTranspose2d(1,seq_len,(1,1))
        self.relu = nn.PReLU()
        
    def forward(self,x):
        bs = x.size(0)
        ts = x.size(1)
        encode = self.encode(x)
        decode = self.decode(encode.view(bs,1,112,112)) 
        
        
             
        img = torch.sigmoid(decode)
        return img#torch.sigmoid(op)

