import torchvision.models as models
from basemodel import *


class res18_sigmoid(BaseModel):
    def __init__(self,name = 'resnet18_sigmoid'):
        super().__init__(name)
        self.loss = []
        self.pre = models.resnet18(pretrained=False)
        self.deconv1 = nn.ConvTranspose2d(1,1,(11,16))
        self.deconv2 = nn.ConvTranspose2d(1,1,(21,31))
        self.deconv3 = nn.ConvTranspose2d(1,1,(21,21))
        self.deconv4 = nn.ConvTranspose2d(1,1,(23,23))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        
        bs = inputs.size(0)

        inputs = torch.cat([inputs,inputs,inputs],dim=1)
        k = self.pre(inputs)
        k = k.view(bs,40,-1)
        
        k = k[:,None,:,:]
        k = f.elu(self.deconv1(k))
        k = f.elu(self.deconv2(k))
        k = f.elu(self.deconv3(k))
        
        return torch.sigmoid(self.deconv4(k))
    def record(self,record):
        self.loss.append(record)
        
        

class res18_softmax(BaseModel):
    def __init__(self,name = 'resnet18_softmax'):
        super().__init__(name)
        self.loss = []
        self.pre = models.resnet18(pretrained=False)
        self.deconv1 = nn.ConvTranspose2d(1,1,(11,16))
        self.deconv2 = nn.ConvTranspose2d(1,1,(21,31))
        self.deconv3 = nn.ConvTranspose2d(1,1,(21,21))
        self.deconv4 = nn.ConvTranspose2d(1,16,(23,23))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        
        bs = inputs.size(0)

        inputs = torch.cat([inputs,inputs,inputs],dim=1)
        
        k = self.pre(inputs)
        k = k.view(bs,40,-1)
        
        k = k[:,None,:,:]
        k = f.elu(self.deconv1(k))
        
        k = f.elu(self.deconv2(k))
        k = f.elu(self.deconv3(k))
        
        
        return  torch.softmax(self.deconv4(k),dim=1)
    def record(self,record):
        self.loss.append(record)