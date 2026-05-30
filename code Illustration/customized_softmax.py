from basemodel import *


class simple(BaseModel):
    def __init__(self,name = 'simple_softmax'):
        super().__init__(name)
        self.loss = []
        self.conv1 = nn.Conv2d(1,1,(8,14))
        self.pool =  nn.MaxPool2d((30,30),stride=1)
        
        self.deconv1 =nn.ConvTranspose2d(1,1,(30,30))
        self.deconv2 = nn.ConvTranspose2d(1,16,(8,14))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        k = f.elu(self.conv1(inputs))
        k = self.pool(k)
        k = f.elu(self.deconv1(k))
        
        k = torch.softmax(self.deconv2(k),dim = 1)
        return k
    def record(self,record):
        self.loss.append(record)
        
#

class four_layer(BaseModel):
    def __init__(self,name = 'FL_softmax'):
        super().__init__(name)
        self.loss = []
        self.conv1 = nn.Conv2d(1,1,(32,32))
        self.pool =  nn.MaxPool2d((16,16),stride=1)
        self.conv2 = nn.Conv2d(1,1,(8,8))
        self.pool2 =  nn.AvgPool2d((8,8),stride=1)
        
        
        self.deconv1 =nn.ConvTranspose2d(1,1,(32,32))
        self.deconv2 = nn.ConvTranspose2d(1,1,(16,16))
        self.deconv3 =nn.ConvTranspose2d(1,1,(8,8))
        self.deconv4 = nn.ConvTranspose2d(1,16,(8,8))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        k = f.elu(self.conv1(inputs))
        k = self.pool(k)
        k = f.elu(self.conv2(k))
        k = self.pool2(k)
        
        k = f.elu(self.deconv1(k))
        k = f.elu(self.deconv2(k))
        k = f.elu(self.deconv3(k))
        k = torch.softmax(self.deconv4(k),dim = 1)
        
        return k
    def record(self,record):
        self.loss.append(record)
        
#

class fl_max_norm(BaseModel):
    def __init__(self,name = 'FL_Max_Norm_softmax'):
        super().__init__(name)
        self.loss = []
        self.conv1 = nn.Conv2d(1,1,(32,32))
        self.pool =  nn.MaxPool2d((16,16),stride=1)
        self.conv2 = nn.Conv2d(1,1,(8,8))
        self.pool2 =  nn.AvgPool2d((8,8),stride=1)
        
        
        self.deconv1 =nn.ConvTranspose2d(1,1,(32,32))
        self.deconv2 = nn.ConvTranspose2d(1,1,(16,16))
        self.deconv3 =nn.ConvTranspose2d(1,1,(8,8))
        self.deconv4 = nn.ConvTranspose2d(1,16,(8,8))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:] 
        k = self.pool(inputs)
        k = f.normalize(f.elu(self.conv1(k)))
        k = f.elu(self.conv2(k))
        k = self.pool2(k)
        
        k = f.elu(self.deconv1(k))
        k = f.elu(self.deconv2(k))
        k = f.elu(self.deconv3(k))
        k = torch.softmax(self.deconv4(k),dim = 1)
        
        return k
    def record(self,record):
        self.loss.append(record)
        
#

