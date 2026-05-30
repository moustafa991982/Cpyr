from basemodel import *


class simple(BaseModel):
    def __init__(self, name = 'simple_sigmoid'):
        super().__init__(name)
        self.loss = []
        self.conv1 = nn.Conv2d(1,1,(8,14))
        self.pool =  nn.MaxPool2d((30,30),stride=1)
        
        self.deconv1 =nn.ConvTranspose2d(1,1,(30,30))
        self.deconv2 = nn.ConvTranspose2d(1,1,(8,14))
        
    def forward(self,inputs):
        
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
                
        
        k = f.elu(self.conv1(inputs))
        k = self.pool(k)
        k = f.elu(self.deconv1(k))
        return torch.sigmoid(self.deconv2(k))[:,0,:,:]
    def record(self,record):
        self.loss.append(record)
# Four Layer       
class four_layer(BaseModel):
    def __init__(self,name = 'FL_sigmoid'):
        super().__init__(name)
        self.loss = []
        self.conv1 = nn.Conv2d(1,1,(32,32))
        self.pool =  nn.MaxPool2d((16,16),stride=1)
        self.conv2 = nn.Conv2d(1,1,(8,8))
        self.pool2 =  nn.AvgPool2d((8,8),stride=1)
        
        
        self.deconv1 =nn.ConvTranspose2d(1,1,(32,32))
        self.deconv2 = nn.ConvTranspose2d(1,1,(16,16))
        self.deconv3 =nn.ConvTranspose2d(1,1,(8,8))
        self.deconv4 = nn.ConvTranspose2d(1,1,(8,8))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        k = f.elu(self.conv1(inputs))
        k = self.pool(k)
        k = f.elu(self.conv2(k))
        k = self.pool2(k)
        
        k = f.elu(self.deconv1(k))
        k = f.elu(self.deconv2(k))
        k = f.elu(self.deconv3(k))
        k = torch.sigmoid(self.deconv4(k))
        
        return k[:,0,:,:]
    def record(self,record):
        self.loss.append(record)
        

#

class fl_norm(BaseModel):
    def __init__(self,name = 'FL_norm_sigmoid'):
        super().__init__(name)
        self.loss = []
        self.conv1 = nn.Conv2d(1,1,(32,32))
        self.pool =  nn.MaxPool2d((16,16),stride=1)
        self.conv2 = nn.Conv2d(1,1,(8,8))
        self.pool2 =  nn.AvgPool2d((8,8),stride=1)
        
        
        self.deconv1 =nn.ConvTranspose2d(1,1,(32,32))
        self.deconv2 = nn.ConvTranspose2d(1,1,(16,16))
        self.deconv3 =nn.ConvTranspose2d(1,1,(8,8))
        self.deconv4 = nn.ConvTranspose2d(1,1,(8,8))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        k = f.elu(self.conv1(inputs))
        k = f.normalize(self.pool(k))
        k = f.elu(self.conv2(k))
        k = self.pool2(k)
        
        k = f.elu(self.deconv1(k))
        k = f.elu(self.deconv2(k))
        k = f.elu(self.deconv3(k))
        k = torch.sigmoid(self.deconv4(k))
        
        return k[:,0,:,:]
    def record(self,record):
        self.loss.append(record)
        
#

class fl_max_norm(BaseModel):
    def __init__(self,name = 'FL_Max_Norm_sigmoid'):
        super().__init__(name)
        self.loss = []
        self.conv1 = nn.Conv2d(1,1,(32,32))
        self.pool =  nn.MaxPool2d((16,16),stride=1)
        self.conv2 = nn.Conv2d(1,1,(8,8))
        self.pool2 =  nn.AvgPool2d((8,8),stride=1)
        
        
        self.deconv1 =nn.ConvTranspose2d(1,1,(32,32))
        self.deconv2 = nn.ConvTranspose2d(1,1,(16,16))
        self.deconv3 =nn.ConvTranspose2d(1,1,(8,8))
        self.deconv4 = nn.ConvTranspose2d(1,1,(8,8))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        
        k = self.pool(inputs)
        k = f.normalize(f.elu(self.conv1(k)))
        k = f.elu(self.conv2(k))
        k = self.pool2(k)
        
        k = f.elu(self.deconv1(k))
        k = f.elu(self.deconv2(k))
        k = f.elu(self.deconv3(k))
        k = torch.sigmoid(self.deconv4(k))
        
        return k[:,0,:,:]
    def record(self,record):
        self.loss.append(record)
        
#

class fl_max(BaseModel):
    def __init__(self,name = 'FL_Max_sigmoid'):
        super().__init__(name)
        self.loss = []
        self.conv1 = nn.Conv2d(1,1,(32,32))
        self.pool =  nn.MaxPool2d((16,16),stride=1)
        self.conv2 = nn.Conv2d(1,1,(8,8))
        self.pool2 =  nn.AvgPool2d((8,8),stride=1)
        
        
        self.deconv1 =nn.ConvTranspose2d(1,1,(32,32))
        self.deconv2 = nn.ConvTranspose2d(1,1,(16,16))
        self.deconv3 =nn.ConvTranspose2d(1,1,(8,8))
        self.deconv4 = nn.ConvTranspose2d(1,1,(8,8))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        
        k = self.pool(inputs)
        k = f.elu(self.conv1(k))
        k = f.elu(self.conv2(k))
        k = self.pool2(k)
        
        k = f.elu(self.deconv1(k))
        k = f.elu(self.deconv2(k))
        k = f.elu(self.deconv3(k))
        k = torch.sigmoid(self.deconv4(k))
        
        return k[:,0,:,:]
    def record(self,record):
        self.loss.append(record)
        
#

class combined(BaseModel):
    def __init__(self,name = 'combined_sigmoid'):
        super().__init__(name)
        self.loss = []
        self.fl_norm = fl_norm()
        self.fl_max = fl_max()
        
        self.conv = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(1,1))
        
    def forward(self,inputs):
        
        inputs = inputs.type(next(iter(self.parameters())).type())
        k1 = self.fl_norm(inputs)[:,None,:,:]
        k2 = self.fl_max(inputs)[:,None,:,:]
        k3 = torch.cat([k1,k2],dim=1)
        
        k = torch.sigmoid(self.conv(k3))
        return k[:,0,:,:]
    def record(self,record):
        self.loss.append(record)
        

        
#

class all_combined(BaseModel):
    def __init__(self,name = 'all_in_all_sigmoid'):
        super().__init__(name)
        self.loss = []
        self.fl_norm = fl_norm()
        self.fl_max = fl_max()
        self.simple = simple()
        self.fl_max_norm = fl_max_norm()
        self.four_layer = four_layer()
        self.conv = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=(1,1))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())
        k1 = self.fl_norm(inputs)[:,None,:,:]
        k2 = self.fl_max(inputs)[:,None,:,:]
        k3 = self.simple(inputs)[:,None,:,:]
        k4 = self.fl_max_norm(inputs)[:,None,:,:]
        k5 = self.four_layer(inputs)[:,None,:,:]
        
        
        k6 = torch.cat([k1,k2,k3,k4,k5],dim=1)
        k = torch.sigmoid(self.conv(k6))
        return k[:,0,:,:]
    def record(self,record):
        self.loss.append(record)
        
#

class Norm_FL(BaseModel):
    def __init__(self,name = 'Norm_FL'):
        super().__init__(name)
        self.loss = []
        self.conv1 = nn.Conv2d(1,1,(32,32))
        self.pool =  nn.MaxPool2d((16,16),stride=1)
        self.conv2 = nn.Conv2d(1,1,(8,8))
        self.pool2 =  nn.AvgPool2d((8,8),stride=1)
        
        
        self.deconv1 =nn.ConvTranspose2d(1,1,(32,32))
        self.deconv2 = nn.ConvTranspose2d(1,1,(16,16))
        self.deconv3 =nn.ConvTranspose2d(1,1,(8,8))
        self.deconv4 = nn.ConvTranspose2d(1,1,(8,8))
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        k = f.elu(self.conv1(f.normalize(inputs)))
        k = self.pool(k)
        k = f.elu(self.conv2(k))
        k = self.pool2(k)
        
        k = f.elu(self.deconv1(k))
        k = f.elu(self.deconv2(k))
        k = f.elu(self.deconv3(k))
        k = torch.sigmoid(self.deconv4(k))
        
        return k[:,0,:,:]
    def record(self,record):
        self.loss.append(record)