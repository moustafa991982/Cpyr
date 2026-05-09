from collections import OrderedDict
from basemodel import *
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        
        #att
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        
        #att
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
       
        return self.conv(dec1)

    def encode(self,x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        return [enc1,enc2,enc3,enc4,bottleneck]
    
    def decode(self,enc1,enc2,enc3,enc4,bottleneck):
        dec4 = self.upconv4(bottleneck)

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)


        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)

        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)

        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    
#

class UNet_with_att(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_with_att, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.sq1 = signal_query(512)
        self.sq2 = signal_query(256)
        self.sq3 = signal_query(128)
        self.sq4 = signal_query(64)
        
        self.attn4 = attn(256,256,256)
        self.attn3 = attn(128,128,128)
        self.attn2 = attn(64,64,64)
        self.attn1 = attn(32,32,32)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        #print('bottle ',bottleneck.shape ,enc4.shape)
        
        dec4 = self.upconv4(bottleneck)
        enc4 = self.attn4(self.sq1(bottleneck) , enc4)
        #print('first block',dec4.shape , enc4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.attn3(self.sq2(dec4) , enc3)
        #print('second block',dec3.shape , enc3.shape)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        enc2 = self.attn2(self.sq3(dec3) , enc2)
        #print('third block',dec2.shape , enc2.shape)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        #print('last block',dec1.shape , enc1.shape)
        enc1 = self.attn1(self.sq4(dec2) , enc1)
        #print('a')
        dec1 = torch.cat((dec1, enc1), dim=1)
        #print('a')
        dec1 = self.decoder1(dec1)
        #print('a')
        return self.conv(dec1)
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    
#
class signal_query(BaseModel):
    def __init__(self, in_ch ,name='signal query'):
        super().__init__(name)
        self.sq = nn.Sequential(nn.ConvTranspose2d(in_ch ,in_ch//2 , kernel_size=2 , stride = 2),
                                 nn.BatchNorm2d(in_ch//2))   
    def forward(self,x):
            return self.sq(x)
 
#
class attn(BaseModel):
    def __init__(self, CH_g , CH_enc , CH_desired,name='att'):
        super().__init__(name )
        
        self.w_g = nn.Sequential(nn.Conv2d(CH_g ,CH_desired , kernel_size=1),
                                 nn.BatchNorm2d(CH_desired))
        self.w_enc = nn.Sequential(nn.Conv2d(CH_enc,CH_desired , kernel_size=1),
                                   nn.BatchNorm2d(CH_desired))
        
        self.psi = nn.Sequential(
            nn.Conv2d(CH_desired,1,kernel_size=1) 
            ,nn.BatchNorm2d(1)
            ,nn.Sigmoid())
        
    def forward(self,g , enc):
            g = self.w_g(g)
            enc_ = self.w_enc(enc)
            psi = torch.add(g,enc_)
            psi = self.psi(psi)
            
            return psi*enc  
#
class Unet_sigmoid(BaseModel):
    def __init__(self,att = False,name = 'U-net_sigmoid'):
        super().__init__(name)
        self.loss = []
        self.one = []
        ##Load pre-trained
        if att:
                self.unet = UNet_with_att().cuda()
        else:
                self.unet = UNet().cuda()
                state_dict = torch.load('unet/brain-segmentation-pytorch-master/weights/unet.pt', map_location="cuda:0")
                self.unet.load_state_dict(state_dict)
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        k = torch.cat([inputs,inputs,inputs],dim = 1)
        k = self.unet(k)
        return torch.sigmoid(k)

    def encode(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        k = torch.cat([inputs,inputs,inputs],dim = 1)
        k = self.unet.encode(k)
        return k
    def decode(self,*decoded_ips):
        k = self.unet.decode(*decoded_ips)
        return torch.sigmoid(k[:,0,:,:])
    
    
    def record(self,record):
        self.loss.append(record)
        
    def one_to_one(self,record):
        self.one.append(record)
        
        
        
#

class Unet_softmax(BaseModel):
    def __init__(self,name='U-net_softmax'):
        super().__init__(name)
        self.loss = []
        self.one = []
        ##Load pre-trained
        self.deconv1 = nn.ConvTranspose2d(3,3,(65,65))
        self.deconv2 = nn.ConvTranspose2d(3,3,(65,65))
        self.deconv3 = nn.ConvTranspose2d(3,3,(65,65))
        
        self.upSampler = nn.ModuleList([self.deconv1 , self.deconv2,self.deconv3])
        
        self.conv1 = nn.Conv2d(1,16,(65,65))
        self.conv2 = nn.Conv2d(16,16,(65,65))
        self.conv3 = nn.Conv2d(16,16,(65,65))
        
        self.downSampler = nn.ModuleList([self.conv1 , self.conv2,self.conv3])
        
        
        self.unet = UNet().cuda()
        state_dict = torch.load('unet/brain-segmentation-pytorch-master/weights/unet.pt', map_location="cuda:0")
        self.unet.load_state_dict(state_dict)
        
        self.baseModel = nn.ModuleList([self.unet])
        
    def forward(self,inputs):
        inputs = inputs.type(next(iter(self.parameters())).type())[:,None,:,:]
        k = torch.cat([inputs,inputs,inputs],dim = 1)
        k = f.elu(self.deconv1(k))
        k = f.elu(self.deconv2(k))
        k = f.elu(self.deconv3(k))
        
        k = self.unet(k)
        
        k = f.elu(self.conv1(k))
        k = f.elu(self.conv2(k))
        k = f.softmax(self.conv3(k),dim=1)
        return k
    
    def record(self,record):
        self.loss.append(record)
        
    def one_to_one(self,record):
        self.one.append(record)
