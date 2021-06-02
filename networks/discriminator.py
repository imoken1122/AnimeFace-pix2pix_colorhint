import torch as th
from torch import nn
import numpy as np

from torchsummary import summary
from networks.norm import get_norm_layer

class Discriminator(nn.Module):
    def __init__(self,opt):
        super().__init__()
        ks = 3
        in_ch = opt.disc_in_ch
        self.opt = opt
        pd = (ks-1)//2
        ndf = in_ch * 2
        self.fc = nn.Conv2d(in_ch,32,ks,padding = pd)
        self.act = nn.LeakyReLU(0.2,True)

        self.cv2 = self.conv_block(32,64,ks,num = 1) 
        self.cv3 = self.conv_block(64,128,ks,pool_kernel=2)
        self.cv4 = self.conv_block(128,256,ks,pool_kernel=2)
        self.output = nn.Conv2d(256,1,kernel_size=1) 

    def conv_block(self,in_ch, out_ch, kernel_size=3, pool_kernel=None,num = 2):
        layers = []
        if self.opt.disc_norm_type == "spectral" : out_ch = in_ch
        for i in range(num):
            if i==0 and pool_kernel is not None:
                layers.append(nn.AvgPool2d(pool_kernel))
            layers.append(nn.Conv2d(in_ch if i ==0 else out_ch ,out_ch, kernel_size,padding= (kernel_size - 1) // 2))
            layers.append(get_norm_layer(out_ch,self.opt.disc_norm_type,obj = layers[-1]))
            layers.append(nn.LeakyReLU(0.2,True))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.act(self.fc(x))
        out = self.cv4(self.cv3(self.cv2(x)))

        return self.output(out)
#summary(Discriminator(),(3,256,256))