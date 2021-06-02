import torch as th
from torch import nn
from torchsummary import summary
from networks.architecture import ResnetBlock,SLEBlock
from networks.norm import get_norm_layer
class Generator(nn.Module):
    def __init__(self,opt):
        super().__init__()
        gen_in_ch = opt.gen_in_ch
        self.device = "cuda" if opt.cuda else "cpu"
        self.opt = opt
        self.enc1 = self.conv_block(gen_in_ch,32,5)                  # [bs, 32, 256, 256]
        self.enc2 =  self.conv_block(32,64,3,pool_kernel = 2) #[bs, 64, 64, 64]
        self.enc3 = self.conv_block(64,128,3,pool_kernel =2) # [bs, 128, 32, 32]
        self.enc4 = self.conv_block(128,256,3,pool_kernel =2) #[bs, 256, 16, 16]
        self.enc5 = self.conv_block(256,512,3,pool_kernel =2) #[ 8, 512, 8, 8]
        self.enc6 = self.conv_block(512,1024,3,pool_kernel =2) #[ 8, 512, 8, 8]

        self.resnet = ResnetBlock(1024,nn.BatchNorm2d)
    
        self.dec = self.conv_block(1024,512,3,pool_kernel =-2) # [bs, 256, 16, 16]
        self.dec1 = self.conv_block(512+512,256,3,pool_kernel =-2) # [bs, 256, 16, 16]
        self.dec2 = self.conv_block(256+256,128,3,pool_kernel =-2) #[bs, 128, 32, 32]
        self.dec3 = self.conv_block(128+128,64,3,pool_kernel =-2) #([bs, 64, 64, 64]
        self.dec4 = self.conv_block(64+64,32,3,pool_kernel =-2) #[bs, 32, 256, 256])
        add = 3 if self.opt.color_sampler_model else 0 
        self.conv1 = nn.Conv2d(32+32+add,3,5,padding=2 )
        self.output = nn.Tanh() # [bs, 3, 256, 256]


        self.z_fc = nn.Linear(opt.z_dim,16384)
        self.z_dec = self.conv_block(1024,1024,pool_kernel=-2)
        self.z_dec1= self.conv_block(1024,512,pool_kernel=-2)
        self.z_dec2= self.conv_block(512,256,pool_kernel=-2)
        self.z_dec3= self.conv_block(256,128,pool_kernel=-2)
        self.z_dec4= self.conv_block(128,64,pool_kernel=-2)
        self.z_dec5= self.conv_block(64,32,pool_kernel=-2)
        self.z_out = nn.Conv2d(32,3,3,padding=1)

    def conv_block(self,in_ch,out_ch,kernel_size=3, pool_kernel=None):
        layers = []
        if pool_kernel is not None:
            if  pool_kernel>0:
                layers.append(nn.AvgPool2d(pool_kernel)) 
            else:
                layers.append(nn.UpsamplingNearest2d(scale_factor=-pool_kernel))

        layers.append(nn.Conv2d(in_ch,out_ch,kernel_size, padding = (kernel_size-1)//2))
        layers.append(get_norm_layer(out_ch,self.opt.gen_norm_type))
       # layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        return nn.Sequential(*layers)

    def forward(self,x):
        if self.opt.color_sampler_model:
            z = th.randn((x.size(0),self.opt.z_dim)).to(self.device)
            z = self.z_fc(z)

            z = z.view(-1, 1024, 4, 4)
            z = self.z_out(self.z_dec5(self.z_dec4(self.z_dec3(self.z_dec2(self.z_dec1(self.z_dec(z)))))))
            x =th.cat([x,z],1)

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)

        inner = self.resnet(x6)
        y = self.dec(inner)
        y1 = self.dec1(th.cat([y,x5],1))
        y2 = self.dec2(th.cat([y1,x4],dim = 1))
        y3 = self.dec3(th.cat([y2,x3],dim = 1))
        y4 = self.dec4(th.cat([y3,x2],dim = 1))
        
        yy = [y4,x1,z] if self.opt.color_sampler_model else [y4,x1]
        out = self.conv1(th.cat(yy,dim =1))

        output = self.output(out)
        return output


class SLE_UNet_Generator(nn.Module):
    def __init__(self,opt=None):
        super().__init__()
        in_ch =opt.gen_in_ch 
        ndf = 32
        self.fc = nn.Conv2d(in_ch,ndf,3,padding=1) # 32, 256
        self.e1 = self.enc_block(ndf,ndf*2) # 64,128
        self.e2 = self.enc_block(ndf*2,ndf*4) # 128,64
        self.e3 = self.enc_block(ndf*4,ndf*8) # 256,32,
        self.e4 = self.enc_block(ndf*8,ndf*16) # 512,16,
        self.e5 = self.enc_block(ndf*16,ndf*32) # 1024,8,

        self.resnet = ResnetBlock(1024,nn.BatchNorm2d) # 1024.8

        self.d = self.dec_block(ndf*32,) #512,16
        self.d1 = self.dec_block(ndf*16) # 256,32
        self.d2 = self.dec_block(ndf*8*2, concat=True)#128,64
        self.d3 = self.dec_block(ndf*4*2 , concat=True) # 64,128
        self.d4 = self.dec_block(ndf*2*2,concat=True) # 32,256

        self.output = nn.Sequential(nn.Conv2d(ndf,3,3,padding=1),nn.Tanh())

        self.sle1 = SLEBlock(64,1024) # x1 x5
        self.sle2 = SLEBlock(128,1024) # x2 x6
        self.sle3 = SLEBlock(256,512) # x3 y1
        #self.sle1 = SLEBlock()
        #self.sle2 = SLEBlock()
        #self.sle3 = SLEBlock()

    def enc_block(self,in_ch,out_ch,kernel_size=3,pool=True):
        layers = []
        pad = (kernel_size-1)//2
        layers.append(nn.Conv2d(in_ch,out_ch,kernel_size,padding=pad))
        layers.append(get_norm_layer(out_ch,"batch"))
        layers.append(nn.LeakyReLU(0.2,False))
        if pool:
            layers.append(nn.AvgPool2d(2)) 
        return nn.Sequential(*layers)

    def dec_block(self,in_ch,kernel_size=3,concat=False):
        layers = []
        pad = (kernel_size-1)//2
        out_ch = in_ch//2 if concat else in_ch
        layers.append(nn.UpsamplingNearest2d(scale_factor=2))
        layers.append(nn.Conv2d(in_ch,out_ch,kernel_size,padding=pad))
        layers.append(get_norm_layer(out_ch,"batch"))
        layers.append(nn.GLU(dim=1)) # dim/2
        return nn.Sequential(*layers)



    def forward(self,x):
        x = self.fc(x)
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)

        s1 = self.sle1(x1,x5)
        x6 = self.resnet(x5)


        s2 = self.sle2(x2,x6)

        y1 = self.d(x6)
        s3 =self.sle3(x3,y1)

        y2 = self.d1(y1)

        y3 = self.d2(th.cat([y2,s3],1))

        y4 = self.d3(th.cat([y3,s2],1))
        y5 = self.d4(th.cat([y4,s1],1))

        output = self.output(y5)
        return output


#print(summary(SLE_UNet_Generator(), (1,256,256)))


"""
class SLE_Generator(nn.Module):
    def __init__(self,opt=None):
        in_ch = 1
        ndf = 32
        self.fc = nn.Conv2d(in_ch,ndf,3,padding=1) # 32, 256
        self.e1 = self.enc_block(ndf,ndf*2) # 64,128
        self.e2 = self.enc_block(ndf*2,ndf*4) # 128,64
        self.e3 = self.enc_block(ndf*4,ndf*8) # 256,32,
        self.e4 = self.enc_block(ndf*8,ndf*16) # 512,16,
        self.e5 = self.enc_block(ndf*32,ndf*32) # 1024,8,

        
        self.d = self.dec_block(ndf*32) #512,16
        self.d1 = self.dec_block(ndf*16) # 256,32
        self.d2 = self.dec_block(ndf*8)#128,64
        self.d3 = self.dec_block(ndf*4) # 64,128
        self.d4 = self.dec_block(ndf*2) # 32,256

        self.output = nn.Sequential(nn.Conv2d(ndf*2,3,3,padding=1),nn.Tanh())

        #self.sle1 = SLEBlock()
        #self.sle2 = SLEBlock()
        #self.sle3 = SLEBlock()

    def enc_block(self,in_ch,out_ch,kernel_size=3,pool=True):
        layers = []
        pad = (kernel_size-1)//2
        layers.append(nn.Conv2d(in_ch,out_ch,kernel_size,padding=pad))
        layers.append(get_norm_layer(out_ch,"batch"))
        layers.append(nn.LeakyReLU(0.2,False))
        if pool:
            layers.append(nn.AvgPool2d(2)) 

    def dec_block(self,in_ch,kernel_size=3,first=False):
        layers = []
        pad = (kernel_size-1)//2
        if first:
            layers.append(nn.ConvTranspose2d(in_ch,in_ch,3,padding=1))
        else:
            layers.append(nn.UpsamplingNearest2d(scale_factor=2))
            layers.append(nn.Conv2d(in_ch,in_ch,kernel_size,padding=pad))
        layers.append(get_norm_layer(in_ch,"batch"))
        layers.append(nn.GLU(dim=1)) # dim/2



    def forward(self,x):
        x = self.fc(x)
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x3)

        s1 = SLEBlock(x5.size(2),x1.size(2))(x5,x1)

        y1 = self.d(s1)
        s2 = SLEBlock(y1.size(2),x2.size(2))(y1,x2)

        y2 = self.d1(s2)
        s3 = SLEBlock(y2.size(2),x3.size(2))(y2,x3)

        y3 = self.d2(s3)
        s4 = SLEBlock(y3.size(2),x3.size(2))(y2,x3)


"""