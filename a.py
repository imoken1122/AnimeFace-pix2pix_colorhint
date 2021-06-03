import torch as th
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.spectral_norm import spectral_norm
from torch import nn
import cv2
import utils
from datasets import Noise
import torchvision as tv
def load():
    img = cv2.imread("./paint/full/0-70697637_p0_master1200.jpg")
    img1 = cv2.resize(img,(256,256))
    img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

    img_sketch = utils.linedraw(img2).unsqueeze(0)

    img = utils.np2tensor(img2)
    print(img2.shape)
    img2 = Noise()(img2)
    print(img2.shape)
    plt.imshow(img2)
    plt.show()
    img_ref = utils.np2tensor(img2)

    return img,img_sketch,img_ref

def show(img):
    img = img[0]
    #img = [x.unsqueeze(0) for x in img][:100]
    img = [img[i*3:(i+1)*3] for i in range(len(img)//3) ][:100]

    grid = tv.utils.make_grid(img,nrow=10)

    #grid = [tv.utils.make_grid(x) for x in grid]
    #grid = th.cat(grid,2)
    grid = np.transpose((grid/2 + 0.5),[1,2,0])
    plt.axis("off")
    plt.imshow(grid)
    plt.savefig(f"inner.png")

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1  = self.b(4,32)
        self.l2  = self.b(32,64)
        self.l3  = self.b(64,128)
        self.l4  = self.b(128,256)
        self.l5  = self.b(256,512)
        self.l6  = self.b(512,1024)

        self.d0 = self.d(1024,512)
        self.d1 = self.d(512,256)
        self.d2 = self.d(256,128)
        self.d3 = self.d(128,64)
        self.d4 = self.d(64,32)
        self.d5 = nn.Sequential(nn.Conv2d(32,3,3,padding=1), nn.Tanh())

        self.inner=None
    def b(self,inp,out):
        layer=[]
        layer.append(nn.Conv2d(inp, out,3,padding= 1))
        layer.append(nn.BatchNorm2d(out) )
        layer.append(nn.LeakyReLU(0.2))
        layer.append(nn.AvgPool2d(2))
        return nn.Sequential(*layer)

    def d(self,inp,out):
        layer=[]
        layer.append(nn.UpsamplingNearest2d(scale_factor=2))
        layer.append(nn.Conv2d(inp, out,3,padding= 1))
        layer.append(nn.BatchNorm2d(out) )
        layer.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layer)
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        self.inner=x

        x = self.d0(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        return x


img,img_sketch,img_ref = load()
input=th.cat([img_sketch,img_ref],1)
model = NN()

output = model(input).detach()
print(output.shape)
show(output)

