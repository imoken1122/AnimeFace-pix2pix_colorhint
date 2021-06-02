import torch as th
from torch.nn.utils.spectral_norm import spectral_norm
from torch import nn
x = th.randn((1,32,125,125))
xx = nn.Conv2d(32,64,3,padding=1)
m = spectral_norm(xx)
print(m(x).shape)