from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm
def get_norm_layer(in_ch,norm_type="instance",obj=None):

    if norm_type == "batch":
        norm_layer = nn.BatchNorm2d(in_ch, affine=False)
    elif norm_type == "instance":
        norm_layer = nn.InstanceNorm2d(in_ch, affine=False)
    else:
        raise ValueError("normalization layer error")
    return norm_layer