import numpy as np
import torch as th
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision as tv
from PIL import Image
import cv2
import utils
import glob
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch





class GridMask():
    def __init__(self, p=1, d_range=(70, 100), r=0.1):
        self.p = p
        self.d_range = d_range
        self.r = r
        
    def __call__(self, sample):
        """
        sample: torch.Tensor(3, height, width)
        """
        if np.random.uniform() > self.p:
            return sample

        side = sample.shape[1]
        d = np.random.randint(*self.d_range, dtype=np.uint8)
        r = int(self.r * d)
        
        mask = np.ones((side+d, side+d), dtype=np.uint8)
        for i in range(0, side+d, d):
            for j in range(0, side+d, d):
                mask[i: i+(d-r), j: j+(d-r)] = 0
        delta_x, delta_y = np.random.randint(0, d, size=2)
        mask = mask[delta_x: delta_x+side, delta_y: delta_y+side]
        sample *= np.expand_dims(mask, 2)
        sample[sample==0] = 255

        return sample
import random
class CircleMask():
    def __init__(self,p=0.85):
        self.p = p
        pass

    def __call__(self,sample):
        

        n = np.random.randint(70, 85,)
        sample = cv2.rectangle(sample, (0, 0), (255, 255), (255, 255, 255), thickness=30, lineType=cv2.LINE_4)

        for i in range(n):
            s = np.random.randint(23, 33,)
            x, y = np.random.randint(5, 255, size=2)
            u = np.random.choice([15,-1],p=[0.4,0.6])
            sample= cv2.circle(sample, (x,y), int(s), (255, 255, 255), thickness=u)
        #return th.from_numpy(sample)
        return sample
        



class Dataset(data.Dataset):
    def __init__(self, dir_path ,opt=None):
        super().__init__()

        self.img_path = glob.glob(f"{dir_path}/*")
        self.tf_full = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
            ])
        self.tf_sketch = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)) 
        ])

        self.len = len(self.img_path)
        self.gridmask = GridMask()
        self.noise = CircleMask()
        self.opt = opt


    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        path = self.img_path[idx]

        img1 = cv2.resize(cv2.imread(path ),(256,256))
        img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

        p = np.random.choice([0,1],1,[0.6,0.4])
        if p == 0:
            img_sketch = utils.linedraw(img2)
        else:
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img_sketch = utils.make_contour_image(gray)
            img_sketch = self.tf_sketch(img_sketch)

        img = self.tf_full(img2)
        img_ref = th.FloatTensor([])





        for n in self.opt.aug:
            if n == "grid":
                img2 = self.gridmask(img2)
            if n == "circle":
                img2 = self.noise(img2)
        img_ref = self.tf_full(img2)
            
        return img,img_sketch,img_ref


def create_train_test_dataloader(opt):

    train_dataloader = DataLoader(Dataset(f"{opt.input_path}/train",opt), batch_size=opt.batch_size, shuffle=True,drop_last=True,num_workers=2)
    test_dataloader = DataLoader(Dataset(f"{opt.input_path}/test",opt), batch_size=opt.batch_size, shuffle=True,drop_last=True,num_workers=2)
    return train_dataloader,test_dataloader
def create_dataloader(opt):
    dataloader = DataLoader(Dataset(f"{opt.input_path}",opt), batch_size=opt.batch_size, shuffle=True,drop_last=True,num_workers=2)

    return dataloader
