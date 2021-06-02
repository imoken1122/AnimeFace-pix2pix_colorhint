import os
import cv2
from torchvision import transforms
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision as tv
import json


def create_dir(dirlist):
    """
    Create directory

    args: dirs (str or list) create all dirs in 'dirs'
    """
    for dirs in dirlist:
        if isinstance(dirs, (list, tuple)):
            for d in dirs:
                if not os.path.exists(os.path.expanduser(d)):
                    os.makedirs(d)
        elif isinstance(dirs, str):
            if not os.path.exists(os.path.expanduser(dirs)):
                os.makedirs(dirs)


def setup_logging(model_list, model_name='./log'):
    
    # Output path where we store experiment log and weights
    model_dir = [os.path.join(model_name, 'models', mn) for mn in model_list]

    fig_dir = os.path.join(model_name, 'figures')
    
    # Create if it does not exist
    create_dir([model_dir, fig_dir])

def save_model(G,D, epoch, opt):
    model_name = opt.model_name
    th.save(G.state_dict(),f"{model_name}/models/net_G/G_{epoch}.pth")
    th.save(D.state_dict(),f"{model_name}/models/net_D/D_{epoch}.pth")

def load_model(net,name,epoch,opt,latest = False):
    if latest:
        file = sorted(glob.glob(f"{opt.model_name}/models/net_{name}/*") ,reverse=True)[0] 
    else:
        file = glob.glob(f"{opt.model_name}/models/net_{name}/{name}_{epoch}.pth")[0]
    if opt.cuda:
        print("GPU loading weight")
        net.load_state_dict(th.load(file))
    else: 
        print("CPU loading weight")
        net.load_state_dict(th.load(file, th.device('cpu')))

    return net


def load_latest_model(net,name,opt):
    file = sorted(glob.glob(f"{opt.model_name}/models/net_{name}/*") ,reverse=True)[0]
    net.load_state_dict(th.load(file))
    return net





def progress_state(epoch=None,iter=None,mode="r",model_name="./log",opt=None):
    dic = {"epoch":"","iter":"","opt":opt}
    if mode == "w":
        with open(f"{model_name}/setup.json",mode) as f:
            dic["epoch"] = str(epoch)
            dic["iter"] = str(iter)
            dic["opt"] = str(opt)
            json.dump(dic,f)
    else:
        f = open(f"{model_name}/setup.json",mode) 
        dic = json.load(f)
        return dic

def plot_generated_image(img_full,img_sketch,img_gen,epoch,suffix, model_name):
    grid1 = tv.utils.make_grid(img_full[:])
    grid2 = tv.utils.make_grid(img_sketch[:])
    grid3 = tv.utils.make_grid(img_gen[:])
    grid = th.cat([grid1,grid2,grid3],dim=1)
    grid = np.transpose((grid/2 + 0.5),[1,2,0])
    plt.axis("off")
    plt.imshow(grid)
    plt.savefig(f"{model_name}/figures/{suffix}_{epoch}.png")

def plot_selfpaint_generated_image(img_gen,img_sketch,img_ref,model_name,suffix):
    grid1 = tv.utils.make_grid(img_sketch)
    grid2 = tv.utils.make_grid(img_ref)
    grid3 = tv.utils.make_grid(img_gen)
    grid = th.cat([grid1,grid2,grid3],dim=2)
    grid = np.transpose((grid/2 + 0.5),[1,2,0])
    plt.axis("off")
    plt.imshow(grid)
    plt.savefig(f"./paint/result/self_paint_ref_output_{suffix}.png") 


def make_contour_image(gray):
        neiborhood24 = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1]],
                                np.uint8)
        dilated = cv2.dilate(gray, neiborhood24, iterations=1)
        diff = cv2.absdiff(dilated, gray)
        contour = 255 - diff

        return contour

def np2tensor(img,real = True):

    if real:
        tf_full = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
                    ])
        return tf_full(img).unsqueeze(0)
    else:
        tf_sketch = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,)) 

            ])
        return tf_sketch(img).unsqueeze(0)

def read_img(path,sketch = False, ref=False):
    if ref:
        img = cv2.resize(cv2.imread(path,-1 ),(256,256))
        idx = np.where(img[:,:,3]==0)
        img[idx] = [255]*4
    else:
        img = cv2.resize(cv2.imread(path ,),(256,256))

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if sketch:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = make_contour_image(gray)

    return img



import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_tensor(img):
    array = np.asarray(img, np.float32) / 255.0 # [0, 1]
    array = np.expand_dims(array, axis=0)
    array = np.transpose(array, [0, 3, 1, 2]) # PyTorchはNCHW
    return torch.as_tensor(array)

def show_tensor(input_image_tensor):
    img = input_image_tensor.numpy() * 255.0
    img = img.astype(np.uint8)[0,0,:,:]    
    plt.imshow(img, cmap="gray")
    plt.show()

def linedraw(img):
    # データの読み込み
    x = load_tensor(img)
    # Y = 0.299R + 0.587G + 0.114B　でグレースケール化
    gray_kernel = torch.as_tensor(
        np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1))
    x = F.conv2d(x, gray_kernel) # 行列積は畳み込み関数でOK
    # 3x3カーネルで膨張1回（膨張はMaxPoolと同じ）
    dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    # 膨張の前後でL1の差分を取る
    diff = torch.abs(x-dilated)    
    # ネガポジ反転
    x = 1.0 - diff
    # 結果表示
    tf = transforms.Compose([transforms.Normalize((0.5,),(0.5,))])
    x = tf(x[0])
    return x