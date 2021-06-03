from trainer import pix2pixTrainer
import matplotlib.pyplot as plt
from option import define_option
from datasets import create_dataloader,GridMask
import utils
import torchvision as tv
import numpy as np
import cv2
import torch as th
import glob
opt = define_option()
trainer = pix2pixTrainer(opt)

full_f = glob.glob("./paint/full/*")
ref_f = glob.glob("./paint/ref/*")
for i in range(len(full_f)):
    for j in range(len(ref_f)):
        sketch_img = utils.read_img(full_f[i], sketch = True)
        ref_img = utils.read_img(ref_f[j])
        #sketch_img = utils.np2tensor(sketch_img,False)
        gen_img = trainer.generate_img([ref_img,sketch_img,ref_img]).detach().cpu()
        utils.plot_mycolorhint_generated_image(gen_img,sketch_img,ref_img,opt.model_name,str(i)+str(j))

