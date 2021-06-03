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
if opt.self_paint_infer:
    full_f = glob.glob("./paint/full_train/*")
    ref_f = glob.glob("./paint/ref_train/*")
    for i in range(len(full_f)):
        for j in range(len(ref_f)):
            sketch_img = utils.read_img(full_f[i], sketch = True)
            ref_img = utils.read_img(ref_f[j],ref=True)
            ref_img = utils.np2tensor(ref_img)
            #sketch_img = utils.np2tensor(sketch_img,False)
            gen_img = trainer.generate_img([ref_img,sketch_img,ref_img]).detach().cpu()
            utils.plot_selfpaint_generated_image(gen_img,sketch_img,ref_img,opt.model_name,str(i)+str(j))


else:

    dataloader = create_dataloader(opt)
    img_list = next(iter(dataloader))
    gen_img = trainer.generate_img(img_list).detach().cpu()
    img,img_sk,ref_img= img_list

    utils.plot_generated_image(img,img_sk,gen_img,None,"test_generate",opt.model_name)

    grid = tv.utils.make_grid(ref_img[:])
    print(grid.shape)
    grid = np.transpose((grid/2 + 0.5),[1,2,0])
    plt.axis("off")
    plt.imshow(grid)
    plt.savefig(f"output2.png")

