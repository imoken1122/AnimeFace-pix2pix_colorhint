from torch import nn
import torch as th
from torch.nn import init
import functools
import networks 
import utils

import torch.nn.functional as F

class pix2pixModel(nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.netG, self.netD = self.initialize_model(opt)
        if opt.isTrain:
            device = "cuda" if opt.cuda else "cpu"
            self.criterionGAN = networks.loss.GANLoss(opt.gan_mode,opt=opt).to(device)
            self.criterionFeat = th.nn.L1Loss().to(device)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss().to(device)
        self.fake_img = None
        self.opt =opt


    def initialize_model(self,opt):
        netG,netD = networks.define_G(opt), networks.define_D(opt) if opt.isTrain else None
        if not opt.isTrain or opt.continue_train or opt.continue_train_latest:
            netG = utils.load_model(netG,"G",opt.which_epoch,opt, latest = True if opt.continue_train_latest else False)
            if opt.isTrain:
                netD = utils.load_model(netD,"D",opt.which_epoch,opt,latest = True if opt.continue_train_latest else False)

        return netG,netD
        
    def forward(self,data,mode):
        real_img,sketch_img,ref_img = data

        if self.opt.cuda :
            real_img,sketch_img,ref_img = real_img.to("cuda"),sketch_img.to("cuda"),ref_img.to("cuda")

        if mode == "generator": 
            g_loss,fake_img = self.calc_G_loss(real_img,sketch_img,ref_img)
            return g_loss,fake_img
        elif mode == "discriminator": 
            d_loss = self.calc_D_loss(real_img,sketch_img,ref_img)
            return d_loss
        elif mode == "inference":
            with th.no_grad():
               fake_img = self.generate_fake(ref_img,sketch_img)
            return fake_img
        else:
            raise ValueError("Not define mode")


    def create_optimizers(self, opt):
        opt_G = th.optim.Adam(self.netG.parameters(),lr=opt.lr,betas = (opt.beta1,opt.beta2))
        opt_D = th.optim.Adam(self.netD.parameters(),lr=opt.lr,betas = (opt.beta1,opt.beta2))
        return opt_G,opt_D

    def calc_G_loss(self,real_img,sketch_img,ref_img):
        
        G_losses = {}
        x = th.cat([sketch_img,ref_img],1)

        self.fake_img = self.netG(x)
        pred_fake,pred_real = self.excute_discriminate(self.fake_img,real_img,sketch_img,ref_img)
        G_losses["GAN"] = self.criterionGAN(pred_fake,True,for_disc=False)
        G_losses["L1"] = self.criterionFeat(self.fake_img,real_img) * 100.

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(self.fake_img, real_img) \
                * self.opt.lambda_vgg

        return G_losses, self.fake_img


    def calc_D_loss(self,real_img,sketch_img,ref_img):
        D_losses = {}
        with th.no_grad():
            fake_img = self.generate_fake(ref_img,sketch_img)
            fake_img = fake_img.detach()
            fake_img.requires_grad_()

        pred_fake,pred_real = self.excute_discriminate(fake_img,real_img,sketch_img,ref_img)

        D_losses["D_fake"] = self.criterionGAN(pred_fake,False)
        D_losses["D_real"] = self.criterionGAN(pred_real,True)
        return D_losses


    def excute_discriminate(self,fake_img,real_img,sketch_img,ref_img):
        fake_concat = th.cat([sketch_img,fake_img,ref_img],1)
        real_concat = th.cat([sketch_img,real_img,ref_img],1)

        fake_and_real=th.cat([fake_concat,real_concat],0)

        disc_out = self.netD(fake_and_real)
        pred_fake,pred_real = disc_out[:len(disc_out)//2],disc_out[len(disc_out)//2:]
        return pred_fake,pred_real

    def generate_fake(self,ref_img,sketch_img):
        x = th.cat([sketch_img,ref_img],1)
        return self.netG(x)

    def save(self,epoch):
        utils.save_model(self.netG,self.netD,epoch,self.opt)

