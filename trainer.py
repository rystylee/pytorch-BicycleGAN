import os
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter

from model import weights_init, Discriminator, Generator, Encoder
from util import sample_z, denormalize


def set_requires_grad(models, requires_grad):
    if not isinstance(models, list):
        models = [models]
    for model in models:
        if model is not None:
            for param in models.parameters():
                param.requires_grad = requires_grad


# GAN loss
def mse_loss(x, target):
    if target == 1:
        label = torch.ones(x.size()).to(x.device)
    elif target == 0:
        label = torch.zeros(x.size()).to(x.device)
    else:
        raise NotImplementedError('[!] The target {] is not found.'.format(target))

    return F.mse_loss(x, label)


def l1_loss(x, target):
    return F.l1_loss(x, target)


class Trainer(object):
    def __init__(self, args):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.batch_size = args.batch_size
        self.half_size = self.batch_size // 2
        assert self.batch_size % 2 == 0, '[!] batch_size is '
        self.nz = args.nz

        self.lambda_kl = args.lambda_kl
        self.lambda_img = args.lambda_img
        self.lambda_z = args.lambda_z

        # Discriminator for cVAE-GAN(encoded vector z)
        self.D_cVAE = Discriminator(args.input_nc + args.output_nc, args.ndf).to(self.device)
        self.D_cVAE.apply(weights_init)
        # print(self.D_cVAE)
        # Discriminator for cLR-GAN(random vector z)
        self.D_cLR = Discriminator(args.input_nc + args.output_nc, args.ndf).to(self.device)
        self.D_cLR.apply(weights_init)

        self.G = Generator(args.input_nc, args.output_nc, args.ngf, args.nz).to(self.device)
        self.G.apply(weights_init)

        self.E = Encoder(args.input_nc, args.nef, args.nz).to(self.device)
        self.E.apply(weights_init)
        # print(self.E)

        # Optimizers
        self.optim_D_cVAE = optim.Adam(self.D_cVAE.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.optim_D_cLR = optim.Adam(self.D_cLR.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.optim_E = optim.Adam(self.E.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        time_str = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter('{}/{}-{}'.format(args.log_dir, args.dataset_name, time_str))

    def __del__(self):
        self.writer.close()

    def all_zero_grad(self):
        self.optim_D_cVAE.zero_grad()
        self.optim_D_cLR.zero_grad()
        self.optim_G.zero_grad()
        self.optim_E.zero_grad()

    def save_weights(self, save_dir, global_step):
        d_cVAE_name = 'D_cVAE_{}.pth'.format(global_step)
        d_cLR_name = 'D_cLR_{}.pth'.format(global_step)
        g_name = 'G_{}.pth'.format(global_step)
        e_name = 'E_{}.pth'.format(global_step)

        torch.save(self.D_cVAE.state_dict(), os.path.join(save_dir, d_cVAE_name))
        torch.save(self.D_cLR.state_dict(), os.path.join(save_dir, d_cLR_name))
        torch.save(self.G.state_dict(), os.path.join(save_dir, g_name))
        torch.save(self.E.state_dict(), os.path.join(save_dir, e_name))

    def optimize(self, A, B, global_step):
        if A.size(0) <= 1:
            return

        A = A.to(self.device)
        B = B.to(self.device)

        cVAE_data = {'A': A[0:self.half_size], 'B': B[0:self.half_size]}
        cLR_data = {'A': A[self.half_size:], 'B': B[self.half_size:]}

        # Logging the input images
        log_imgs = torch.cat([cVAE_data['A'], cVAE_data['B']], 0)
        log_imgs = torchvision.utils.make_grid(log_imgs)
        log_imgs = denormalize(log_imgs)
        self.writer.add_image('cVAE_input', log_imgs, global_step)

        log_imgs = torch.cat([cLR_data['A'], cLR_data['B']], 0)
        log_imgs = torchvision.utils.make_grid(log_imgs)
        log_imgs = denormalize(log_imgs)
        self.writer.add_image('cLR_input', log_imgs, global_step)

        # ----------------------------------------------------------------
        # 1. Train D
        # ----------------------------------------------------------------

        # -----------------------------
        # Optimize D in cVAE-GAN
        # -----------------------------
        # Generate encoded latent vector
        mu, logvar = self.E(cVAE_data['B'])
        std = torch.exp(logvar / 2)
        random_z = sample_z(self.half_size, self.nz, 'gauss').to(self.device)
        encoded_z = (random_z * std) + mu

        # Generate fake image
        fake_img_cVAE = self.G(cVAE_data['A'], encoded_z)
        log_imgs = torchvision.utils.make_grid(fake_img_cVAE)
        log_imgs = denormalize(log_imgs)
        self.writer.add_image('cVAE_fake_encoded', log_imgs, global_step)

        real_pair_cVAE = torch.cat([cVAE_data['A'], cVAE_data['B']], dim=1)
        fake_pair_cVAE = torch.cat([cVAE_data['A'], fake_img_cVAE], dim=1)

        real_D_cVAE_1, real_D_cVAE_2 = self.D_cVAE(real_pair_cVAE)
        fake_D_cVAE_1, fake_D_cVAE_2 = self.D_cVAE(fake_pair_cVAE.detach())

        # The loss for small patch & big patch
        loss_D_cVAE_1 = mse_loss(real_D_cVAE_1, target=1) + mse_loss(fake_D_cVAE_1, target=0)
        loss_D_cVAE_2 = mse_loss(real_D_cVAE_2, target=1) + mse_loss(fake_D_cVAE_2, target=0)

        self.writer.add_scalar('loss/loss_D_cVAE_1', loss_D_cVAE_1.item(), global_step)
        self.writer.add_scalar('loss/loss_D_cVAE_2', loss_D_cVAE_2.item(), global_step)

        # -----------------------------
        # Optimize D in cLR-GAN
        # -----------------------------
        # Generate fake image
        fake_img_cLR = self.G(cLR_data['A'], random_z)
        log_imgs = torchvision.utils.make_grid(fake_img_cLR)
        log_imgs = denormalize(log_imgs)
        self.writer.add_image('cLR_fake_random', log_imgs, global_step)

        real_pair_cLR = torch.cat([cLR_data['A'], cLR_data['B']], dim=1)
        fake_pair_cLR = torch.cat([cVAE_data['A'], fake_img_cLR], dim=1)

        real_D_cLR_1, real_D_cLR_2 = self.D_cLR(real_pair_cLR)
        fake_D_cLR_1, fake_D_cLR_2 = self.D_cLR(fake_pair_cLR.detach())

        # Loss for small patch & big patch
        loss_D_cLR_1 = mse_loss(real_D_cLR_1, target=1) + mse_loss(fake_D_cLR_1, target=0)
        loss_D_cLR_2 = mse_loss(real_D_cLR_2, target=1) + mse_loss(fake_D_cLR_2, target=0)

        self.writer.add_scalar('loss/loss_D_cVAE_1', loss_D_cVAE_1.item(), global_step)
        self.writer.add_scalar('loss/loss_D_cVAE_2', loss_D_cVAE_2.item(), global_step)

        loss_D = loss_D_cVAE_1 + loss_D_cVAE_2 + loss_D_cLR_1 + loss_D_cLR_2
        self.writer.add_scalar('loss/loss_D', loss_D.item(), global_step)

        # -----------------------------
        # Update D
        # -----------------------------
        # set_requires_grad([], False)
        self.all_zero_grad()
        loss_D.backward()
        self.optim_D_cVAE.step()
        self.optim_D_cLR.step()

        # ----------------------------------------------------------------
        # 2. Train G & E
        # ----------------------------------------------------------------

        # -----------------------------
        # GAN loss
        # -----------------------------
        # Generate encoded latent vector
        mu, logvar = self.E(cVAE_data['B'])
        std = torch.exp(logvar / 2)
        random_z = sample_z(self.half_size, self.nz, 'gauss').to(self.device)
        encoded_z = (random_z * std) + mu

        # Generate fake image
        fake_img_cVAE = self.G(cVAE_data['A'], encoded_z)
        # self.writer.add_images('cVAE_output', fake_img_cVAE.add(1.0).mul(0.5), global_step)
        fake_pair_cVAE = torch.cat([cVAE_data['A'], fake_img_cVAE], dim=1)

        # Fool D_cVAE
        fake_D_cVAE_1, fake_D_cVAE_2 = self.D_cVAE(fake_pair_cVAE)

        # Loss for small patch & big patch
        loss_G_cVAE_1 = mse_loss(fake_D_cVAE_1, target=1)
        loss_G_cVAE_2 = mse_loss(fake_D_cVAE_2, target=1)

        # Random latent vector and generate fake image
        random_z = sample_z(self.half_size, self.nz, 'gauss').to(self.device)
        fake_img_cLR = self.G(cLR_data['A'], random_z)
        fake_pair_cLR = torch.cat([cLR_data['A'], fake_img_cLR], dim=1)

        # Fool D_cLR
        fake_D_cLR_1, fake_D_cLR_2 = self.D_cLR(fake_pair_cLR)

        # Loss for small patch & big patch
        loss_G_cLR_1 = mse_loss(fake_D_cLR_1, target=1)
        loss_G_cLR_2 = mse_loss(fake_D_cLR_2, target=1)

        loss_G = loss_G_cVAE_1 + loss_G_cVAE_2 + loss_G_cLR_1 + loss_G_cLR_2
        self.writer.add_scalar('loss/loss_G', loss_G.item(), global_step)

        # -----------------------------
        # KL-divergence (cVAE-GAN)
        # -----------------------------
        kl_div = torch.sum(0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1)) * self.lambda_kl
        self.writer.add_scalar('loss/kl_div', kl_div.item(), global_step)

        # -----------------------------
        # Reconstruction of image B (|G(A, z) - B|) (cVAE-GAN)
        # -----------------------------
        loss_img_recon = l1_loss(fake_img_cVAE, cVAE_data['B']) * self.lambda_img
        self.writer.add_scalar('loss/loss_img_recon', loss_img_recon.item(), global_step)

        loss_E_G = loss_G + kl_div + loss_img_recon
        self.writer.add_scalar('loss/loss_E_G', loss_E_G.item(), global_step)

        # -----------------------------
        # Update E & G
        # -----------------------------
        self.all_zero_grad()
        loss_E_G.backward(retain_graph=True)
        self.optim_E.step()
        self.optim_G.step()

        # ----------------------------------------------------------------
        # 3. Train only G
        # ----------------------------------------------------------------

        # -----------------------------
        # Reconstruction of random latent code (|E(G(A, z)) - z|) (cLR-GAN)
        # -----------------------------
        # This step should update only G.
        # See https://github.com/junyanz/BicycleGAN/issues/5 for details.
        mu, logvar = self.E(fake_img_cLR)

        loss_z_recon = l1_loss(mu, random_z) * self.lambda_z
        self.writer.add_scalar('loss/loss_z_recon', loss_z_recon.item(), global_step)

        # -----------------------------
        # Update G
        # -----------------------------
        self.all_zero_grad()
        loss_z_recon.backward()
        self.optim_G.step()
