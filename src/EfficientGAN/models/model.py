from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pytorch_lightning as pl

from .generator import Generator
from .discriminator import Discriminator
from .encoder import Encoder


# networks, optimizer, trainingの3つに分けられる。
class Model(pl.LightningModule):

    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams = hparams

        self.g = Generator(z_dim=hparams['latent_dim'])
        self.d = Discriminator(z_dim=hparams['latent_dim'])
        self.e = Encoder(z_dim=hparams['latent_dim'])

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.g(z)

    def configure_optimizers(self):
        lr = self.hparams['lr']
        b1 = self.hparams['b1']
        b2 = self.hparams['b2']

        opt_d = optim.Adam(self.d.parameters(), lr=lr, betas=(b1, b2))
        opt_g = optim.Adam(self.g.parameters(), lr=lr, betas=(b1, b2))
        opt_e = optim.Adam(self.e.parameters(), lr=lr, betas=(b1, b2))

        return [opt_d, opt_g, opt_e], []

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs, _ = batch
        self.last_imgs = imgs

        # train discriminator
        if optimizer_idx == 0:
            # real loss
            z_out_real = self.e(imgs)
            d_out_real, _ = self.d(imgs, z_out_real)

            valid = torch.ones(imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            real_loss = self.criterion(d_out_real.view(-1), valid)

            # fake loss
            z = torch.randn(imgs.shape[0], self.hparams['latent_dim'])
            if self.on_gpu:
                z = z.cuda(imgs.device.index)
            self.generated_imgs = self.g(z)
            d_out_fake, _ = self.d(self.generated_imgs, z)

            fake = torch.zeros(imgs.size(0), 1)
            if self.on_gpu:
                fake = fake.cuda(imgs.device.index)

            fake_loss = self.criterion(d_out_fake.view(-1), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train generator
        if optimizer_idx == 1:
            z = torch.randn(imgs.shape[0], self.hparams['latent_dim'])
            if self.on_gpu:
                z = z.cuda(imgs.device.index)
            self.generated_imgs = self.g(z)
            d_out_fake, _ = self.d(self.generated_imgs, z)

            valid = torch.ones(imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            g_loss = self.criterion(d_out_fake.view(-1), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            return output

        # train encoder
        if optimizer_idx == 2:
            z_out_real = self.e(imgs)
            d_out_real, _ = self.d(imgs, z_out_real)

            fake = torch.zeros(imgs.size(0), 1)
            if self.on_gpu:
                fake = fake.cuda(imgs.device.index)

            e_loss = self.criterion(d_out_real.view(-1), fake)
            tqdm_dict = {'e_loss': e_loss}
            output = OrderedDict({
                'loss': e_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def test_step(self, batch, batch_nb):
        pass

    def test_epoch_end(self, outputs):
        pass

    def on_epoch_end(self):
        z = torch.randn(8, self.hparams['latent_dim'])
        # match gpu device (or keep as cpu)
        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
