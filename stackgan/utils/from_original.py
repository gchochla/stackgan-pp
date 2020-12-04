"""Load pretrained model from original repo
to model defined in this repo."""

# pylint: disable=no-member
# pylint: disable=arguments-differ

############################################################
### taken from https://github.com/hanzhanggit/StackGAN-v2 ##

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from stackgan.modules import StackGAN_v2
from stackgan.utils.submodules import ConditioningAugmentationOLD


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)

# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )


    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = 1024
        self.ef_dim = 128
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = 100 + 128
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())


        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code=None):
        in_code = torch.cat((c_code, z_code), 1)
        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=2):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = 128
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, h_code), 1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = 64
        self.define_module()

    def define_module(self):
        self.ca_net = CA_NET()

        self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
        self.img_net1 = GET_IMAGE_G(self.gf_dim)
        self.h_net2 = NEXT_STAGE_G(self.gf_dim)
        self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2)
        self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)

    def forward(self, z_code, text_embedding=None):
        c_code, mu, logvar = self.ca_net(text_embedding)

        fake_imgs = []
        h_code1 = self.h_net1(z_code, c_code)
        fake_img1 = self.img_net1(h_code1)
        fake_imgs.append(fake_img1)
        h_code2 = self.h_net2(h_code1, c_code)
        fake_img2 = self.img_net2(h_code2)
        fake_imgs.append(fake_img2)
        h_code3 = self.h_net3(h_code2, c_code)
        fake_img3 = self.img_net3(h_code3)
        fake_imgs.append(fake_img3)

        return fake_imgs, mu, logvar

### taken from https://github.com/hanzhanggit/StackGAN-v2 ##
############################################################

def load_original_pretrained(model_fn):
    """Loads pretrained original model.

    Args:
        model_fn(str): path to model checkpoint.

    Returns:
        Pretrained G_NET.
    """

    netG = nn.DataParallel(G_NET(), device_ids=[0])
    netG.load_state_dict(torch.load(model_fn, map_location='cpu'))
    return next(iter(netG.children()))

def match_nets(my_net, their_net):
    """Transfer weights between original implementation
    and this implementation.

    Args:
        my_net(nn.Module): StackGAN define in this repo.
        their_net(nn.Module): StackGAN define in original repo.
    """
    my_net.cond_aug.mu_std[0] = their_net.ca_net.fc
    my_net.decoders[0].fc[0] = their_net.h_net1.fc[0]
    my_net.decoders[0].fc[1] = their_net.h_net1.fc[1]
    my_net.decoders[0].upsampling_blocks[0][1] = their_net.h_net1.upsample1[1]
    my_net.decoders[0].upsampling_blocks[0][2] = their_net.h_net1.upsample1[2]
    my_net.decoders[0].upsampling_blocks[1][1] = their_net.h_net1.upsample2[1]
    my_net.decoders[0].upsampling_blocks[1][2] = their_net.h_net1.upsample2[2]
    my_net.decoders[0].upsampling_blocks[2][1] = their_net.h_net1.upsample3[1]
    my_net.decoders[0].upsampling_blocks[2][2] = their_net.h_net1.upsample3[2]
    my_net.decoders[0].upsampling_blocks[3][1] = their_net.h_net1.upsample4[1]
    my_net.decoders[0].upsampling_blocks[3][2] = their_net.h_net1.upsample4[2]
    my_net.decoders[1].join_block[0] = their_net.h_net2.jointConv[0]
    my_net.decoders[1].join_block[1] = their_net.h_net2.jointConv[1]
    my_net.decoders[1].res_blocks[0].resblock[0] = their_net.h_net2.residual[0].block[0]
    my_net.decoders[1].res_blocks[0].resblock[3] = their_net.h_net2.residual[0].block[3]
    my_net.decoders[1].res_blocks[0].resblock[1] = their_net.h_net2.residual[0].block[1]
    my_net.decoders[1].res_blocks[0].resblock[4] = their_net.h_net2.residual[0].block[4]
    my_net.decoders[1].res_blocks[1].resblock[0] = their_net.h_net2.residual[1].block[0]
    my_net.decoders[1].res_blocks[1].resblock[3] = their_net.h_net2.residual[1].block[3]
    my_net.decoders[1].res_blocks[1].resblock[1] = their_net.h_net2.residual[1].block[1]
    my_net.decoders[1].res_blocks[1].resblock[4] = their_net.h_net2.residual[1].block[4]
    my_net.decoders[1].upsampling_block[1] = their_net.h_net2.upsample[1]
    my_net.decoders[1].upsampling_block[2] = their_net.h_net2.upsample[2]
    my_net.decoders[2].join_block[0] = their_net.h_net3.jointConv[0]
    my_net.decoders[2].join_block[1] = their_net.h_net3.jointConv[1]
    my_net.decoders[2].res_blocks[0].resblock[0] = their_net.h_net3.residual[0].block[0]
    my_net.decoders[2].res_blocks[0].resblock[3] = their_net.h_net3.residual[0].block[3]
    my_net.decoders[2].res_blocks[0].resblock[1] = their_net.h_net3.residual[0].block[1]
    my_net.decoders[2].res_blocks[0].resblock[4] = their_net.h_net3.residual[0].block[4]
    my_net.decoders[2].res_blocks[1].resblock[0] = their_net.h_net3.residual[1].block[0]
    my_net.decoders[2].res_blocks[1].resblock[3] = their_net.h_net3.residual[1].block[3]
    my_net.decoders[2].res_blocks[1].resblock[1] = their_net.h_net3.residual[1].block[1]
    my_net.decoders[2].res_blocks[1].resblock[4] = their_net.h_net3.residual[1].block[4]
    my_net.decoders[2].upsampling_block[1] = their_net.h_net3.upsample[1]
    my_net.decoders[2].upsampling_block[2] = their_net.h_net3.upsample[2]
    my_net.generator.generate[0] = their_net.img_net3.img[0]
    my_net.generator.generate[0].bias = nn.Parameter(
        torch.zeros(my_net.generator.generate[0].out_channels)
    )

def load_pretrained_directly_from_original(model_fn):
    """Loads pretrained sample generator from weights
    of original implementation.

    Args:
        model_fn(str): path to checkpoint of
            original StackGAN-v2.

    Returns:
        SampleGenerator as defines in this repo.
    """

    gnet = load_original_pretrained(model_fn)
    sample_generator = StackGAN_v2(16, 1, 128, 100, reverse=True).sample_generator()
    sample_generator.cond_aug = ConditioningAugmentationOLD()
    match_nets(sample_generator, gnet)
    return sample_generator

def load_pretrained_indirectly_from_original(model_fn):
    """Loads pretrained sample generator from weights
    of this repo's implementation, but with compatible
    code compared to the original implementation.

    Args:
        model_fn(str): path to checkpoint of
            this repo's StackGAN-v2.

    Returns:
        SampleGenerator as defines in this repo.
    """

    sample_generator = StackGAN_v2(16, 1, 128, 100, reverse=True).sample_generator()
    sample_generator.cond_aug = ConditioningAugmentationOLD()
    sample_generator.load_state_dict(torch.load(model_fn, map_location='cpu'))
    return sample_generator
