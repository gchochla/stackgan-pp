"""Various building blocks. All are meant to be usable
inside a nn.Sequential module. Naming of functions
deviates from snake_case because every block should be
treated as a nn.Module."""

import torch.nn as nn

def Conv2dCongruent(in_channels, out_channels, kernel_size, bias=False):  # pylint: disable=invalid-name
    """2D Convolution that preserves input width and height.

    Args:
        in_channels(int), out_channels(int): channels of convolution.
        kernel_size(int): convolution kernel size, must be odd.
        bias(bool, optional): whether to use bias in convolution,
            default=`False`.

    Returns:
        Described nn.Module.

    Raises:
        AssertionError: kernel_size is even integer.
    """

    assert kernel_size % 2 == 1, 'Kernel size must be odd'

    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                     padding=(kernel_size // 2), bias=bias)

def UpsamplingBlock(in_channels, out_channels, kernel_size):  # pylint: disable=invalid-name
    """Upsampling block (factor of 2) as used in StackGAN_v2.

    Performs upsampling by a factor of 2 and then a Conv2dCongruent
    convolution to avoid artifacts of deconvolutions. BatchNorm is
    applied. GLU is the actuvation function.

    Arguments:
        in_channels(int), out_channels(int): # channels of convolution.
        kernel_size(int): convolution kernel size, must be odd.

    Returns:
        Described nn.Module

    Raises:
        AssertionError: kernel_size is even integer.
    """

    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        Conv2dCongruent(in_channels, 2 * out_channels, kernel_size),
        nn.BatchNorm2d(2 * out_channels),
        nn.GLU(1),
    )

    return block

class ResidualBlock(nn.Module):
    """Residual block.

    Performs two Conv2dCongruent convolutions and adds
    result back into the input tensor. GLU is also used
    as the activation function.

    Attributes:
        resblock(nn.Module): block containing 2 Conv2dCongruent
            blocks, meant to be added to the original input.
    """

    def __init__(self, channels, kernel_size):
        """Init.

        Args:
            channels(int): input # channels.
            kernel_size(int): convolution kernel size.

        Raises:
            AssertionError: kernel_size is even integer.
        """

        super().__init__()
        self.resblock = nn.Sequential(
            Conv2dCongruent(channels, 2 * channels, kernel_size),
            nn.BatchNorm2d(2 * channels),
            nn.GLU(1),
            Conv2dCongruent(channels, channels, kernel_size),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        """Add output of block to input.

        Args:
            x(torch.Tensor): input.
        """

        return self.resblock(x) + x

def JoinBlock(in_channels, cond_dim):  # pylint: disable=invalid-name
    """Block to join representation with conditioning variable
    in StackGAN_v2.

    Uses a Conv2dConguent to project the channels back to the
    original representation's channels. Batchnorm is applied.
    GLU is the activation function. Concatenation of representation
    and conditioning variable is assumed to have been achieved by
    considering the vector dimension as the channel dimension of a
    volume and its element replicated to match the width and height
    of the representation.

    Args:
        in_channels(int): input # channels.
        cond_dim(int): dim of conditioning variable to merge.

    Returns:
        Described nn.Module.
    """

    block = nn.Sequential(
        Conv2dCongruent(in_channels + cond_dim, 2 * in_channels, 3),
        nn.BatchNorm2d(2 * in_channels),
        nn.GLU(1)
    )

    return block

def DownsamplingBlock(in_channels, out_channels, kernel_size,  # pylint: disable=invalid-name
                      batchnorm=True):
    """Downsampling block (factor of 2) as used in StackGAN_v2.

    A nn.Cond2d is applied that halves input width and height.
    Batchnorm is applied if specified so. LeakyReLU (a=0.2)
    is the activation function.

    Args:
        in_channels(int), out_channels(int): # channels of convolution.
        kernel_size(int): convolution kernel size.
        batchnorm(bool, optional): whether to apply batchnorm,
            default=`True`.

    Returns:
        Described nn.Module.

    Raises:
        AssertionError: kernel_size is odd integer."""

    assert kernel_size % 2 == 0, 'Kernel size must be even'

    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride=2, padding=kernel_size//2-1, bias=False),
        nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
        nn.LeakyReLU(0.2),
    )

    return block

def ChannelReducingBlock(in_channels, out_channels, kernel_size):  # pylint: disable=invalid-name
    """Block used in StackGAN discriminators to reduce # channels.

    Uses Conv2dCongruent to retain wight and height. Batchnorm is
    applied. LeakyReLU is the activation function.

    Args:
        in_channels(int), out_channels(int): # channels of convolution.
        kernel_size(int): convolution kernel size.

    Returns:
        Described nn.Module.

    Raises:
        AssertionError: kernel_size is even integer.
    """

    block = nn.Sequential(
        Conv2dCongruent(in_channels, out_channels, kernel_size),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )

    return block
