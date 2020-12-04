"""Various major components."""

# pylint: disable=no-member

import torch
import torch.nn as nn

from stackgan.utils.blocks import (
    UpsamplingBlock, Conv2dCongruent, JoinBlock,
    ResidualBlock, DownsamplingBlock, ChannelReducingBlock
)

class ConditioningAugmentation(nn.Module):
    """Conditioning Augmentation module.

    Projects the caption embeddings (traditionally produced by Reed et al, 2016
    in text-to-image synthesis frameworks) to a low(er)-dimensional vector and
    adds Gaussian noise with zero mean and learnable diagonal variance.

    Attributes:
        mu_std(nn.Module): the linear layer yielding the mean and
            the standard deviation ("concatenated").
    """

    def __init__(self, cond_dim=128, emb_dim=1024):  # hyperparameters from respective papers
        """Init.

        Args:
            cond_dim(int, optional): Dimension of produced conditioning vector,
                default=`128`, asin StackGAN_v2.
            emb_dim(int, optional): Dimension of original embedding vector,
                default=`1024`, by Reed et al, 2016.
        """

        super().__init__()
        self.mu_std = nn.Sequential(
            nn.Linear(emb_dim, 4 * cond_dim), # *4 because mu & Sigma + GLUop
            nn.GLU(1)
        )

    def forward(self, emb):  # pylint: disable=arguments-differ
        """Forward prop.

        Args:
            emb(torch.Tensor): caption embedding.

        Returns:
            A tuple containing the final conditioning variable,
            the computed mean and computed (diagonal) standard deviation.
        """

        mu_std = self.mu_std(emb)
        cond_dim = mu_std.size(1) // 2
        mus, stds = mu_std[:, :cond_dim], mu_std[:, cond_dim:]
        eps = torch.randn_like(stds)  # N(0, I)
        # reparametrization trick
        return mus + eps * stds, mus, stds

class ConditioningAugmentationOLD(nn.Module):
    """Conditioning Augmentation module.

    Projects the caption embeddings (traditionally produced by Reed et al, 2016
    in text-to-image synthesis frameworks) to a low(er)-dimensional vector and
    adds Gaussian noise with zero mean and learnable diagonal variance. This is the
    implementation used in original repo.

    Attributes:
        mu_std(nn.Module): the linear layer yielding the mean and
            the log of variance ("concatenated").
    """

    def __init__(self, cond_dim=128, emb_dim=1024):  # hyperparameters from respective papers
        """Init.

        Args:
            cond_dim(int, optional): Dimension of produced conditioning vector,
                default=`128`, asin StackGAN_v2.
            emb_dim(int, optional): Dimension of original embedding vector,
                default=`1024`, by Reed et al, 2016.
        """

        super().__init__()
        self.mu_std = nn.Sequential(
            nn.Linear(emb_dim, 4 * cond_dim), # *4 because mu & Sigma + GLUop
            nn.GLU(1)
        )

    def forward(self, emb):  # pylint: disable=arguments-differ
        """Forward prop.

        Args:
            emb(torch.Tensor): caption embedding.

        Returns:
            A tuple containing the final conditioning variable,
            the computed mean and computed (diagonal) standard deviation.
        """

        mu_std = self.mu_std(emb)
        cond_dim = mu_std.size(1) // 2
        mus, logvars = mu_std[:, :cond_dim], mu_std[:, cond_dim:]
        stds = (logvars / 2).exp()
        eps = torch.randn_like(stds)  # N(0, I)
        # reparametrization trick
        return mus + eps * stds, mus, stds

class RootDecoder(nn.Module):
    """Decoder that upsamples its input. Channels are divided by 2
    and other dims are upsampled by 2 for each upsampling block.
    Initial module in StackGAN_v2 (root).

    Projects concatenated condititioning variable and noise vector
    to size (init_channels, 4, 4) with a Linear layer. Batchnorm
    is applied. GLU is the activations function. Afterwards,
    n_upsamples UpsamplingBlock blocks are used, resulting in a
    representation of size (init_channels / 2 ** n_upsample,
    4 * 2 ** n_upsample, 4 * 2 ** n_upsample).

    Attributes:
        fc(nn.Module): linear layer to project conditioning variable
            and noise to (4*4*init_channels)-dimensional vector
            [view is required to get proper shape].
        upsampling_blocks(nn.Module): n_upsample UpsamplingBlock blocks,
            each doubling width and height but halving # channels.
    """

    def __init__(self, init_channels, cond_dim, z_dim, n_upsample=4):
        """Init.

        Args:
            init_channels(int): # channels of initial representation.
            cond_dim(int): dimension of conditioning variable.
            z_dim(int): dimension of noise.
            n_upsamples(int, optional): # upsampling blocks, default=`4`.

        Raises:
            AssertionError: 2 ** n_upsample is greater init_channels.
        """

        super().__init__()

        # use linear layer to project input to 4 x 4 x init_channels
        # view is used to get appropriate shape in forward()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim + z_dim, 4 * 4 * init_channels * 2, bias=False),
            nn.BatchNorm1d(init_channels * 4 * 4 * 2),
            nn.GLU(1),
        )

        # halve channels with each block
        channels = [init_channels // 2 ** i for i in range(n_upsample+1)]
        assert channels[-1] > 0, 'Too many upsampling blocks / Too few channels'
        self.upsampling_blocks = nn.Sequential(
            *[UpsamplingBlock(in_ch, out_ch, 3) for in_ch, out_ch in
              zip(channels[:-1], channels[1:])]
        )

    def forward(self, c, z):  # pylint: disable=arguments-differ
        """Forward prop.

        Args:
            c(torch.Tensor): conditioning variable.
            z(torch.Tensor): random noise vector.

        Returns:
            Representation of size  (init_channels / 2 ** n_samples,
            4 * 2 ** n_samples, 4 * 2 ** n_samples).
        """

        x = torch.cat((c, z), dim=-1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.upsampling_blocks(x)

        return x

class NodeDecoder(nn.Module):
    """Decoder with residual blocks and one upsample block.
    Channels are divided by 2 and other dims are upsampled by 2.
    Intermediate module in StackGAN_v2 (node).

    Uses a JoinBlock to merge the conditioning variable with the
    representation of the previous decoder part. Afterwards,
    n_residual ResidualBlock blocks are used and, finally,
    an UpsamplingBlock that doubles width and height
    and halves # channels.

    Attributes:
        join_block(nn.Module): JoinBlock block used to
            join conditioning variable with input representation.
        res_blocks(nn.Module): n_residual ResidualBlock blocks.
        upsampling_block(nn.Module): UpsamplingBlock block
            halving # channels and doubling width and height.
        concat(function): function that concats representation
            and conditioning variable.
    """

    def __init__(self, in_channels, cond_dim, n_residual=2, reverse=False):
        """Init.

        Args:
            in_channels(int): # channels of input representation.
            cond_dim(int): dimension of conditioning vector.
            n_residual(int, optional): # ResidualBlock blocks,
                default=`2`.
            reverse(bool, optional): whether to reverse arg order in
                concatenation, default=`False`.
        """

        super().__init__()

        # join convolution for representations and conditioning variables
        self.concat = lambda x, c: _concat_channels(x, c, reverse)
        self.join_block = JoinBlock(in_channels, cond_dim)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(in_channels, 3) for _ in range(n_residual)]
        )

        self.upsampling_block = UpsamplingBlock(in_channels, in_channels // 2, 3)

    def forward(self, x, c):  # pylint: disable=arguments-differ
        """Forward prop.

        Args:
            x(torch.Tensor): representation of previous decoder.
            c(torch.Tensor): conditioning variable.
        """

        x = self.concat(x, c)
        x = self.join_block(x)
        x = self.res_blocks(x)
        x = self.upsampling_block(x)

        return x

class Generator(nn.Module):
    """Shallow congruent generator.

    Produces an image with the width and the height of the
    input representation using Conv2dConguent.

    Attributes:
        generate(nn.Module): Conv2dConguent block projecting
            to 3 channels and tanh for proper range.
    """

    def __init__(self, in_channels):
        """Init.

        Args:
            in_channels(int): # channels of input representation.
        """

        super().__init__()

        self.generate = nn.Sequential(
            Conv2dCongruent(in_channels, 3, 3, bias=True),
            nn.Tanh()
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        """Generate image at the same scale as the input.

        args:
            x(torch.Tensor): input representation.
        """

        x = self.generate(x)
        return x

class Discriminator(nn.Module):
    """Joint Conditional & Unconditional discriminator.

    Outputs probabilities both for unconditional generation
    and conditional. It first downsamples the input image with
    DownsamplingBlock blocks until height and width are equal to 4.
    However, every block also doubles # channels. Given that we want
    a (8 * init_channels, 4, 4), we use enough ChannelReducingBlock
    blocks, each halving # channels to get to the desired
    dimensionality. For the unconditional loss, a nn.Conv2d is
    used before the sigmoid, whereas for the conditional, we
    first join the representation with the conditioning variable
    with a ChannelsReducingBlock by reprojecting to the
    representation's size before the nn.Conv2d. Can optionally
    be auxiliary classifier.

    Attributes:
        downsampling_blocks(nn.Module): enough DownsamplingBlock
            blocks to reach width and height equal to 4
            (each halves them). # channels are doubled with
            each also.
        channel_manager(nn.Module): final # channels ought
            to be 8*init_channels, so enough ChannelReducingBlock
            blocks are used, each halving the # channels.
        prob_uncond(nn.Module): Conv2d resulting in a single
            element for sigmoid.
        join_block(nn.Module): ChannelReducingBlock block to
            join conditioning variable with extracted
            representation for conditional evaluation of the
            input image.
        prob_cond(nn.Module): Conv2d resulting in a single
            element for sigmoid.
        concat(function): function that concats representation
            and conditioning variable.
    """

    def __init__(self, init_channels, img_dim, cond_dim, n_class=None, reverse=False):
        """Init.

        Args:
            init_channels(int): # channels after the first
                convolutional block.
            img_dim(int): dimension of input image.
            cond_dim(int): dimension of conditioning vector.
            n_class(int|None): number of output classes for auxiliary
                classifier, default=`None` (no auxiliary classifier).
            reverse(bool, optional): whether to reverse arg order in
                concatenation, default=`False`.
        """

        super().__init__()

        # downsample until img_dim == 4
        # every time we downsample, channels are doubled
        # however, we want the final channels to be 8 * init_channels
        # so we halve them with Conv2dCongruent blocks if necessary

        # -1 because of 3 -> init_channels block
        n_downsample = int(torch.log2(torch.tensor(img_dim / 4))) - 1  # pylint: disable=not-callable
        n_conguent = max(0, n_downsample - 3) # takes 3 from 1 to 8 when x2

        # downsample until 4x4
        channels = [3] + [init_channels * 2 ** i for i in range(n_downsample+1)]
        self.downsampling_blocks = nn.Sequential(
            # trick wont work properly if init_channels == 3, doesnt really matter
            *[DownsamplingBlock(in_ch, out_ch, 4, batchnorm=(in_ch != 3))
              for in_ch, out_ch in zip(channels[:-1], channels[1:])]
        )

        # return to 8 * init_channels channels
        channels = [channels[-1] // 2 ** i for i in range(n_conguent+1)]
        self.channel_manager = nn.Sequential(
            *[ChannelReducingBlock(in_ch, out_ch, 3) for in_ch, out_ch in
              zip(channels[:-1], channels[1:])]
        )

        self.prob_uncond = nn.Sequential(
            nn.Conv2d(8 * init_channels, 1, 4, stride=4),
            nn.Sigmoid()
        )

        # join block for conditional loss
        self.join_block = ChannelReducingBlock(8 * init_channels + cond_dim,
                                               8 * init_channels, 3)
        self.prob_cond = nn.Sequential(
            nn.Conv2d(8 * init_channels, 1, 4, stride=4),
            nn.Sigmoid()
        )

        self.concat = lambda x, c: _concat_channels(x, c, reverse)

        if n_class is not None:
            self.auxiliary_classifier = nn.Conv2d(8 * init_channels, n_class, 4, stride=4)

    def forward(self, x, cond_var):  # pylint: disable=arguments-differ
        """Forward prop.

        Args:
            x(torch.Tensor): input image.
            c(torch.Tensor): conditioning variable.

        Returns:
            torch.Tensor of size (2, batch_size) containing conditional [0, :] and
            unconditional probabilities [1, :]. If auxiliary classifier, also
            returns logits of size (batch_size, n_class).
        """

        x = self.downsampling_blocks(x)
        x = self.channel_manager(x)

        prob_uncond = self.prob_uncond(x)

        x = self.concat(x, cond_var)
        x = self.join_block(x)

        prob_cond = self.prob_cond(x)

        probs = torch.stack((prob_cond.view(-1), prob_uncond.view(-1)), dim=0)

        try:
            logits = self.auxiliary_classifier(x)
            return probs, logits.squeeze()
        except AttributeError:
            return probs

def _concat_channels(repre, cond_var, reverse=False):
    """Concatenates repr and cond as suggested for the representation
    and the conditioning variable in the StackGAN_v2 architecture.

    Concatenates repr and cond channel-wise by viewing cond as
    a 1x1 image and replicating it to match repr's width and height.

    Args:
        repre(torch.Tensor): representation of size
            (batch_size, channels, height, width).
        cond_var(torch.Tensor): conditioning variable vector.
        reverse(bool, optional): whether to reverse arg order in
            concatenation, default=`False`.

    Returns:
        torch.Tensor of concatenated representation if cond is not
        `None`, else the original representation.
    """

    if cond_var is not None:
        cond_var = cond_var.view(cond_var.size(0), -1, 1, 1)
        cond_var = cond_var.repeat(1, 1, repre.size(2), repre.size(3))
        if reverse:
            return torch.cat((cond_var, repre), dim=1)
        return torch.cat((repre, cond_var), dim=1)
    return repr

class MLP(nn.Module):
    """Simple MLP.

    Attributes:
        layers(nn.Module): sequence of layers.
    """

    def __init__(self, in_features, out_features, hidden_layers=None, dropout=0,
                 hidden_actf=nn.LeakyReLU(0.2), output_actf=nn.ReLU()):
        """Init.

        Args:
            in_features(int): input dimension.
            out_features(int): final output dimension.
            hidden_layers(list of ints|int|None): list of hidden
                layer sizes of arbitrary length or int for one
                hidden layer.
        """

        if hidden_layers is None:
            hidden_layers = []
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        super().__init__()

        hidden_layers = [in_features] + hidden_layers + [out_features]

        layers = []
        for i, (in_f, out_f) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
            layers.append(nn.Linear(in_f, out_f))

            if i != len(hidden_layers) - 2:
                # up to second-to-last layer
                layers.append(hidden_actf)
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(output_actf)  # ok to use relu, resnet feats are >= 0

        self.layers = nn.Sequential(*layers)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward propagation.

        Args:
            x(torch.Tensor): input of size (batch, in_features).

        Returns:
            A torch.Tensor of size (batch, out_features).
        """
        return self.layers(x)

    def init_diagonal(self):
        """Sets weights of linear layers to approx I
        and biases to 0.
        """
        def init_fn(mod):
            """Function to pass to .apply()."""
            classname = mod.__class__.__name__
            if classname.find('Linear') != -1:
                init = torch.randn_like(mod.weight) / 10
                init[range(mod.in_features), range(mod.in_features)] = 1
                mod.weight = nn.Parameter(init, requires_grad=True)
                if mod.bias is not None:
                    mod.bias = nn.Parameter(torch.zeros_like(mod.bias), requires_grad=True)

        self.apply(init_fn)

class ConcatMLP(MLP):
    """Simple MLP that accepts two inputs
    which are to be concatenated."""

    def forward(self, x, y):  # pylint: disable=arguments-differ
        """Concatenates input and noise and forward propagates.
        x's and input's dimensions must add up to in_features.

        Args:
            x(torch.Tensor): input.
            noise(torch.Tensor): randomly sampled noise.

        Returns:
            A torch.Tensor of size (batch, out_features).
        """

        x = torch.cat((x, y), dim=1)
        return super().forward(x)

class RandomRounding(nn.Module):
    """Module that randomly rounds real valued
    attributes (values in [0, 1]).

    Attributes:
        prob(float): probability of rounding a value.
    """

    def __init__(self, prob):
        """Init.

        Args:
            prob(float): probability of rounding a value.
        """
        super().__init__()
        self.prob = prob

    def forward(self, x):  # pylint: disable=arguments-differ
        """Applies rounding.

        Args:
            x(torch.Tensor): attribute tensors.

        Returns:
            A torch.Tensor of the randomly-rounded input.
        """

        inds = torch.bernoulli(torch.ones_like(x) * self.prob)
        return x.where(inds == 0, x.round())
