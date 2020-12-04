"""Losses."""

# pylint: disable=no-member

import torch
import torch.nn as nn

class AuxiliaryClassificationLoss(nn.Module):
    """Auxiliary Classification loss.

    Computes cross entropy loss wrt the classification
    logits from the discriminators.

    Attributes:
        individual(bool): whether to return loss of each discriminator
            seperately as 2nd value, default=`False`.
        criterion(nn.Module): loss per discriminator.
    """

    def __init__(self, individual=False):
        """Init.

        Args:
            individual(bool): whether to return loss of each discriminator
                seperately as 2nd value, default=`False`.
        """
        super().__init__()
        self.individual = individual
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):  # pylint: disable=arguments-differ
        """Computes loss.

        Args:
            logits(torch.Tensor): logits of size (3, batch_size, n_class).
            labels(torch.Tensor): integer [synthetic] labels of size (batch_size).

        Returns:
            Sum of losses for all discriminators, and torch.Tensor of
            individual losses if individual==`True`.
        """

        losses = []
        for i in range(logits.size(0)):  # iterate discrs
            losses.append(self.criterion(logits[i], labels))
        if self.individual:
            return sum(losses), torch.stack(losses)
        return sum(losses)

class DiscriminatorsLoss(nn.Module):
    """Ensemble-of-discriminators' loss.

    Computes the loss of all the discriminators for all settings,
    generated images, real images and real images with mismatching
    embeddings.

    Attributes:
        lambda_uncond(Number): Coefficient of unconditional
            probabilities.
        lambda_wrong(Number): Coefficient of probabilities of fake
            and mismatching inputs.
        criterion(nn.Module): criterion per discriminator real/fake
            output.
    """

    def __init__(self, lambda_uncond=1, lambda_wrong=1):
        """Init.

        Args:
            lambda_uncond(Number, optional): Coefficient of unconditional
                probabilities, default=`1`.
            lambda_wrong(Number, optional): Coefficient of probabilities of fake
                and mismatching inputs, default=`1`.
        """
        super().__init__()
        self.lambda_uncond = lambda_uncond
        self.lambda_wrong = lambda_wrong
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, fake_probs, real_probs, mis_probs):  # pylint: disable=arguments-differ
        """Computes loss of all discriminators.

        Args:
            fake_probs(torch.Tensor): size (3, 2, batch_size) with probabilities
                of the discriminators for generated images.
            real_probs(torch.Tensor): size (3, 2, batch_size) with probabilities
                of the discriminators for real images.
            mis_probs(torch.Tensor): size (3, batch_size) with probabilities
                of the discriminators for real images with mismatching embeddings.
        """

        wrong_labels = torch.zeros_like(fake_probs[:, 0])
        right_labels = torch.ones_like(real_probs[:, 0])

        loss = self.lambda_wrong * self.criterion(fake_probs[:, 0], wrong_labels) # cond
        loss = loss + self.lambda_wrong * self.lambda_uncond * \
            self.criterion(fake_probs[:, 1], wrong_labels) # uncond
        loss = loss + self.criterion(real_probs[:, 0], right_labels) # cond
        loss = loss + self.lambda_uncond * self.criterion(real_probs[:, 1], right_labels) # uncond
        loss = loss + self.lambda_wrong * self.criterion(mis_probs, wrong_labels) # mismatch->cond

        return loss / fake_probs.size(-1)


class GeneratorsLoss(nn.Module):
    # (fake_probs, lambda_uncond=1):
    """Ensemble-of-generators' loss.

    Attributes:
        lambda_uncond(Number): Coefficient of unconditional probabilities.

        fake_probs(torch.Tensor): size (3, 2, batch_size) with probabilities
            of the discriminators for generated images.
    """

    def __init__(self, lambda_uncond=1):
        """Init.

        Args:
            lambda_uncond(Number, optional): Coefficient of unconditional
                probabilities, default=`1`.
        """
        super().__init__()
        self.lambda_uncond = lambda_uncond
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, fake_probs):  # pylint: disable=arguments-differ
        """Computes loss.

        Args:
            fake_probs(torch.Tensor): size (3, 2, batch_size) with probabilities
                of the discriminators for generated images.
            """
        right_labels = torch.ones_like(fake_probs[:, 0])
        loss = self.criterion(fake_probs[:, 0], right_labels) # cond
        loss = loss + self.lambda_uncond * self.criterion(fake_probs[:, 1], right_labels) # uncond

        return loss / fake_probs.size(-1)

class KLDivergence(nn.Module):
    """KL Divergence.

    Multivariate KL divergence of N(mu, std ** 2)
    from N(0, I) as shown in
    `https://en.wikipedia.org/wiki/Multivariate_normal_distribution`.

    Attributes:
        eps(float): constant to add to avoid 0 in logarithm.
    """

    def __init__(self, eps=1e-5):
        """Init.

        Args:
            eps(float): constant to add to avoid 0 in logarithm,
                default=`1e-5`.
        """
        super().__init__()
        self.eps = eps

    def forward(self, mus, stds):  # pylint: disable=arguments-differ
        """Computes loss.

        Args:
            mus(torch.Tensor): size (batch_size, cond_dim) with the means.
            stds(torch.Tensor): size (batch_size, cond_dim) with the standard
                deviation elements.

        Returns:
            Loss.
        """

        # NOTE: this calculates the multivariate kl div
        # in contrast to original repo where univariate is calculated
        return 0.5 * ((stds ** 2).sum(dim=-1) + (mus ** 2).sum(dim=-1)
                      - (stds ** 2 + self.eps).log().sum(dim=-1) - stds.size(-1)).mean()
