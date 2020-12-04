"""StackGAN_v2 trainer."""

# pylint: disable=no-member

import os
import time
import argparse

import torch
import torch.optim as optim
from torchvision.transforms import transforms

from stackgan.modules import StackGAN_v2
from stackgan.utils.datasets import CUBDatasetLazy
from stackgan.utils.losses import (
    AuxiliaryClassificationLoss, DiscriminatorsLoss, GeneratorsLoss,
    KLDivergence
)

class StackGANv2Trainer:
    """StackGAN_v2 trainer.

    Trains or finetunes a StackGAN_v2 model.

    Attributes:
        stackgan(nn.Module): instance of StackGAN_v2.
        noise_dim(int): dimension of noise vector.
        dataset(torch.utils.data.DataLoader): training dataset
            of images and embeddings.
        label_flip_prob(float): probability of flipping real and
            fake labels of images during training.
        g_optimizer(optim.Optimizer): decoder & generators optimizer.
        d_optimizer(optim.Optimizer): discriminators optimizer.
        loss_coeffs(tuple of Numbers): 3 Numbers used during training
            to scale losses (uncond, wrong, kl -> check losses doc).
        epoch_range(range): training epochs (may not start from 0
            if finetuning).
        root_image_dir(str): directory where generated images are
            saved during training.
        sample_embeddings(torch.Tensor): embeddings used to generate
            images that are saved after every epoch.
        model_fn(str): filename of model checkpoint.
        checkpoint_interval(int): per how many epochs to save model.
        verbose(bool): whether to print metrics per epoch to stdout.
        metric_logger(TextIOWrapper): pointer to logging file.
    """

    def __init__(self, gan_kw, dataset_kw, optim_kw, log_kw):
        """Init.

        Args:
            gan_kw(dict): StackGAN-related arguments ->
                (restore[optional], Ng, Nd, cond_dim,
                noise_dim, device).
            dataset_kw(dict): dataset-related arguments ->
                (dataset_dir, image_dir, embedding_dir,
                available_classes[optional], batch_size).
            optim_kw(dict): optimization-related arguments ->
                (label_flip_prob, glr, dlr, beta_1,
                uncond_coef[optional], wrong_coef[optional],
                kl_coef[optional], epochs, aux_coef[optional]).
            log_kw(dict): logging-related arguments ->
                (log_dir, n_samples, model_dir,
                checkpoint_interval).
        """

        self.device = gan_kw['device']

        uncond_coef = optim_kw.get('uncond_coef', 1)
        wrong_coef = optim_kw.get('wrong_coef', 1)
        # 2 / cond_dim because of dif. implementation of KL from original
        kl_coef = optim_kw.get('kl_coef', 2 / gan_kw['cond_dim'])
        aux_coef = optim_kw.get('aux_coef', 0.)

        self.loss_coefs = uncond_coef, wrong_coef, kl_coef, aux_coef

        ###############################################################
        ### dataset
        ###############################################################
        dataset = CUBDatasetLazy(dataset_kw['dataset_dir'], dataset_kw['image_dir'],
                                 dataset_kw['embedding_dir'],
                                 dataset_kw.get('available_classes', None),
                                 train=True)

        self.dataset = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=4,
                                                   batch_size=dataset_kw['batch_size'])

        ###############################################################
        ### model
        ###############################################################
        n_class = len(dataset.synthetic_ids) if aux_coef > 0 else None
        self.stackgan = StackGAN_v2(gan_kw['Ng'], gan_kw['Nd'], gan_kw['cond_dim'],
                                    gan_kw['noise_dim'], n_class)
        self.stackgan.to(self.device).train()
        self.noise_dim = gan_kw['noise_dim']

        ###############################################################
        ### optimizers
        ###############################################################

        # separate params of decoder & generators from discriminators ones
        g_params = []
        for g_submod in self.stackgan.generators:
            g_params = g_params + list(g_submod.parameters())
        for g_submod in self.stackgan.decoders:
            g_params = g_params + list(g_submod.parameters())
        # NOTE: Conditioning Augmentation is only trained through generator error
        g_params = g_params + list(self.stackgan.cond_aug.parameters())

        d_params = []
        for d_submod in self.stackgan.discriminators:
            d_params = d_params + list(d_submod.parameters())

        # probability of flipping the labels of real and generated images
        self.label_flip_prob = optim_kw.get('label_flip_prob', 0)

        self.g_optimizer = optim.Adam(g_params, lr=optim_kw['glr'],
                                      betas=(optim_kw['beta_1'], 0.999))  # 0.999 is default

        self.d_optimizer = optim.Adam(d_params, lr=optim_kw['dlr'],
                                      betas=(optim_kw['beta_1'], 0.999))  # 0.999 is default

        self.epoch_range = range(optim_kw['epochs'])

        ###############################################################
        ### logging
        ###############################################################
        hyparams = [gan_kw['Ng'], gan_kw['Nd'], gan_kw['cond_dim'],
                    gan_kw['noise_dim'], optim_kw['glr'], optim_kw['dlr'],
                    optim_kw['beta_1'], uncond_coef, wrong_coef, kl_coef, aux_coef,
                    self.label_flip_prob]
        hyparam_str = '_'.join([str(hyparam) for hyparam in hyparams])

        # file to write metrics to
        if not os.path.exists(log_kw['log_dir']):
            os.makedirs(log_kw['log_dir'])
        self.metrics_fn = os.path.join(log_kw["log_dir"], '{}_metrics.csv'.format(hyparam_str))

        # root image dir
        self.root_image_dir = os.path.join(log_kw['log_dir'], '{}_images'.format(hyparam_str))
        if not os.path.exists(self.root_image_dir):
            os.makedirs(self.root_image_dir)

        # pick random but const images and corr embeddings for qualitative comp
        idc = torch.randperm(len(dataset))[:log_kw.get('n_samples', 6)]
        self.sample_embeddings = torch.stack([dataset[i.item()][2].to(self.device) for i in idc],
                                             dim=0)
        # save actual images for reference
        self.log_sample_images(data=[dataset[i.item()][0] for i in idc])

        # where to save model and optimizer dicts
        if not os.path.exists(log_kw['model_dir']):
            os.makedirs(log_kw['model_dir'])
        self.model_fn = os.path.join(log_kw['model_dir'], '{}.pt'.format(hyparam_str))
        self.checkpoint_interval = log_kw.get('checkpoint_interval', optim_kw['epochs'] // 10)

        self.verbose = log_kw.get('verbose', False)

        ###############################################################
        ### restoring (if asked to do so)
        ###############################################################

        if gan_kw.get('restore', False):
            try:
                state_dict = torch.load(self.model_fn, map_location=self.device)
                self.stackgan.load_state_dict(state_dict['model'])
                self.g_optimizer.load_state_dict(state_dict['g_optimizer'])
                self.d_optimizer.load_state_dict(state_dict['d_optimizer'])
                self.epoch_range = range(state_dict['epoch']+1, optim_kw['epochs'])
                print('Model checkpoint found!')
            except FileNotFoundError:
                print('No model checkpoint found!')


    def __call__(self):
        """Trains StackGAN_v2.

        StackGAN_v2 is trained on the CUB dataset. Probabilities of the
        discriminator for all possible image-embedding pairs are logged,
        along with generated images at the end of each epoch.
        """

        self.stackgan.train()

        lambda_uncond, lambda_wrong, lambda_kl, lambda_aux = self.loss_coefs
        clsf_criterion = AuxiliaryClassificationLoss(individual=True)
        discr_criterion = DiscriminatorsLoss(lambda_uncond, lambda_wrong)
        gen_criterion = GeneratorsLoss(lambda_uncond)
        kl_divergence = KLDivergence()

        for epoch in self.epoch_range:
            init_time = time.time()

            # epoch metrics
            fake_probs_avg = torch.zeros(3, 2, device=self.device)
            real_probs_avg = fake_probs_avg.clone()
            mis_probs_avg = torch.zeros(3, device=self.device)
            clsf_fake_loss_avg = torch.zeros(3, device=self.device)
            clsf_real_loss_avg = clsf_fake_loss_avg.clone()

            for imgs, mis_imgs, embs, lbls in self.dataset:

                # transfer to device
                imgs, mis_imgs, embs, lbls = (imgs.to(self.device), mis_imgs.to(self.device),
                                              embs.to(self.device), lbls.to(self.device))

                # noise vec
                noise = torch.randn(imgs.size(0), self.noise_dim, device=self.device)

                ###########################
                ###  optimize generators ##

                gen_imgs, _, mus, stds = self.stackgan(embs, noise)
                if lambda_aux > 0:
                    fake_probs, fake_logits = self.stackgan.discr_fake_images(gen_imgs, mus=mus)
                    clsf_loss, ind_losses = clsf_criterion(fake_logits, lbls)
                    # update metric
                    clsf_fake_loss_avg += ind_losses.detach()
                else:
                    fake_probs = self.stackgan.discr_fake_images(gen_imgs, mus=mus)
                    clsf_loss = 0

                gan_loss = gen_criterion(fake_probs)
                kl_loss = kl_divergence(mus, stds)

                g_loss = (gan_loss + lambda_kl * kl_loss + lambda_aux * clsf_loss)

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                ###  optimize generators ##
                ###########################

                ###########################
                ### optimize discriminators

                if lambda_aux > 0:
                    fake_probs, fake_logits = self.stackgan.discr_fake_images(
                        [gen_imgs_i.detach() for gen_imgs_i in gen_imgs], mus=mus
                    )
                    real_probs, real_logits = self.stackgan.discr_real_images(imgs, mus=mus)
                    mis_probs, _ = self.stackgan.discr_real_images(mis_imgs, mus=mus)
                    mis_probs = mis_probs[:, 0] # only keep cond

                    clsf_loss_fake, ind_losses_fake = clsf_criterion(fake_logits, lbls)

                    clsf_loss_real, ind_losses_real = clsf_criterion(real_logits, lbls)

                    clsf_loss = clsf_loss_real + clsf_loss_fake
                    # update metric
                    clsf_real_loss_avg += ind_losses_real.detach()
                    clsf_fake_loss_avg += ind_losses_fake.detach()
                else:
                    fake_probs = self.stackgan.discr_fake_images(
                        [gen_imgs_i.detach() for gen_imgs_i in gen_imgs], mus=mus
                    )
                    real_probs = self.stackgan.discr_real_images(imgs, mus=mus)
                    mis_probs = self.stackgan.discr_real_images(mis_imgs,
                                                                mus=mus)[:, 0]  # only keep cond

                    clsf_loss = 0

                if torch.rand(1).item() < self.label_flip_prob:  # flip labels
                    gan_loss = discr_criterion(real_probs, fake_probs, mis_probs)  # pylint: disable=arguments-out-of-order
                else:  # normal execution
                    gan_loss = discr_criterion(fake_probs, real_probs, mis_probs)

                d_loss = gan_loss + lambda_aux * clsf_loss

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                ### optimize discriminators
                ###########################

                # update metrics
                fake_probs_avg += fake_probs.sum(dim=-1).detach()
                real_probs_avg += real_probs.sum(dim=-1).detach()
                mis_probs_avg += mis_probs.sum(dim=-1).detach()

            if (epoch + 1) % self.checkpoint_interval == 0:
                torch.save({
                    'model': self.stackgan.state_dict(),
                    'g_optimizer': self.g_optimizer.state_dict(),
                    'd_optimizer': self.d_optimizer.state_dict(),
                    'epoch': epoch
                }, self.model_fn)

            # average
            fake_probs_avg /= len(self.dataset.dataset)
            real_probs_avg /= len(self.dataset.dataset)
            mis_probs_avg /= len(self.dataset.dataset)
            clsf_fake_loss_avg /= 2 * len(self.dataset.dataset)  # /2 because discr and gen optim
            clsf_real_loss_avg /= len(self.dataset.dataset)

            self.log_metrics(epoch, fake_probs_avg, real_probs_avg,
                             mis_probs_avg, clsf_fake_loss_avg,
                             clsf_real_loss_avg, init_time)
            self.log_sample_images(epoch=epoch)

    def log_metrics(self, epoch, fake_probs, real_probs, mis_probs,
                    clsf_fake_loss, clsf_real_loss, init_time=None):
        """Logs metrics.

        Write probabilities of discriminator in a csv file.

        Args:
            epoch(int): epoch of training.
            fake_probs(torch.Tensor): [average] probabilities of the
                discriminators for generated images. Size (3, 2).
            real_probs(torch.Tensor): [average] probabilities of real
                images paired correctly with embeddings. Size (3, 2).
            mis_probs(torch.Tensor): [average] probabilities of real
                images paired incorrectly with embeddings. Size (3, 2).
            clsf_fake_loss(torch.tensor): [average] classification loss across all
                discriminators for generated images. Size (3,).
            clsf_real_loss(float): [average] classification loss across all
                discriminators for real images. Size (3,).
            init_time(Number, optional): time epoch started, only printed if
                verbose was set to `True`.
        """

        lambda_uncond, _, _, lambda_aux = self.loss_coefs

        conds = [''] + (['un'] if lambda_uncond > 0 else [])
        inds = [0] + ([1] if lambda_uncond > 0 else [])

        # iter cond/uncond 1st, discr 2nd and type of prob 3rd
        # can iter prob tensors' 2nd dim, then 1st dim and then tensors themselves
        headers = (['epoch'] +
                   ['{}_D{}_{}cond'.format(type, i, cond) for type in ['fake', 'real']
                    for i in range(1, 4) for cond in conds] +
                   ['mis_D{}_cond'.format(i) for i in range(1, 4)])

        if lambda_aux > 0:
            headers += ['{}_clsf_D{}'.format(type, i) for type in ['fake', 'real']
                        for i in range(1, 4)]

        # add headers before first log, truncate previous file
        if epoch == 0:
            with open(self.metrics_fn, 'w') as log:
                log.write(','.join(headers) + '\n')

        # .view(-1) results in rows concat'd, e.g. [[1,2], [3,4]] -> [1,2,3,4]
        row = ([epoch] + list(fake_probs[:, inds].view(-1)) +
               list(real_probs[:, inds].view(-1)) + list(mis_probs))

        if lambda_aux > 0:
            row += list(clsf_fake_loss) + list(clsf_real_loss)

        with open(self.metrics_fn, 'a') as log:
            log.write(','.join([str(float(x)) for x in row]) + '\n')

        if self.verbose:
            if init_time is not None:
                headers = ['time'] + headers
                epoch_time = int(time.time() - init_time)
                row = ['{}min{}sec'.format(epoch_time // 60, epoch_time % 60)] + row
            print('\n'.join(['{}: {}'.format(header, metric)
                             for header, metric in zip(headers, row)]))

    def log_sample_images(self, data=None, epoch=-1):
        """Image logger.

        Saves sample images for visual inspection. If data is
        provided, it is considered to be images and is logged.
        Else, sample_embeddings are used to generate images.
        Images are arbitrarily but consistently enumerated, every
        image gets its own directory inside the root_image_dir
        directory, and every epoch inside its own directory as well.

        Args:
            data(list of torch.Tensors): optional, real images
                to be saved.
            epoch(int): number of epoch generator has been trained.
        """

        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to('cpu')),
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),
            transforms.ToPILImage()
        ])

        if data is None:
            # use sample embeddings
            noise = torch.randn(self.sample_embeddings.size(0), self.noise_dim,
                                device=self.device)
            self.stackgan.eval()
            with torch.no_grad():
                images = self.stackgan.generate(self.sample_embeddings, noise)
            self.stackgan.train()

            for j in range(images[0].size(0)): # iterate embeddings

                img_dir = os.path.join(self.root_image_dir, 'image_{}'.format(j), str(epoch))
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                for i, images_scale in enumerate(images): # iterate scales
                    image = transform(images_scale[j])
                    image.save(os.path.join(img_dir, 'scale_{}.jpg'.format(i)))

        else:
            # data are real images
            for j, image in enumerate(data):

                img_dir = os.path.join(self.root_image_dir, 'image_{}'.format(j))
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                image = transform(image)
                image.save(os.path.join(img_dir, 'real.jpg'))

if __name__ == '__main__':
    def train():
        """Function that parses arguments from command line
        (execute file with -h for info) and trains the model.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('-ng', dest='Ng', default=32, type=int,
                            help='N_g as defined in the original paper')
        parser.add_argument('-nd', dest='Nd', default=64, type=int,
                            help='N_d as defined in original paper')
        parser.add_argument('-cd', '--cond_dim', default=128, type=int,
                            help='dimension of final conditioning variable')
        parser.add_argument('-zd', '--noise_dim', type=int, default=100,
                            help='noise dimension')
        parser.add_argument('-dvc', '--device', type=str, default='cuda:0',
                            help='device to run on')
        parser.add_argument('--restore', default=False, action='store_true',
                            help='restore checkpoint with the same hyperparameters')
        parser.add_argument('-dd', '--dataset_dir', type=str, required=True,
                            help='dataset root directory')
        parser.add_argument('-id', '--image_dir', type=str, required=True,
                            help='image directory wrt dataset dir')
        parser.add_argument('-ed', '--emb_dir', type=str, required=True,
                            help='embedding directory wrt dataset dir')
        parser.add_argument('-avc', '--available_classes', type=str, default=None,
                            help='txt to choose subset of classes')
        parser.add_argument('-bs', '--batch_size', type=int, default=64,
                            help='batch size')
        parser.add_argument('--glr', type=float, default=2e-4,
                            help='generator learning rate')
        parser.add_argument('--dlr', type=float, default=2e-4,
                            help='discriminators learning rate')
        parser.add_argument('--kl_coef', type=float, help='coefficient of KLD loss')
        parser.add_argument('--uncond_coef', type=float, default=0,
                            help='coefficient of unconditional losses')
        parser.add_argument('--wrong_coef', type=float, default=0.5,
                            help='coefficient of discriminator fake input losses')
        parser.add_argument('--aux_coef', type=float, default=0.1,
                            help='coefficient of classification loss')
        parser.add_argument('--epochs', type=int, default=600,
                            help=('number of training epochs. can be used'
                                  ' with restore to further train a network'))
        parser.add_argument('-lfp', dest='label_flip_prob', type=float, default=0.,
                            help='prob of switching real and fake labels during a batch')
        parser.add_argument('-ld', '--log_dir', type=str, required=True,
                            help='root directory of logs')
        parser.add_argument('-v', '--verbose', default=False, action='store_true',
                            help='whether to print error metrics during training')
        parser.add_argument('-md', '--model_dir', type=str, required=True,
                            help='directory of model')
        parser.add_argument('-ci', '--checkpoint_interval', type=int, default=30,
                            help='per how many epochs to save model checkpoint')


        args = parser.parse_args()

        gan_kw = dict(Ng=args.Ng, Nd=args.Nd, cond_dim=args.cond_dim,
                      noise_dim=args.noise_dim, device=args.device,
                      restore=args.restore)
        dataset_kw = dict(dataset_dir=args.dataset_dir, image_dir=args.image_dir,
                          embedding_dir=args.emb_dir, batch_size=args.batch_size,
                          available_classes=args.available_classes)
        try:
            kl_coef = args.kl_coef
        except AttributeError:
            kl_coef = 2 / gan_kw['cond_dim']

        optim_kw = dict(glr=args.glr, dlr=args.dlr, beta_1=0.5, kl_coef=kl_coef,
                        uncond_coef=args.uncond_coef, wrong_coef=args.wrong_coef,
                        aux_coef=args.aux_coef,
                        epochs=args.epochs, label_flip_prob=args.label_flip_prob)
        log_kw = dict(log_dir=args.log_dir, n_samples=6, verbose=args.verbose,
                      model_dir=args.model_dir, checkpoint_interval=args.checkpoint_interval)

        trainer = StackGANv2Trainer(gan_kw, dataset_kw, optim_kw, log_kw)
        trainer()

    train()
