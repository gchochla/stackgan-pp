"""Dataset."""

# pylint: disable=no-member

import pickle
import os
import glob
from PIL import Image

import pandas as pd
import torch
from torchvision.transforms import transforms

from stackgan.utils.conf import CACHE_DIR

class CUBDatasetEager(torch.utils.data.Dataset):
    """CUB dataset.

    Class for CUB Dataset with precomputed embeddings. Loads
    images at once during initialization. If your RAM cannot
    sustain such a load, use CUBDatasetLazy instead. If training
    dataset, mismatching image is also returned.

    Attributes:
        embeddings(torch.Tensor): embeddings of captions.
        images(list of PIL Images): images corresponding to each
            embedding (at the same index).
        class_ids(list): class of image and embeddings (at the
            same index).
        train(bool): whether this is the training dataset.
        synthetic_ids(dict): correspondence between
            real and synthetic IDs. Meant for use
            during testing.
        transform(torchvision Transform): transform applied to
            every PIL image (after boundind box).
    """

    def __init__(self, dataset_dir, image_dir, embedding_dir,
                 available_classes=None, train=None, use_bbox=True):
        """Init.

        Args:
            dataset_dir(str): root directory of dataset.
            image_dir(str): image directory w.r.t. dataset_dir.
            embedding_dir(str): embedding directory w.r.t
                dataset_dir.
            available_classes(str, optional): txt file to define
                restrict the classes used from the predefined
                train/test split, default=`None`.
            train(bool, optional): indicating whether training
                on this dataset. If not provided, it is determined
                by the embedding_dir name.
            use_bbox(bool): whether to crop using bboxes,
                default=`True`.
        """

        #########################################
        ########## parse pickle files ###########
        #########################################

        embeddings_fn = os.path.join(dataset_dir, embedding_dir,
                                     'char-CNN-RNN-embeddings.pickle')
        with open(embeddings_fn, 'rb') as emb_fp:
            # latin1 enc hack bcs pickle compatibility issue between python2 and 3
            embeddings = torch.tensor(pickle.load(emb_fp, encoding='latin1'))  # pylint: disable=not-callable

        class_ids_fn = os.path.join(dataset_dir, embedding_dir,
                                    'class_info.pickle')
        with open(class_ids_fn, 'rb') as cls_fp:
            # latin1 enc hack bcs pickle compatibility issue between python2 and 3
            class_ids = pickle.load(cls_fp, encoding='latin1')

        img_fns_fn = os.path.join(dataset_dir, embedding_dir,
                                  'filenames.pickle')
        with open(img_fns_fn, 'rb') as fns_fp:
            # latin1 enc hack bcs pickle compatibility issue between python2 and 3
            img_fns = pickle.load(fns_fp, encoding='latin1')

        ####################################################
        ####################################################

        if available_classes:  # if available_classes is set
                               # keep only them
            # get class ids used in dataset
            with open(os.path.join(dataset_dir, available_classes), 'r') as avcls:
                available_class_ids = [int(line.strip().split('.')[0])
                                       for line in avcls.readlines()]

            idcs = [i for i, cls_id in enumerate(class_ids) if cls_id in available_class_ids]

            self.embeddings = embeddings[idcs]
            image_filenames = [img_fns[i] for i in idcs]
            self.class_ids = [cls_id for cls_id in class_ids if cls_id in available_class_ids]

        else: # if available_classes is not set, keep them all
            self.embeddings = embeddings
            image_filenames = img_fns
            self.class_ids = class_ids

        unique_ids = set(self.class_ids)
        self.synthetic_ids = dict(zip(unique_ids, range(len(unique_ids))))

        bboxes = _load_bboxes(dataset_dir)

        # crop to bbox, make 3 channels if grayscale
        if use_bbox:
            load_transform = transforms.Compose([
                transforms.Lambda(lambda x: _bbox_crop(*x)),
                transforms.Lambda(lambda x: transforms.Grayscale(3)(x) if _is_grayscale(x) else x),
                transforms.Resize(304)
            ])
        else:
            load_transform = transforms.Compose([
                # x[0] to get rid of bbox
                transforms.Lambda(lambda x: transforms.Grayscale(3)(x[0])
                                  if _is_grayscale(x[0]) else x[0]),
                transforms.Resize(304)
            ])

        self.transform = transforms.Compose([
            transforms.RandomRotation(2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.images = []
        for img_fn in image_filenames:
            self.images.append(
                load_transform(
                    (Image.open(os.path.join(dataset_dir, image_dir, img_fn + '.jpg')),
                     bboxes[img_fn])
                )
            )

        if train is not None:
            self.train = train
        else:
            # if train is not set, `embedding_dir` should be embeddings_{train, test}
            self.train = (embedding_dir.split('_')[1] == 'train')

    def __len__(self):
        """Return len of dataset.

        Returns:
            Number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """Returns an image, its embedding, and maybe
        a mismatching image.

        Retrieve an image, one of its embeddings and,
        if this is a training dataset, a mismatching image.
        Class ID is last returned value.

        Arguments:
            idx(int): index.

        Returns:
            An image as a torch.Tensor of size (3,256,256),
            if training a mismatching image and one of its
            embeddings, and its class id (synthetic if testing).
        """

        image = self.transform(self.images[idx])

        if not self.train:
            return image, self.synthetic_ids[self.class_ids[idx]]

        rand_caption = torch.randint(10, (1,)).item()
        embedding = self.embeddings[idx, rand_caption]

        while True:
            # get an image from a different class (match-aware discr)
            mis_idx = torch.randint(len(self), (1,)).item()
            if self.class_ids[idx] != self.class_ids[mis_idx]:
                break

        mis_image = self.transform(self.images[mis_idx])

        return image, mis_image, embedding, self.synthetic_ids[self.class_ids[idx]]


    def embeddings_by_class(self):
        """Fetches the embeddings per class.

        Yields:
            torch.Tensor with embeddings of size
            (#, 10, 1024) and the corresponding
            int synthetic ID.
        """

        prev = 0

        while True:
            curr_id = self.class_ids[prev]

            for curr in range(prev + 1, len(self)):
                if self.class_ids[curr] != curr_id:
                    break  # break at first point where id changes

            if curr == prev:  # handle case with one instance in class
                yield self.embeddings[prev][None, ...], self.synthetic_ids[curr_id]
            else:
                yield self.embeddings[prev:curr], self.synthetic_ids[curr_id]

            prev = curr

            if curr == len(self) - 1:
                break

class CUBDatasetLazy(torch.utils.data.Dataset):
    """CUB dataset.

    Class for CUB Dataset with precomputed embeddings. Reads
    images constantly with PIL and doesn't load them at once.
    To load and keep them as an attribute, use CUBDatasetEager.
    If training dataset, mismatching image is also returned.

    Attributes:
        embeddings(torch.Tensor): embeddings of captions.
        image_filenames(list): filename of image corresponding
            to each embedding (at the same index).
        class_ids(list): class of image and embeddings (at the
            same index).
        dataset_dir(str): directory of data.
        image_dir(str): directory of actual images relative to
            dataset_dir.
        train(bool): whether this is the training dataset.
        synthetic_ids(dict): correspondence between
            real and synthetic IDs. Meant for use
            during testing.
        bboxes(dict): keys are the filenames
            of images and values the bounding box to
            retain a proper image to body ratio.
        transform(torchvision Transform): transform applied to every PIL image
            (as is read from image_dir).
    """

    def __init__(self, dataset_dir, image_dir, embedding_dir,
                 available_classes=None, train=None, use_bbox=True):
        """Init.

        Args:
            dataset_dir(str): root directory of dataset.
            image_dir(str): image directory w.r.t. dataset_dir.
            embedding_dir(str): embedding directory w.r.t
                dataset_dir.
            available_classes(str, optional): txt file to define
                restrict the classes used from the predefined
                train/test split, default=`None`.
            train(bool, optional): indicating whether training
                on this dataset. If not provided, it is determined
                by the embedding_dir name.
            use_bbox(bool): whether to crop using bboxes,
                default=`True`.
        """

        #########################################
        ########## parse pickle files ###########
        #########################################

        embeddings_fn = os.path.join(dataset_dir, embedding_dir,
                                     'char-CNN-RNN-embeddings.pickle')
        with open(embeddings_fn, 'rb') as emb_fp:
            # latin1 enc hack bcs pickle compatibility issue between python2 and 3
            embeddings = torch.tensor(pickle.load(emb_fp, encoding='latin1'))  # pylint: disable=not-callable

        class_ids_fn = os.path.join(dataset_dir, embedding_dir,
                                    'class_info.pickle')
        with open(class_ids_fn, 'rb') as cls_fp:
            # latin1 enc hack bcs pickle compatibility issue between python2 and 3
            class_ids = pickle.load(cls_fp, encoding='latin1')

        img_fns_fn = os.path.join(dataset_dir, embedding_dir,
                                  'filenames.pickle')
        with open(img_fns_fn, 'rb') as fns_fp:
            # latin1 enc hack bcs pickle compatibility issue between python2 and 3
            img_fns = pickle.load(fns_fp, encoding='latin1')

        ####################################################
        ####################################################

        if available_classes:  # if available_classes is set
                               # keep only them
            # get class ids used in dataset
            with open(os.path.join(dataset_dir, available_classes), 'r') as avcls:
                available_class_ids = [int(line.strip().split('.')[0])
                                       for line in avcls.readlines()]

            idcs = [i for i, cls_id in enumerate(class_ids) if cls_id in available_class_ids]

            self.embeddings = embeddings[idcs]
            self.image_filenames = [img_fns[i] for i in idcs]
            self.class_ids = [cls_id for cls_id in class_ids if cls_id in available_class_ids]

        else:  # if available_classes is not set, keep them all
            self.embeddings = embeddings
            self.image_filenames = img_fns
            self.class_ids = class_ids

        unique_ids = set(self.class_ids)
        self.synthetic_ids = dict(zip(unique_ids, range(len(unique_ids))))

        self.dataset_dir = dataset_dir
        self.image_dir = image_dir

        if train is not None:
            self.train = train
        else:
            # if train is not set, `embedding_dir` should be embeddings_{train, test}
            self.train = (embedding_dir.split('_')[1] == 'train')

        self.bboxes = _load_bboxes(dataset_dir)

        common_postfix = transforms.Compose([
            transforms.Resize(304),
            transforms.RandomRotation(2),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        if use_bbox:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: _bbox_crop(*x)),
                transforms.Lambda(lambda x: transforms.Grayscale(3)(x) if _is_grayscale(x) else x),
                common_postfix,
            ])
        else:
            self.transform = transforms.Compose([
                # x[0] to get rid of bbox
                transforms.Lambda(lambda x: transforms.Grayscale(3)(x[0])
                                  if _is_grayscale(x[0]) else x[0]),
                common_postfix,
            ])

    def __len__(self):
        """Return len of dataset.

        Returns:
            Number of images in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Returns an image, its embedding, and maybe
        a mismatching image.

        Retrieve an image, if this is a training dataset
        one of its embeddings and a mismatching image.
        Class ID is last returned value.

        Arguments:
            idx(int): index.

        Returns:
            An image as a torch.Tensor of size (3,256,256),
            if training a mismatching image and one of its
            embeddings, and its class id.
        """

        image_fn = self.image_filenames[idx]
        image = Image.open(os.path.join(self.dataset_dir, self.image_dir, image_fn + '.jpg'))

        if not self.train:
            return (self.transform((image, self.bboxes[image_fn])),
                    self.synthetic_ids[self.class_ids[idx]])

        rand_caption = torch.randint(10, (1,)).item()
        embedding = self.embeddings[idx, rand_caption]

        while True:
            # get an image from a different class (match-aware discr)
            mis_idx = torch.randint(len(self), (1,)).item()
            if self.class_ids[idx] != self.class_ids[mis_idx]:
                break

        mis_image_fn = self.image_filenames[mis_idx]
        mis_image = Image.open(os.path.join(self.dataset_dir, self.image_dir,
                                            mis_image_fn + '.jpg'))

        return (self.transform((image, self.bboxes[image_fn])),
                self.transform((mis_image, self.bboxes[mis_image_fn])),
                embedding, self.synthetic_ids[self.class_ids[idx]])

    def embeddings_by_class(self):
        """Fetches the embeddings per class.

        Yields:
            torch.Tensor with embeddings of size
            (#, 10, 1024) and the corresponding
            int synthetic ID.
        """

        prev = 0

        while True:
            curr_id = self.class_ids[prev]

            for curr in range(prev + 1, len(self)):
                if self.class_ids[curr] != curr_id:
                    break  # break at first point where id changes

            if curr == prev:  # handle case with one instance in class
                yield self.embeddings[prev][None, ...], self.synthetic_ids[curr_id]
            else:
                yield self.embeddings[prev:curr], self.synthetic_ids[curr_id]

            prev = curr

            if curr == len(self) - 1:
                break

class CUBTextDatasetLazy(torch.utils.data.Dataset):
    """CUB dataset with actual text instead of
    representations.

    Class for CUB Dataset with captions. Reads images constantly
    with PIL and doesn't load them at once.

    Attributes:
        filenames(list): filename of images and texts alike.
        class_ids(list): class of image and embeddings (at the
            same index).
        dataset_dir(str): directory of data.
        image_dir(str): directory of actual images relative to
            dataset_dir.
        text_dir(str): directory of texts relative to
            dataset_dir.
        train(bool): whether this is the training dataset.
        synthetic_ids(dict): correspondence between
            real and synthetic IDs. Meant for use
            during testing.
        bboxes(dict): keys are the filenames
            of images and values the bounding box to
            retain a proper image to body ratio.
        transform(torchvision Transform): transform applied
            to every PIL image (as is read from image_dir).
    """

    def __init__(self, dataset_dir, image_dir, text_dir, available_classes, use_bbox=True):
        """Init.

        Args:
            dataset_dir(str): root directory of dataset.
            image_dir(str): image directory w.r.t. dataset_dir.
            text_dir(str): embedding directory w.r.t
                dataset_dir.
            available_classes(str): txt file to define
                restrict the classes used from the predefined
                train/test split.
            use_bbox(bool): whether to crop using bboxes,
                default=`True`.
        """

        self.dataset_dir = dataset_dir
        self.image_dir = image_dir
        self.text_dir = text_dir

        # get class ids used in dataset
        with open(os.path.join(dataset_dir, available_classes), 'r') as avcls:
            available_class_ids = [int(line.strip().split('.')[0])
                                   for line in avcls.readlines()]

        class_ids = []
        filenames = []
        for cls_dir in os.listdir(os.path.join(dataset_dir, image_dir)):
            if int(cls_dir.split('.')[0]) in available_class_ids:
                cls_imgs = os.listdir(os.path.join(dataset_dir, image_dir, cls_dir))
                filenames.extend(
                    [os.path.join(os.path.join(cls_dir, os.path.splitext(cls_img)[0]))
                     for cls_img in cls_imgs]
                )
                class_ids.extend([int(cls_dir.split('.')[0])] * len(cls_imgs))

        self.class_ids = class_ids
        self.filenames = filenames
        unique_ids = set(self.class_ids)
        self.synthetic_ids = dict(zip(unique_ids, range(len(unique_ids))))

        self.train = available_classes.startswith('train')

        self.bboxes = _load_bboxes(dataset_dir)

        common_postfix = transforms.Compose([
            transforms.Resize(304),
            transforms.RandomRotation(2),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        if use_bbox:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: _bbox_crop(*x)),
                transforms.Lambda(lambda x: transforms.Grayscale(3)(x) if _is_grayscale(x) else x),
                common_postfix,
            ])
        else:
            self.transform = transforms.Compose([
                # x[0] to get rid of bbox
                transforms.Lambda(lambda x: transforms.Grayscale(3)(x[0])
                                  if _is_grayscale(x[0]) else x[0]),
                common_postfix,
            ])

    def __len__(self):
        """Returns len of dataset.

        Returns:
            Number of images in the dataset.
        """
        return len(self.filenames)


    def __getitem__(self, idx):
        """Returns an image, its caption, and a mismatching image.

        Retrieves an image, one of its captions and,
        if this is a training dataset, a mismatching image.
        Class ID is last returned value.

        Arguments:
            idx(int): index.

        Returns:
            An image as a torch.Tensor of size (3,256,256),
            if training a mismatching image and one of its
            captions of size (70, 201), and its class id.
        """

        image_fn = self.filenames[idx]
        image = Image.open(os.path.join(self.dataset_dir, self.image_dir,
                                        image_fn + '.jpg'))

        if not self.train:
            return (self.transform((image, self.bboxes[image_fn])),
                    self.synthetic_ids[self.class_ids[idx]])

        rand_caption = torch.randint(10, (1,)).item()
        with open(os.path.join(self.dataset_dir, self.text_dir, image_fn + '.txt'), 'r') as txt_fp:
            text = txt_fp.readlines()[rand_caption].strip().lower()
        text = process_text(text)

        while True:
            # get an image from a different class (match-aware discr)
            mis_idx = torch.randint(len(self), (1,)).item()
            if self.class_ids[idx] != self.class_ids[mis_idx]:
                break

        mis_image_fn = self.filenames[mis_idx]
        mis_image = Image.open(os.path.join(self.dataset_dir, self.image_dir,
                                            mis_image_fn + '.jpg'))

        return (self.transform((image, self.bboxes[image_fn])),
                self.transform((mis_image, self.bboxes[mis_image_fn])),
                text, self.synthetic_ids[self.class_ids[idx]])

    def captions_by_class(self):
        """Fetches the captions per class.

        Yields:
            torch.Tensor with captions of size
            (#, 10, 70, 201) and the corresponding
            int synthetic ID.
        """
        cls_dirs = {os.path.split(filename)[0] for filename in self.filenames}
        for cls_dir in cls_dirs:
            texts = []
            for filename in os.listdir(os.path.join(self.dataset_dir, self.text_dir, cls_dir)):
                if not filename.endswith('.txt'):
                    continue
                with open(os.path.join(self.dataset_dir, self.text_dir, cls_dir, filename), 'r') \
                    as txt_fp:
                    texts.append(torch.stack([process_text(text.strip().lower())
                                              for text in txt_fp.readlines()]))
            yield torch.stack(texts), self.synthetic_ids[int(cls_dir.split('.')[0])]

def process_text(text):
    """Transform np array of ascii codes to one-hot sequence."""

    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
    ohvec = torch.zeros(len(alphabet), 201)

    ohvec[[alphabet.index(tok) for tok in text][:201], range(min(len(text), 201))] = 1

    return ohvec

def _load_bboxes(dataset_dir):
    """Retrieve bounding boxes.

    Builds a dictionary of {filename: bounding_box} pairs
    to crop images to 75% body to image ratio.

    Args:
        dataset_dir: Dataset directory.

    Returns:
        A dictionary of
        class_dir/image_filename (without extension): list of bbox coordinates
        key-value pairs.
    """

    # id 4xcoords
    df_bboxes = pd.read_csv(os.path.join(dataset_dir, 'bounding_boxes.txt'),
                            delim_whitespace=True, header=None).astype(int)
    # id fn
    df_corr_fns = pd.read_csv(os.path.join(dataset_dir, 'images.txt'),
                              delim_whitespace=True, header=None)

    bbox_dict = {
        os.path.splitext(df_corr_fns.iloc[i][1])[0]: df_bboxes.iloc[i][1:].tolist()
        for i in range(len(df_bboxes))
    }

    return bbox_dict

def _bbox_crop(image, bbox):
    """Crop PIL.Image according to bbox.

    Args:
        image(PIL.Image): image to crop
        bbox(iterable): iterable with 4 elements.

    Returns:
        Cropped image.
    """

    width, height = image.size
    ratio = int(max(bbox[2], bbox[3]) * 0.75)
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    y_low = max(0, center_y - ratio)
    y_high = min(height, center_y + ratio)
    x_low = max(0, center_x - ratio)
    x_high = min(width, center_x + ratio)
    image = image.crop([x_low, y_low, x_high, y_high])

    return image

def _is_grayscale(image):
    """Return if image is grayscale.

    Assert if image only has 1 channel.

    Args:
        image(PIL.Image): image to check.

    Returns:
        bool indicating whether image is grayscale.
    """

    try:
        # channel==1 is 2nd channel
        image.getchannel(1)
        return False
    except ValueError:
        return True

class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset for synthetic samples.

    Dataset to store and retrieve synthetic samples rather than
    holding them all in RAM. Only cares for the samples it
    was used to store. Stores with sequential filenames akin
    to indices for trivial indexing.

    Attributes:
        dataset_dir(str): dataset directory.
        n_samples(int): number of samples.
        sample_key(str): key of dictionary used to store
            and retrieve samples.
        label_key(str): key of dictionary used to store
            and retrieve corresponding labels.
        template_fn(str): formattable filename for each
            different sample.
    """

    def __init__(self, dataset_dir=None):
        """Init.

        Args:
            dataset_dir(str, optional): directory of dataset,
                default=directory 'dataset' under the
                invisible-to-git cache directory specified
                in configuration file.
        """

        if dataset_dir is None:
            dataset_dir = os.path.join(CACHE_DIR, 'dataset')

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        self.dataset_dir = dataset_dir
        self.n_samples = 0
        self.sample_key = 'sample'
        self.label_key = 'label'
        self.fn_template = os.path.join(dataset_dir, 'sample_{}.pt')

    @classmethod
    def existing(cls, dataset_dir=None):
        """Init from existing directory.

        Args:
            dataset_dir(str, optional): directory of dataset,
                default=directory 'dataset' under the
                invisible-to-git cache directory specified
                in configuration file.
        """

        obj = cls(dataset_dir)
        obj.n_samples = len(os.listdir(obj.dataset_dir))
        return obj

    def _getitem(self, idx):
        """__getitem__ but only for ints.

        Args:
            idx(int): index.

        Returns:
            torch.Tensors sample and label.
        """

        if idx < - len(self) or idx >= len(self):
            raise IndexError('Index {} out of range'.format(idx))

        if idx < 0:
            idx += len(self)

        # saved as cpu tensors
        sample_dict = torch.load(self.fn_template.format(idx))
        return sample_dict[self.sample_key], sample_dict[self.label_key]

    def __getitem__(self, idx):
        """Loads and returns a sample and its label.

        Args:
            idx(int|slice|torch.Tensor|list): index/indices
                of sample(s).

        Returns:
            torch.Tensors sample(s) and label(s).
        """

        if torch.is_tensor(idx):
            if idx.ndim == 0:
                idx = idx.item()
            else:
                idx = list(idx.numpy())

        if isinstance(idx, int):
            return self._getitem(idx)

        if isinstance(idx, slice):
            # slice (for kNN etc)
            samples, labels = [], []
            for i in range(*idx.indices(len(self))):
                sample, label = self._getitem(i)
                samples.append(sample)
                labels.append(label)

            if not samples:
                raise IndexError('No elements corresponding to {}'.format(idx))

            return torch.stack(samples), torch.stack(labels)

        if isinstance(idx, list):
            samples, labels = [], []
            for i in idx:
                sample, label = self._getitem(i)
                samples.append(sample)
                labels.append(label)

            if not samples:
                raise IndexError('No elements corresponding to {}'.format(idx))

            return torch.stack(samples), torch.stack(labels)

        raise IndexError('Unhandled index type')

    def __len__(self):
        """Returns number of stored samples."""
        return self.n_samples

    def save_pairs(self, samples, label):
        """Saves sample-label pairs.

        Saves pairs of samples and their corresponding label
        (assumed to be the same for all samples, thus only an
        integer is expected) with a filename specified by the
        template and order of receival.

        Args:
            samples(torch.tensor): batch of samples.
            label(int): their corresponding label.
        """

        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)  # pylint: disable=not-callable

        samples = samples.cpu()
        label = label.cpu()

        sample_dict = {self.label_key: label}

        for i in range(samples.size(0)):
            sample_dict[self.sample_key] = samples[i]
            torch.save(sample_dict, self.fn_template.format(self.n_samples))
            self.n_samples += 1

class SyntheticImageDataset(torch.utils.data.Dataset):
    """Dataset for synthetic images.

    Dataset to store and retrieve synthetic images with
    JPEG compression rather than holding them all in RAM.
    Only cares for the images it was used to store.
    Stores with sequential filenames akin to indices for
    trivial indexing.

    Attributes:
        dataset_dir(str): dataset directory.
        n_samples(int): number of samples.
        template_fn(tuple of strs): formattable filename for
            each different image. Tuple because 1st dim uniquely
            identifies an image, 2nd dim encodes label.
        save_transform(torchvision Transform): tensor transform
            to save as image.
        load_transform(torchvision Transform): image transform
            to load as tensor (& data augmentation).
    """

    def __init__(self, dataset_dir=None):
        """Init.

        Args:
            dataset_dir(str, optional): directory of dataset,
                default=directory 'dataset' under the
                invisible-to-git cache directory specified
                in configuration file.
        """

        if dataset_dir is None:
            dataset_dir = os.path.join(CACHE_DIR, 'dataset')

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        self.dataset_dir = dataset_dir
        self.n_samples = 0
        self.fn_template = os.path.join(dataset_dir, 'image_{}_'), '{}.jpg'
        self.save_transform = transforms.Compose([
            transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
            transforms.ToPILImage(),
        ])
        self.load_transform = transforms.Compose([
            transforms.Resize(304),
            transforms.RandomRotation(2),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    @classmethod
    def existing(cls, dataset_dir=None):
        """Init from existing directory.

        Args:
            dataset_dir(str, optional): directory of dataset,
                default=directory 'dataset' under the
                invisible-to-git cache directory specified
                in configuration file.
        """

        obj = cls(dataset_dir)
        obj.n_samples = len(os.listdir(obj.dataset_dir))
        return obj

    def _getitem(self, idx):
        """__getitem__ but only for ints.

        Args:
            idx(int): index.

        Returns:
            torch.Tensors image and label.
        """

        if idx < - len(self) or idx >= len(self):
            raise IndexError('Index {} out of range'.format(idx))

        if idx < 0:
            idx += len(self)

        # saved as cpu tensors
        image_fn = glob.glob(self.fn_template[0].format(idx) + '*')[0]  # certain there is only 1
        image = self.load_transform(Image.open(image_fn))
        return image, int(image_fn.split('_')[-1].split('.')[0])

    def __getitem__(self, idx):
        """Loads and returns a sample and its label.

        Args:
            idx(int|slice|torch.Tensor|list): index/indices
                of sample(s).

        Returns:
            torch.Tensors image(s) and label(s).
        """

        if torch.is_tensor(idx):
            if idx.ndim == 0:
                idx = idx.item()
            else:
                idx = list(idx.numpy())

        if isinstance(idx, int):
            return self._getitem(idx)

        if isinstance(idx, slice):
            # slice (for kNN etc)
            samples, labels = [], []
            for i in range(*idx.indices(len(self))):
                sample, label = self._getitem(i)
                samples.append(sample)
                labels.append(label)

            if not samples:
                raise IndexError('No elements corresponding to {}'.format(idx))

            return torch.stack(samples), torch.stack(labels)

        if isinstance(idx, list):
            samples, labels = [], []
            for i in idx:
                sample, label = self._getitem(i)
                samples.append(sample)
                labels.append(label)

            if not samples:
                raise IndexError('No elements corresponding to {}'.format(idx))

            return torch.stack(samples), torch.stack(labels)

        raise IndexError('Unhandled index type')

    def __len__(self):
        """Returns number of stored images."""
        return self.n_samples

    def save_pairs(self, samples, label):
        """Saves sample-label pairs.

        Saves pairs of images and their corresponding label
        (assumed to be the same for all samples, thus only an
        integer is expected) with a filename specified by the
        template and order of receival.

        Args:
            samples(torch.tensor): batch of samples.
            label(int): their corresponding label.
        """

        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)  # pylint: disable=not-callable

        samples = samples.cpu()
        label = label.cpu()

        image_fn_template = self.fn_template[0] + self.fn_template[1].format(label)

        for i in range(samples.size(0)):
            self.save_transform(samples[i]).save(image_fn_template.format(self.n_samples))
            self.n_samples += 1
