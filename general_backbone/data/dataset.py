""" Quick n Simple Image Folder, Tarfile based DataSet

Copyright general_backbone
"""
from albumentations.augmentations.geometric.resize import SmallestMaxSize
import torch.utils.data as data
import os
import torch
import logging

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image

from .parsers import create_parser

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):
    """A wrapper of image dataset.

    Same as :obj:`torch.utils.data.Dataset`
    Args:
        root : root directory of image dataset
        parser (function: None): function to parse image from tar or folder
        class_map (string: '' ): class names map to indices.
        load_bytes (boolean: False): load image from bytes or image path.
        transform (obj: None) : transformation image technique.
    """

    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):
    """A wrapper of image dataset.

    Same as :obj:`torch.utils.data.IterableDataset`
    Args:
        root : root directory of image dataset
        parser (function: None): function to parse image from tar or folder
        split (string: 'train'): dataset type (train/test/val).
        batch_size (int: None): batch_size of iterable image dataset
        class_map (string: '' ): class names map to indices.
        load_bytes (boolean: False): load image from bytes or image path.
        repeats (int: 0): number of repeating dataset
        transform (obj: None) : transformation image technique.
    """

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            class_map='',
            load_bytes=False,
            repeats=0,
            transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training, batch_size=batch_size, repeats=repeats)
        else:
            self.parser = parser
        self.transform = transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if target is None:
                target = torch.tensor(-1, dtype=torch.long)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes
    Same as :obj:`torch.utils.data.IterableDataset`
    Args:
        dataset (object: Dataset) : object dataset used to train.
        num_splits (int: 2): total time of augumentation is applied
    """

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)

class AugmentationDataset(data.Dataset):
    '''Dataset wrapper to perform Augmentation Dataset
    Same as :obj:`torch.utils.data.Dataset`

    Args:
        dict_transform (dict: None): the dictionary includes the keys are Albumentations Transformation and the values are corresponding parameters.
        images_filepaths (list):  list of all image paths
        transform (albumentations.Compose: None): a compose of transformation.
    '''
    def __init__(self, dict_transform, images_filepaths, transform=None):
        self.dict_transform = dict_transform
        self.images_filepaths = images_filepaths
        self.transform = transform

    @property
    def transform(self):
        return self.transform

    def _set_transforms(self, x):
        
        def init_transform(type_tran):
            if type_tran=='SmallestMaxSize':
                tran = A.SmallestMaxSize(**self.dict_transform['SmallestMaxSize'])
            elif type_tran=='ShiftScaleRotate':
                tran = A.ShiftScaleRotate(**self.dict_transform['ShiftScaleRotate'])
            elif type_tran=='RandomCrop':
                tran = A.RandomCrop(**self.dict_transform['RandomCrop'])
            elif type_tran=='RGBShift':
                tran = A.RGBShift(**self.dict_transform['RGBShift'])
            elif type_tran=='RandomBrightnessContrast':
                tran = A.RandomBrightnessContrast(**self.dict_transform['RandomBrightnessContrast'])
            elif type_tran=='Normalize':
                tran = A.Normalize(**self.dict_transform['Normalize'])
            else:
                tran = exec("A.{}(**self.dict_transform['{}'])".format(type_tran))
            return tran
        
        step_transforms = [init_transform(type_tran) for type_tran in self.dict_transform]

        transform = A.compose(
            step_transforms
        )
        return transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = Image.open(image_filepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image