from genericpath import exists
from shutil import Error, ignore_patterns
import numpy as np
import cv2
import os
import shutil
import torch
from torch.utils import data
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from general_backbone.configs import image_clf_config
from general_backbone.utils import ConfigDict, Config

def flip(img, annotation):
    img = np.fliplr(img).copy()
    h, w = img.shape[:2]

    x_min, y_min, x_max, y_max = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    bbox = np.array([w - x_max, y_min, w - x_min, y_max])
    for i in range(len(landmark_x)):
        landmark_x[i] = w - landmark_x[i]

    new_annotation = list()
    new_annotation.append(x_min)
    new_annotation.append(y_min)
    new_annotation.append(x_max)
    new_annotation.append(y_max)

    for i in range(len(landmark_x)):
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return img, new_annotation


def channel_shuffle(img, annotation):
    if (img.shape[2] == 3):
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr)
        img = img[..., ch_arr]
    return img, annotation


def random_noise(img, annotation, limit=[0, 0.2], p=0.5):
    if np.random.random() < p:
        H, W = img.shape[:2]
        noise = np.random.uniform(limit[0], limit[1], size=(H, W)) * 255

        img = img + noise[:, :, np.newaxis] * np.array([1, 1, 1])
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img, annotation


def random_brightness(img, annotation, brightness=0.3):
    alpha = 1 + np.random.uniform(-brightness, brightness)
    img = alpha * img
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_contrast(img, annotation, contrast=0.3):
    coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img = alpha * img + gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_saturation(img, annotation, saturation=0.5):
    coef = np.array([[[0.299, 0.587, 0.114]]])
    alpha = np.random.uniform(-saturation, saturation)
    gray = img * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    img = alpha * img + (1.0 - alpha) * gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_hue(image, annotation, hue=0.5):
    h = int(np.random.uniform(-hue, hue) * 180)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image, annotation


def scale(img, annotation):
    f_xy = np.random.uniform(-0.4, 0.8)
    origin_h, origin_w = img.shape[:2]

    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    h, w = int(origin_h * f_xy), int(origin_w * f_xy)
    image = cv2.resize(img, (h, w),
                   preserve_range=True,
                   anti_aliasing=True,
                   mode='constant').astype(np.uint8)

    new_annotation = list()
    for i in range(len(bbox)):
        bbox[i] = bbox[i] * f_xy
        new_annotation.append(bbox[i])

    for i in range(len(landmark_x)):
        landmark_x[i] = landmark_x[i] * f_xy
        landmark_y[i] = landmark_y[i] * f_xy
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return image, new_annotation


def rotate(img, annotation, alpha=30):

    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,
                                          (img.shape[1], img.shape[0]))

    point_x = [bbox[0], bbox[2], bbox[0], bbox[2]]
    point_y = [bbox[1], bbox[3], bbox[3], bbox[1]]

    new_point_x = list()
    new_point_y = list()
    for (x, y) in zip(landmark_x, landmark_y):
        new_point_x.append(rot_mat[0][0] * x + rot_mat[0][1] * y +
                           rot_mat[0][2])
        new_point_y.append(rot_mat[1][0] * x + rot_mat[1][1] * y +
                           rot_mat[1][2])

    new_annotation = list()
    new_annotation.append(min(new_point_x))
    new_annotation.append(min(new_point_y))
    new_annotation.append(max(new_point_x))
    new_annotation.append(max(new_point_y))

    for (x, y) in zip(landmark_x, landmark_y):
        new_annotation.append(rot_mat[0][0] * x + rot_mat[0][1] * y +
                              rot_mat[0][2])
        new_annotation.append(rot_mat[1][0] * x + rot_mat[1][1] * y +
                              rot_mat[1][2])

    return img_rotated_by_alpha, new_annotation


class WLFWDatasets(data.Dataset):
    def __init__(self, file_list=None, transforms=None, transforms_alb=None, include_gt=True, image=None, target_image_size=(256, 256), is_albumentation=True):
        self.include_gt = include_gt
        self.transforms = transforms
        self.transforms_alb = transforms_alb
        self.debug = False
        self.TARGET_IMAGE_SIZE = target_image_size
        self.is_albumentation = is_albumentation
        if not self.include_gt:
            self.img = image
            self.lines = ["infer image"]
        else:
            self.line = None
            self.path = None
            self.landmarks = None
            self.filenames = None
            self.transforms = transforms
            self.transforms_alb = transforms_alb
            with open(file_list, 'r') as f:
                self.lines = f.readlines()

    def __getitem__(self, index):
        if self.include_gt:
            self.line = self.lines[index].strip().split()
            # print('len self.line: ', len(self.line))
            # print(self.line[0])
            self.img = cv2.imread(self.line[0])
            self.img = cv2.resize(self.img, self.TARGET_IMAGE_SIZE, interpolation = cv2.INTER_LINEAR)
            self.landmark = np.asarray(self.line[1:137], dtype=np.float32)
            if self.debug:
                int_locations = self.landmark.copy()
                int_locations *= self.TARGET_IMAGE_SIZE[0]
                int_locations = int_locations.astype(np.int32).reshape(-1, 2)
                for i in int_locations:
                    cv2.circle(self.img, (i[0], i[1]), 1, (0, 0, 255), 1)
                cv2.imwrite("test/{:03d}.png".format(index), self.img)
                if index == 0:
                    print("landmark index {}: \n {}".format(index, self.landmark))

            if self.transforms_alb:
                if self.is_albumentation:
                    landmark = self.landmark.reshape(-1, 2)*self.TARGET_IMAGE_SIZE[0]
                    transformed = self.transforms_alb(image=self.img, bboxes=[[0, 0, 256, 256]], category_ids=[0], keypoints=landmark)
                    self.img = transformed["image"]
                    self.landmark = transformed["keypoints"]
                    self.landmark = np.array(self.landmark).reshape(-1)/self.TARGET_IMAGE_SIZE[0]
                else:
                    self.img = self.transforms_alb(self.img)
        
            # Preprocessing image
            if self.transforms:
                    self.img = self.transforms(self.img)
            return (self.img, self.landmark)
        else:
            if self.img.shape != (self.TARGET_IMAGE_SIZE[0], self.ARGET_IMAGE_SIZE[0], 3):
                self.img = cv2.resize(self.img, self.TARGET_IMAGE_SIZE, cv2.INTER_LINEAR)
            if self.transforms_alb:
                if self.is_albumentation:
                    landmark = self.landmark.reshape(-1, 2)*self.TARGET_IMAGE_SIZE[0]
                    transformed = self.transforms_alb(image=self.img, bboxes=[[0, 0, 256, 256]], category_ids=[0], keypoints=landmark)
                    self.img = transformed["image"]
                else:
                    self.img = self.transforms_alb(self.img)
            if len(self.img.size()) == 3:
                self.img = torch.unsqueeze(self.img, axis=0)

            # Preprocessing image
            if self.transforms:
                    self.img = self.transforms(self.img)
            return self.img

    def __len__(self):
        return len(self.lines)


class AlbImageDataset(data.Dataset):
    '''
        This dataset that accept to customize your Image training process. Especially that accept Albumentation's transformations
        Args:
            data_dir (str: 'toydata/image_classification'): root data directory
            name_split (str: train): Type of dataset in ['train', 'test', 'validation'].
            transforms_alb (albumentations.Compose : None): a compose of albumentations transformation.
            input_size (tuple: (256, 256)): Input image size
            debug (bool: False): whether save transformation debug image
            dir_debug (str: 'tmp/alb_img_debug'): place to save debug image
            class_2_idx (dict: None): mapping class name to indices. If None, model auto mapping folder classname in dataset.
    '''

    def __init__(self, data_dir='toydata/image_classification', name_split='train', \
            transforms_alb=None, input_size=(256, 256), debug=False, \
            dir_debug = 'tmp/alb_img_debug', class_2_idx=None):
        self.data_dir = data_dir
        self.name_split = name_split
        self.transforms_alb = transforms_alb
        self.input_size = input_size
        self.debug = debug
        self.dir_debug = dir_debug
        self.f_path = os.path.join(self.data_dir, self.name_split)
        self.img_paths = []
        self.labels = []
        for root, dirs, files in os.walk(self.f_path):
            for file in files:
                if file.endswith(('.png', '.jpeg', '.jpg')):
                    label = os.path.basename(root)
                    self.img_paths.append(os.path.join(root, file))
                    self.labels.append(label)
        
        # Generate class to index. If class_2_idx is None, it is defined by labels by default 
        if class_2_idx is None:
            self.class_dict = {}
            start_idx = 0
            for label in set(self.labels):
                if label not in self.class_dict:
                    self.class_dict[label] = start_idx
                    start_idx += 1
        else:
            self.class_dict = class_2_idx
        self.class_ids = list(self.class_dict.values())

        # Mapping labels to labels_index
        self.labels = list(map(lambda x: self.class_dict[x], self.labels))

        # Shuffle dataset
        self.idx_shuf = np.arange(len(self.img_paths))
        np.random.shuffle(self.idx_shuf)
        self.img_paths = [self.img_paths[i] for i in self.idx_shuf]
        self.labels = [self.labels[i] for i in self.idx_shuf]

        # Create debug folder
        if self.debug:
            if not os.path.exists(self.dir_debug):
                os.makedirs(self.dir_debug, exist_ok=True)
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        img = cv2.imread(img_path)
        
        if self.transforms_alb and not isinstance(self.transforms_alb, A.Compose):
            raise TypeError('transform_alb must be an object of A.Compose')

        if self.transforms_alb:
            tran_res = self.transforms_alb(image=img, category_ids=self.class_ids)
            img = tran_res['image']
        else:
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
            img = torch.Tensor(img).permute(2, 0, 1)
        if self.debug:
            if isinstance(img, torch.Tensor):
                img = torchvision.transforms.ToPILImage()(img)
                img.save(os.path.join(self.dir_debug, '{:3d}.png'.format(index)))
            elif isinstance(img, np.array):
                cv2.imwrite(os.path.join(self.dir_debug, '{:3d}.png'.format(index)), img)
            else:
                raise(TypeError, 'Can not save image from type {}'.format(type(img)))

        return (img, label)

    def __len__(self):
        return len(self.img_paths)


class AugmentationDataset(data.Dataset):
    '''
        This dataset that accept to customize your Image training process. Especially that accept Albumentation's transformations
        Args:
            data_dir (str: 'toydata/image_classification'): root data directory
            name_split (str: train): Type of dataset in ['train', 'test', 'validation'].
            config_file (str : ''general_backbone/configs/image_clf_config.py'): a link to config file
            dict_transform (dict:None) : dictionary of transformation
            input_size (tuple: (256, 256)): Input image size
            debug (bool: False): whether save transformation debug image
            dir_debug (str: 'tmp/alb_img_debug'): place to save debug image
            class_2_idx (dict: None): mapping class name to indices. If None, model auto mapping folder classname in dataset.
    '''

    def __init__(self, data_dir='toydata/image_classification', name_split='train', \
            config_file = 'general_backbone/configs/image_clf_config.py', dict_transform=None, \
            input_size=(256, 256), debug=False, dir_debug = 'tmp/alb_img_debug', class_2_idx=None):

        self.data_dir = data_dir
        self.name_split = name_split
        self.input_size = input_size
        
        self.dict_transform = None
        if dict_transform:
            self.dict_transform = dict_transform
        elif config_file:
            cfg = Config.fromfile(config_file)
            self.dict_transform = cfg.data_conf.dict_transform
            self.input_size = cfg.img_size
        
        self.debug = debug
        self.dir_debug = dir_debug
        self.f_path = os.path.join(self.data_dir, self.name_split)
        self.img_paths = []
        self.labels = []
        for root, dirs, files in os.walk(self.f_path):
            for file in files:
                if file.endswith(('.png', '.jpeg', '.jpg')):
                    label = os.path.basename(root)
                    self.img_paths.append(os.path.join(root, file))
                    self.labels.append(label)
        if self.dict_transform:
            self.transforms_alb = self.set_transform(self.dict_transform)
        else:
            self.transforms_alb = None
        # Generate class to index. If class_2_idx is None, it is defined by labels by default 
        if class_2_idx is None:
            self.class_dict = {}
            start_idx = 0
            for label in set(self.labels):
                if label not in self.class_dict:
                    self.class_dict[label] = start_idx
                    start_idx += 1
        else:
            self.class_dict = class_2_idx
        self.class_ids = list(self.class_dict.values())

        # Mapping labels to labels_index
        self.labels = list(map(lambda x: self.class_dict[x], self.labels))

        # Shuffle dataset
        self.idx_shuf = np.arange(len(self.img_paths))
        np.random.shuffle(self.idx_shuf)
        self.img_paths = [self.img_paths[i] for i in self.idx_shuf]
        self.labels = [self.labels[i] for i in self.idx_shuf]

        # Create debug folder
        self._create_debug()

    def set_transform(self, dict_transform):
        step_transforms = [self.init_transform(type_tran) for type_tran in dict_transform]

        transform = A.Compose(
            step_transforms
        )
        return transform

    def init_transform(self, type_tran):
        if type_tran=='SmallestMaxSize':
            tran = A.SmallestMaxSize(**self.dict_transform['SmallestMaxSize'])
        elif type_tran=='Resize':
            tran = A.Resize(**self.dict_transform['Resize'])
        elif type_tran=='ShiftScaleRotate':
            tran = A.ShiftScaleRotate(**self.dict_transform['ShiftScaleRotate'])
        elif type_tran=='RandomCrop':
            tran = A.RandomCrop(**self.dict_transform['RandomCrop'])
        elif type_tran=='RandomResizedCrop':
            tran = A.RandomResizedCrop(**self.dict_transform['RandomResizedCrop'])
        elif type_tran=='RandomBrightnessContrast':
            tran = A.RandomBrightnessContrast(**self.dict_transform['RandomBrightnessContrast'])
        elif type_tran=='Normalize':
            tran = A.Normalize(**self.dict_transform['Normalize'])
        elif type_tran=='ToTensorV2':
            tran = ToTensorV2(**self.dict_transform['ToTensorV2'])
        elif type_tran=='Blur':
            tran = A.Blur(**self.dict_transform['Blur'])
        elif type_tran=='GaussNoise':
            tran = A.GaussNoise(**self.dict_transform['GaussNoise'])
        elif type_tran=='GaussianBlur':
            tran = A.GaussianBlur(**self.dict_transform['GaussianBlur'])
        elif type_tran=='GlassBlur':
            tran = A.GlassBlur(**self.dict_transform['GlassBlur'])
        elif type_tran=='HueSaturationValue':
            tran = A.HueSaturationValue(**self.dict_transform['HueSaturationValue'])
        elif type_tran=='ColorJitter':
            tran = A.ColorJitter(**self.dict_transform['ColorJitter'])
        elif type_tran=='MedianBlur':
            tran = A.MedianBlur(**self.dict_transform['MedianBlur'])
        elif type_tran=='RGBShift':
            tran = A.RGBShift(**self.dict_transform['RGBShift'])
        elif type_tran=='VerticalFlip':
            tran = A.VerticalFlip(**self.dict_transform['VerticalFlip'])
        else:
            raise TypeError('currently we do not support {} augmentation type'.format(type_tran))
        return tran
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        img = cv2.imread(img_path)
        
        if self.transforms_alb and not isinstance(self.transforms_alb, A.Compose):
            raise TypeError('transform_alb must be an object of A.Compose')

        if self.transforms_alb:
            tran_res = self.transforms_alb(image=img, category_ids=self.class_ids)
            img = tran_res['image']
        else:
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
            img = torch.Tensor(img).permute(2, 0, 1)

        if self.debug:
            if isinstance(img, torch.Tensor):
                img = torchvision.transforms.ToPILImage()(img)
                img.save(os.path.join(self.dir_debug, '{:3d}.png'.format(index)))
            elif isinstance(img, np.array):
                cv2.imwrite(os.path.join(self.dir_debug, '{:3d}.png'.format(index)), img)
            else:
                raise(TypeError, 'Can not save image from type {}'.format(type(img)))
        return (img, label)

    def _create_debug(self):
        if self.debug:
            if os.path.exists(self.dir_debug):
                shutil.rmtree(self.dir_debug, ignore_errors=True)
                os.makedirs(self.dir_debug, exist_ok=True)

    def __len__(self):
        return len(self.img_paths)