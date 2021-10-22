from general_backbone.data.alb_augment import AlbImageDataset, AlbImageDataset
from general_backbone import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import albumentations as A
from albumentations.pytorch import ToTensorV2

if __name__ == '__main__':
    transforms_alb = A.Compose(
        [   
            A.RandomResizedCrop(width=256, height=256, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.5),
            A.ColorJitter (brightness=0.35, contrast=0.5, saturation=0.5, hue=0.2, always_apply=False, p=0.5),
            A.ShiftScaleRotate (shift_limit_y=(0.05, 0.4), scale_limit=0.25, rotate_limit=30, interpolation=0, border_mode=4, always_apply=False, p=0.2),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            A.Resize(224, 224),
            ToTensorV2()
        ]
    )
    abldataset = AlbImageDataset(data_dir='toydata/image_classification',
                            name_split='train',
                            transforms_alb=transforms_alb,
                            input_size=(256, 256), 
                            debug=True, 
                            dir_debug = 'tmp/alb_img_debug', 
                            class_2_idx=None)

    for i in range(50):
        img, label = abldataset.__getitem__(i)

    
