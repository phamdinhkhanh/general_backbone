# Copyright (c) general_backbone. All rights reserved.
# --------------------Config for model training------------------------
from logging import debug

from torchvision.transforms.transforms import Resize
from general_backbone import scheduler


train_conf = dict(
    # General config
    model='resnet50',
    epochs=300,
    start_epoch=0,
    pretrained=True,
    num_classes=1000,    
    eval_metric='top1',

    # Checkpoint
    output='checkpoint/resnet50',
    checkpoint_hist=10,
    initial_checkpoint=None,
    resume=None,
    no_resume_opt=False,

    # Logging
    log_interval=50,
    log_wandb=False,
    local_rank=0,

    # DataLoader
    batch_size=16,
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True, 
    shuffle=True,

    # Learning rate
    lr=0.0005,
    lr_noise_pct=0.67,
    lr_noise_std=1.0,
    lr_cycle_mul=1.0,
    lr_cycle_decay=0.1,
    lr_cycle_limit=1.0,
    sched='cosin',
    min_lr=1e-6, 
    warmup_lr=0.0001,
    warmup_epochs=5,
    lr_k_decay=1.0,
    decay_epochs=100,
    decay_rate=0.1,
    patience_epochs=10,
    cooldown_epochs=10,
)


test_conf = dict(
    # Data Loader
    batch_size=16,
    shuffle=False,
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True
)

# --------------------Config for Albumentation Transformation
# You can add to dict_transform a new Albumentation Transformation class with its argument and values:
# Learn about all Albumentation Transformations, refer to link: https://albumentations.ai/docs/getting_started/transforms_and_targets/
# Note: the order in the dictionary is matched with the processive order of transformations
data_root = 'toydata/image_classification'
img_size=224

data_conf=dict(
    dict_transform=dict(
        RandomResizedCrop={'width':256, 'height':256, 'scale':(0.9, 1.0), 'ratio':(0.9, 1.1), 'p':0.5},
        ColorJitter={'brightness':0.35, 'contrast':0.5, 'saturation':0.5, 'hue':0.2, 'always_apply':False, 'p':0.5},
        ShiftScaleRotate={'shift_limit':0.05, 'scale_limit':0.05, 'rotate_limit':15, 'p':0.5},
        RGBShift={'r_shift_limit': 15, 'g_shift_limit': 15, 'b_shift_limit': 15, 'p': 0.5},
        RandomBrightnessContrast={'p': 0.5},
        Normalize={'mean':(0.485, 0.456, 0.406), 'std':(0.229, 0.224, 0.225)},
        Resize={'height':img_size, 'width': img_size},
        ToTensorV2={'always_apply':True}
        ),

    class_2_idx=None, # Dictionary link class with indice. For example: {'dog':0, 'cat':1}, Take the folder name for label If None.
    img_size=img_size,
    data = dict(
        train=dict(
            data_dir=data_root,
            name_split='train',
            is_training=True,
            debug=False, # If you want to debug Augumentation, turn into True
            dir_debug = 'tmp/alb_img_debug', # Directory where to save Augmentation debug
            shuffle=True
            ),
        eval=dict(
            data_dir=data_root,
            name_split='test',
            is_training=False,
            shuffle=False
            )
    )
)