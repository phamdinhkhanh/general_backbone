# --------------------Config for model training

from logging import debug
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

    # Optimizer

    
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

data_conf = dict(
    dict_transform = dict(
        SmallestMaxSize={'max_size': 160},
        ShiftScaleRotate={'shift_limit':0.05, 'scale_limit':0.05, 'rotate_limit':15, 'p':0.5},
        RandomCrop={'height':128, 'width':128},
        RGBShift={'r_shift_limit': 15, 'g_shift_limit': 15, 'b_shift_limit': 15, 'p': 0.5},
        RandomBrightnessContrast={'p': 0.5},
        Normalize={'mean':(0.485, 0.456, 0.406), 'std':(0.229, 0.224, 0.225)},
        ToTensorV2=None
        ),
    
    class_2_idx=None, # Dictionary link class with indice. For example: {'dog':0, 'cat':1}, Take the folder name for label If None.
    img_size=224,
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