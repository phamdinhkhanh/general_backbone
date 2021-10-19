# Training

You can use `tools/train.py` to train a model.

Here is the full usage of the script:

```shell
python tools/train.py [ARGS]
```

| ARGS      | Type                  |  Description                                                 |
| -------------- | --------------------- |  ----------------------------------------------------------- |
| **Necessary Parameters**                                  |
| `--data-dir`          | str                   |  Path to dataset. Required initialization parameter |
| `--dataset`  or `-d`       | str                   |  dataset type | x 
| `--model`          | str                   |  Name of model to train (default: "resnet50"). Run `general_backbone.list_models` to get all backbones are supported|
| `--epochs` | int | 'number of epochs to train (default: 300)' |
| `--batch-size` or  `-b` | int |  input batch size for training (default: 128) |
| `--pretrained`          | action= store_true                   |  Start with pretrained version of specified network (if avail) |
| `--initial-checkpoint`          | str                   |  Initialize model from this checkpoint (default: none) |
| `--resume`          | str                   | Resume full model and optimizer state from checkpoint (default: none) |
| `--output` | action=store_true | path to output folder where saves model (default: none, current dir) |
| `--num-classes` | int |  number of label classes (default:none, set value according to total classes of dataset) |
| `--img-size` | int |  Image patch size (default: None => model default) |
| `--validation-batch-size` or `-vb` | int |  validation batch size override (default: None) |
| **Optimizer**                                  |
| `--opt` | int |  Optimizer algorithm (default: "sgd") | x 
| `--opt-eps` | float |  Optimizer Epsilon (default: None, use opt default) | 
| `--opt-betas` | float |  Optimizer Betas (default: None, use opt default) | 
| `--momentum` | float |  Optimizer momentum (default: 0.9) |  
| `--weight-decay` | float |  weight decay (default: 2e-5) | 
| `--clip-grad` | float |  Clip gradient norm (default: None, no clipping) |  
| `--clip-mode` | float |  Gradient clipping mode. One of ("norm", "value", "agc") | 
| **Learning rate schedule parameters**                                  |
| `--sched` | str | LR scheduler (default: "cosine") | x 
| `--lr` | float | learning rate (default: 0.05) |
| `--lr-cycle-mul` | float | learning rate cycle len multiplier (default: 1.0) |
| `--lr-cycle-decay` | float | amount to decay each learning rate cycle (default: 0.5) |
| `--lr-cycle-limit` | float | learning rate cycle limit, cycles enabled if > 1 |
| `--lr-k-decay` | float | 'learning rate k-decay for cosine/poly (default: 1.0)' |
| `--warmup-lr` | float | 'warmup learning rate (default: 0.0001)' |
| `--min-lr` | float | 'lower lr bound for cyclic schedulers that hit 0 (1e-5)' |
| `--epoch-repeats` | int | epoch repeat multiplier (number of times to repeat dataset epoch per train epoch). |
| `--start-epoch` | int | manual epoch number (useful on restarts |
| `--decay-epochs` | int | epoch interval to decay LR |
| `--warmup-epochs` | int | epochs to warmup LR, if scheduler supports |
| `--cooldown-epochs` | int | epochs to cooldown LR at min_lr, after cyclic schedule ends |
| `--patience-epochs` | int | patience epochs for Plateau LR scheduler (default: 10) |
| `--decay-rate` or `dr` | float | LR decay rate (default: 0.1) |
| **Augmentation**                                  |
| `--no-aug` | action=store_true | Disable all training augmentation, override other train aug args |
| `--scale` | float | Random resize scale (default: 0.08 1.0) |
| `--ratio` | float | Random resize aspect ratio (default: 0.75 1.33) |
| `--hflip` | float | Horizontal flip training aug probability |
| `--vflip` | float | Vertical flip training aug probability |
| `--color-jitter` | float | Color jitter factor (default: 0.4) |
| `--aug-repeats` | int | Number of augmentation repetitions (distributed training only) (default: 0) |
| `--aug-splits` | int | Number of augmentation splits (default: 0, valid: 0 or >=2) |
| **Loss**                                  |
| `--jsd-loss` | int | Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`. |
| `--bce-loss` | int | Enable BCE loss w/ Mixup/CutMix use. |
| `--bce-target-thresh` | int | Threshold for binarizing softened BCE targets (default: None, disabled) |
| `--mixup` | float | mixup alpha, mixup enabled if > 0. (default: 0.) |
| `--cutmix` | float | cutmix alpha, cutmix enabled if > 0. (default: 0.) |
| `--cutmix-minmax` | float | cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None) |
| `--mixup-prob`| float | Probability of performing mixup or cutmix when either/both is enabled |
| `--mixup-switch-prob` | float | Probability of performing mixup or cutmix when either/both is enabled |
| `--mixup-mode` | str | How to apply mixup/cutmix params. Per "batch", "pair", or "elem" |
| `--mixup-off-epoch` | int | Turn off mixup after this epoch, disabled if 0 (default: 0) |
| `--smoothing` | float | Label smoothing (default: 0.1) |
| **Batch norm parameters only works with gen_efficientnet**                                  |
| `--bn-tf` | action=store_true | Use Tensorflow BatchNorm defaults for models that support it (default: False) |
| `--bn-momentum` | float | BatchNorm momentum override (if not None) |
| `--bn-eps` | float | BatchNorm epsilon override (if not None) |
| `--sync_bn` | action=store_true | Enable NVIDIA Apex or Torch synchronized BatchNorm. |
| `--dist_bn` | str | Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "") |
| `--split-bn` | action=store_true | Enable separate BN layers per augmentation split. |
| **Model Exponential Moving Average** |
| `--model-ema` | action=store_true | Enable tracking moving average of model weights |
| `--model-ema-decay` | float | decay factor for model weights moving average (default: 0.9998) |
| **Setup computer** |
| `--seed` | int | random seed (default: 42) |
| `--checkpoint-hist` | int | number of checkpoints to keep (default: 10) |
| `--workers` or `-j` | int | how many CPU training processes to use (default: 4) |
| `--save-images` | action=store_true | save images of input bathes every log interval for debugging |
| `--amp` | action=store_true | use NVIDIA Apex AMP or Native AMP for mixed precision training |
| `--apex-amp` | action=store_true | Use NVIDIA Apex AMP mixed precision |
| `--native-amp` | action=store_true | Use Native Torch AMP mixed precision |
| `--channels-last` | action=store_true | Use channels_last memory layout |
| `--pin-mem` | action=store_true | Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU. |
| `--eval-metric` | str | evaluation metric (default: "top1") |
| `--torchscript` | action=store_true | convert model torchscript for inference |

