# Training

You can use `tools/train.py` to train a model.

Here is the full usage of the script:

```shell
python tools/train.py [ARGS]
```
You can define all of your parameters in config file. It will override the arguments parser. I supply a default config file `general_backbone/configs/image_clf_config.py` for your study. To execute with your config file

```
python tools/train.py general_backbone/configs/image_clf_config.py
```

The meaning of config file parameters and arguments parser are the same as below table:


| ARGS      | Type                  |  Description                                                 |
| -------------- | --------------------- |  ----------------------------------------------------------- |
| **Priority config file** |
| `--config` or `-c` | str | Path of config fire for training. Your config file's arguments overrides the arguments parser (default: None) |
| **General Parameters**                                  |
| `--model` | str | Name of model to train (default: "resnet50") |
| `--epochs` | int | Number of epochs to train (default: 300) |
| `--start-epoch` | int | Manual number of start epoch (default: 0)  |
| `--num-classes` | int | Number of categores in image classification task (default: None)| 
| `--pretrained` | action_store:True | Whether using pretrained model (default: False) |
| `--eval-metric` | str | Best metric to evaluate (default: 'top1') |
| **Checkpoint**                                  |
| `--output` | str | Path to output folder (default: current directory folder) |
| `--initial-checkpoint` | str | Initialize model from this checkpoint (default: none) |
| `--checkpoint-hist`| int | Number of checkpoints to keep (default: 10) |
| `--recovery_interval` | int | Number of interval checkpoint to save (default:10) | 
| `--resume`| str | Resume full model and optimizer state from checkpoint (default: none) |
| `--no-resume-opt` | str | prevent resume of optimizer state when resuming model |
| **Logging**                                  |
| `--log-interval` | str | how many batches to wait before logging training status (default:50) |
| `--log-wandb` | str | log training and validation metrics to wandb (default:False) |
| `--local-rank`| int | if equal 0, log information on local (default: 0) |
| **DataLoader & Datset**                                  | 
| `--data-dir` | str | path to root dataset |
| `--img-size` | int | input image size |
| `--batch-size` or `-b` | int | input batch size for training (default: 32) |
| `--num-workers` or `-j` | int | how many training processes to use (default: 4) |
| `--pin-memory` | action_store:True | Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU. |
| `--shuffle` or `-j` | action_store:True | Is shuffle dataset before training |
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