1. Create virtual environment

```
conda create -n gen_backbone python=3.7 -y
conda activate gen_backbone
```

2. Install dependent packages

The dependent packages for `general_backbone` are listed in `requirements.txt`:

```
torch>=1.4.0
torchvision>=0.5.0
pyyaml
huggingface_hub
```

You simply install by running:

```
pip install -r requirements.txt
```

3. Install package:

`general_backbone` package can be installed as below:

```
pip install -v -e .
```

Execute command to test the success of installation:

```
python3 tools/train.py --model resnet50 --data_dir toydata --batch-size 8 --output checkpoint/resnet50
```

Using package:

```
import general_backbone
general_backbone.list_models()
```