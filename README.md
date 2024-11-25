Modified by 《Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts》
and the original repo: https://github.com/shijxcs/LIFT
## Requirements

* Python 3.8
* PyTorch 2.0
* Torchvision 0.15
* Tensorboard

- Other dependencies are listed in [requirements.txt](requirements.txt).

To install requirements, run:

```sh
conda create -n lift python=3.8 -y
conda activate lift
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorboard
pip install -r requirements.txt
```

We encourage installing the latest dependencies. If there are any incompatibilities, please install the dependencies with the following versions.

```
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.2.1
yacs==0.1.8
tqdm==4.64.1
ftfy==6.1.1
regex==2022.7.9
timm==0.6.12
```

## Hardware

Most experiments can be reproduced using a single GPU with 20GB of memory (larger models such as ViT-L require more memory).

- To further reduce the GPU memory cost, gradient accumulation is recommended. Please refer to [Usage](#usage) for detailed instructions.

## Quick Start on the CIFAR-100-LT dataset

```bash
# run LIFT on CIFAR-100-LT (with imbalanced ratio=100)
python main.py -d cifar100_ir100 -m clip_vit_b16 adaptformer True
```

By running the above command, you can automatically download the CIFAR-100 dataset and run the method (LIFT).

## Running on Large-scale Long-tailed Datasets

### Prepare the Dataset

Download the dataset [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018).

Put files in the following locations and change the path in the data configure files in [configs/data](configs/data):


- iNaturalist 2018

```
Path/To/Dataset
└─ train_val2018
   ├─ Actinopterygii
   |  ├─ 2229
   |  |  ├─ 2c5596da5091695e44b5604c2a53c477.jpg
   |  |  └─ ......
   |  └─ ......
   └─ ......
```

### Reproduction

To reproduce the logs result, please run

```bash
# run LIFT on iNaturalist 2018 
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20
```

**Note**: The code is using the "dali_dataset" variable to control the switch for data loading and is using the default PyTorch transform. You can switch to DALI for data loading by commenting out line 81 in trainer.py to enable DALI. Both PyTorch and DALI can run by the command.

Moreover, `[options]` can facilitate modifying the configure options in [utils/config.py](utils/config.py). Following are some examples.

- To specify the root path of datasets, add `root Path/To/Datasets`.

- To change the output directory, add an option like `output_dir NewExpDir`. Then the results will be saved in `output/NewExpDir`.

- To assign a single GPU (for example, GPU 0), add an option like `gpu 0`.