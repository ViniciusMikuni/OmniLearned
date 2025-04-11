# OmniLearn v2 Repository

## Install

```bash
pip install omnilearned
```

## Data

A few standard datasets can be directly downloaded using the command:

```bash
omnilearned dataloader -d DATASET -f OUTPUT/PATH
```
Datasets available are: top/qg/aspen/atlas/jetclass/h1
If ```--d pretrain``` is used instead, aspen, atlas, jetclass, and h1 datasets will be downloaded. The total size of the pretrain dataset is around 4T so be sure to have enough space available.


## Training:

Single GPU training can be started using:

```bash
omnilearned train  -o ./ --save_tag test --dataset DATASET --path OUTPUT/PATH
```

For multiple GPUs and SLURM you can use the ```train.sh``` example script

```bash
#Inside an interactive session run
./train.sh
```

## Contributing

### Linting
To lint the code, run:

```bash
ruff format .
ruff check --fix .
```