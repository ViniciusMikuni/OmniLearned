# OmniLearned Official Repository

This repository contains the software package necessary to reproduce all the results presented in the OmniLearned paper, as well as intructions on how to use your own dataset!

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Using the Pre-trained checkpoint](#using-the-pre-trained-checkpoint)


## Installation

```bash
pip install omnilearned
```


## Data

A few standard datasets can be directly downloaded using the command:

```bash
omnilearned dataloader -d DATASET -f OUTPUT/PATH
```

Datasets available from the package are: top, qg, aspen, atlas, jetclass, jetclass2, h1, jetnet150, jetnet30, cms_qcd, cms_bsm, atlas_flav


If ```--dataset pretrain``` is used instead, aspen, atlas, jetclass, jetclass2, cms_qcd, cms_bsm, and h1 datasets will be downloaded. The total size of the pretrain dataset is around 4T so be sure to have enough space available!

These datasets are open and available from elsewhere, please cite the following resources depending on the dataset used:

<details>
<summary><b>Top Tagging Community Dataset: Show BibTeX citation</b></summary>

```bibtex
@article{Kasieczka:2019dbj,
    author = "Butter, Anja and others",
    editor = "Kasieczka, Gregor and Plehn, Tilman",
    title = "{The Machine Learning landscape of top taggers}",
    eprint = "1902.09914",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.21468/SciPostPhys.7.1.014",
    journal = "SciPost Phys.",
    volume = "7",
    pages = "014",
    year = "2019"
}
```

</details> 


<details>
<summary><b>Quark Gluon Dataset: Show BibTeX citation</b></summary>

```bibtex
@article{Komiske:2018cqr,
    author = "Komiske, Patrick T. and Metodiev, Eric M. and Thaler, Jesse",
    title = "{Energy Flow Networks: Deep Sets for Particle Jets}",
    eprint = "1810.05165",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "MIT-CTP 5064",
    doi = "10.1007/JHEP01(2019)121",
    journal = "JHEP",
    volume = "01",
    pages = "121",
    year = "2019"
}
```

</details> 

<details>
<summary><b>ATLAS Top Tagging Dataset: Show BibTeX citation</b></summary>

```bibtex
@article{ATLAS:2024rua,
    author = "Aad, Georges and others",
    collaboration = "ATLAS",
    title = "{Accuracy versus precision in boosted top tagging with the ATLAS detector}",
    eprint = "2407.20127",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    reportNumber = "CERN-EP-2024-159",
    doi = "10.1088/1748-0221/19/08/P08018",
    journal = "JINST",
    volume = "19",
    number = "08",
    pages = "P08018",
    year = "2024"
}
```

</details> 


<details>
<summary><b>H1 DIS Dataset: Show BibTeX citation</b></summary>

```bibtex
@article{Britzger:2021xcx,
    author = "Britzger, Daniel and Levonian, Sergey and Schmitt, Stefan and South, David",
    collaboration = "H1",
    title = "{Preservation through modernisation: The software of the H1 experiment at HERA}",
    eprint = "2106.11058",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    reportNumber = "MPP-2021-87, DESY-21-097",
    doi = "10.1051/epjconf/202125103004",
    journal = "EPJ Web Conf.",
    volume = "251",
    pages = "03004",
    year = "2021"
}

```
</details>


<details>
<summary><b>JetNet Dataset: Show BibTeX citation</b></summary>

```bibtex
@inproceedings{Kansal:2021cqp,
    author = "Kansal, Raghav and Duarte, Javier and Su, Hao and Orzari, Breno and Tomei, Thiago and Pierini, Maurizio and Touranakou, Mary and Vlimant, Jean-Roch and Gunopulos, Dimitrios",
    title = "{Particle Cloud Generation with Message Passing Generative Adversarial Networks}",
    booktitle = "{35th Conference on Neural Information Processing Systems}",
    eprint = "2106.11535",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    month = "6",
    year = "2021"
}

```
</details>


<details>
<summary><b>ATLAS Flavour Tagging Dataset: Show BibTeX citation</b></summary>

```bibtex
@article{ATLAS:2025dkv,
    author = "Aad, Georges and others",
    collaboration = "ATLAS",
    title = "{Transforming jet flavour tagging at ATLAS}",
    eprint = "2505.19689",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    reportNumber = "CERN-EP-2025-103",
    month = "5",
    year = "2025"
}

```
</details>


<details>
<summary><b>Aspen Open Jets, CMS QCD, and BSM Datasets: Show BibTeX citation</b></summary>

```bibtex
@article{Amram:2024fjg,
    author = {Amram, Oz and Anzalone, Luca and Birk, Joschka and Faroughy, Darius A. and Hallin, Anna and Kasieczka, Gregor and Kr{\"a}mer, Michael and Pang, Ian and Reyes-Gonzalez, Humberto and Shih, David},
    title = "{Aspen Open Jets: unlocking LHC data for foundation models in particle physics}",
    eprint = "2412.10504",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "FERMILAB-PUB-24-0941-AD",
    doi = "10.1088/2632-2153/ade58f",
    journal = "Mach. Learn. Sci. Tech.",
    volume = "6",
    number = "3",
    pages = "030601",
    year = "2025"
}

```
</details>


<details>
<summary><b>JetClass Dataset: Show BibTeX citation</b></summary>

```bibtex
@article{Qu:2022mxj,
    author = "Qu, Huilin and Li, Congqiao and Qian, Sitian",
    title = "{Particle Transformer for Jet Tagging}",
    eprint = "2202.03772",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "2",
    year = "2022"
}

```
</details>


<details>
<summary><b>JetClass 2 Dataset: Show BibTeX citation</b></summary>

```bibtex
@article{Li:2024htp,
    author = "Li, Congqiao and others",
    title = "{Accelerating Resonance Searches via Signature-Oriented Pre-training}",
    eprint = "2405.12972",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "FERMILAB-PUB-24-0699-V",
    month = "5",
    year = "2024"
}

```
</details>




## Training:

Examples for different datasets can be found in the ```train.sh``` script. As an example, let's train the small model using the community top tagging dataset

### Get the data

```bash
omnilearned dataloader --dataset top --folder PATH/TO/YOU/STORAGE
```

### Start the training using the small model

```bash
omnilearned train  -o ./ --save-tag test_training --dataset top --path PATH/TO/YOU/STORAGE --size small --epoch 1
```

This command will only train the model for a single epoch.


Similarly, for multiple GPUs and work nodes with SLURM support you can use the ```train.sh``` example script

```bash
#Inside an interactive SLURM session or in your job submission script
./train.sh
```

To train a generative model instead you simply need to change the ```--mode``` flag to ```generator```. For example:

```bash
omnilearned train  -o ./ --save-tag test_training --dataset top --path PATH/TO/YOU/STORAGE --size small --epoch 1 --mode generator
```


## Evaluation

The evaluate script can be used to evaluate the results of the training and to save a file containing the relevant outputs. In the case of classification, a npz file will be created containing the classifier outputs, true labels, and anything saved as part of the "global" features in the dataset file. Let's quickly evaluate the model we just trained:

```bash
omnilearned evaluate  -i ./ -o ./ --save-tag test_training --dataset top --path PATH/TO/YOU/STORAGE --size small
```

You can inspect the npz file generated and quickly calculate any metric, for example:

```bash
import numpy as np
from sklearn.metrics import roc_auc_score

data = np.load("outputs_test_training_top_0.npz")
predictions = data["prediction"]
labels = data["pid"]

auc = roc_auc_score(labels, predictions[:,1])
print(f"AUC: {auc:.4f}")

```

## Using the Pre-trained checkpoint

Even though we provide all the ingredients required to perform the model pre-training we also make the trained checkpoints available, so you can easily fine-tune your own relevant dataset. For example, let's again train a model using the top tagging dataset, but this time we will fine-tune our model.


```bash
omnilearned train  -o ./ --save-tag test_training_fine_tune --dataset top --path PATH/TO/YOU/STORAGE --size small --epoch 1 --fine-tune --pretrain-tag pretrain_s
```

We also provide trained checkpoints for the medium (m) and large (l) models. The evaluation is carried out exactly the same as before, just change the name of the checkpoint to be loaded.

## Creating Your Own Dataset

Instead of using the pre-loaded dataset you can use OmniLearned on your own problem. For this create a folder named ```custom``` where your dataset will be saved. Inside this folder, create the subfolders train/test/val.



## Contributing

### Linting
To lint the code, run:

```bash
ruff format .
ruff check --fix .
```