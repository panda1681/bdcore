# BDCore
Code for KBS Manuscript "Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling"

## Requirements

* Python (tested on 3.8.8)
* [PyTorch](http://pytorch.org/) (tested on 1.8.0)
* overrides
* tqdm

## Training and Evaluation
### NYT
Train the model on NYT with the following steps:

1. Download the dataset from this [link](https://drive.google.com/file/d/1LgqeJS7M4Tbopu_2XbPecN-hi4tbP5_S/view?usp=sharing), and unzip *.rar into ./data/nyt/
2. python main.py --mode train --exp nyt_wdec

