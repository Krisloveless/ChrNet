# ChrNet: A chromosome-based 1D convolutional neural network for predicting immune cell types in breast cancer

![front image](https://raw.githubusercontent.com/Krisloveless/ChrNet/master/cover.png)

This repository is hosting the source code for Keras-based machine learning model **ChrNet**, which stands for Chromosome-based 1D-CNN network.

### Prerequisites

The code requires python3. When dataset is large, you may force cpu in `main.py` to process the training or predicting, which is recommended when predicting the output.<br />
The default input for model is 24 chromosomes (including *1-22*, *X* and *Y*). Inputs however can be customized by inputting another reference_bed in `main.py` and modify the different size of chr_list in `model.py`.

### Output labels

The default output cell types are:
"*Dendritic cell*", "*NK cell*", "*B cell*", "*CD4 T cell*", "*CD8 T cell*", "*CD14+ monocyte*" and "*Other*".

### File struct
<pre>
.
├── data.py              
├── **dataset**
│   ├── testing_tpm.tsv
│   └── training_tpm.tsv
├── front.png
├── main.py
├── model.py
├── **pre_trained**
│   └── ChrNet.hdf5
├── README.md
└── **reference**
    └── hg19.sorted.bed
</pre>
`data.py`: Storing all functions and tools.<br />
`dataset`: Storing training and testing set.<br />
`main.py`: The main function for ChrNet.<br />
`model.py`: Storing the model.<br />
`pre_trained`: Storing pre_trained weight for ChrNet.<br />
`reference`: Storing the reference bed for model.<br />
