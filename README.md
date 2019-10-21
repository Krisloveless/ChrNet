# ChrNet: A chromosome-based 1D convolutional neural network for predicting immune cell types
![front image](https://raw.githubusercontent.com/Krisloveless/ChrNet/master/front.png)

This repository is hosting the source code for Keras-based machine learning model **ChrNet**, which stands for **Chr** omosome-based 1D-CNN **net** work.
Our model is flexiable in both input end and output end, capable of retraining.<br />
Input end: Accept different versions of reference genome.<br />
Output end: Accept different output cell types.

### Prerequisites

The code requires python3. When dataset is large, you may force cpu in `main.py` to process the training or predicting, which is recommended when predicting the output.<br />
The default input for model is 24 chromosomes (including *1-22*, *X* and *Y*). Inputs however can be customized by inputting another reference_bed in `main.py` and modify the different sizes of chr_list in `model.py`.

### Output labels

The default output cell types are:
"*Dendritic cell*", "*NK cell*", "*B cell*", "*CD4+ T cell*", "*CD8+ T cell*", "*CD14+ monocyte*" and "*Other*".
We accept customized output cell types for retraining. In `data.py`, change the variable *dict_label* to your own
dictionary labels to alter the cell type output.

### Other functions

Our IntegratedGradients function was acquired from [here](https://github.com/hiranumn/IntegratedGradients). findMetaFeature followed the procedure from [**Improving interpretability of deep learning models: splicing codes as a case study**](https://www.biorxiv.org/content/10.1101/700096v1) -- Anupama Jha, Joseph K. Aicher, Deependra Singh, Yoseph Barash, 2019.


### File struct
<pre>
.
├── data.py              
├── dataset/
│   ├── testing_tpm.tsv
│   └── training_tpm.tsv
├── front.png
├── main.py
├── model.py
├── IntegratedGradients.py
├── pre_trained/
│   └── ChrNet.hdf5
├── README.md
└── reference/
    └── hg19.sorted.bed
</pre>
`data.py`: Storing all functions and tools.<br />
`dataset`: Storing training and testing set.<br />
`main.py`: The main function for ChrNet.<br />
`model.py`: Storing the model.<br />
`pre_trained`: Storing pre_trained weight for ChrNet.<br />
`reference`: Storing the reference bed for model.<br />
