# Tutorial 1: Training a model

## Overview
This tutorial is intended to guide users on how to train a new 
reverse homology model from scratch, starting from a directory of 
fasta files containing homologous groups of sequences. The output
of this tutorial is a file containing the weights of a trained model,
which will be used in subsequent tutorials to extract features from
IDRs globally for downstream analyses and to make predictions about
mutations in specific IDRs.

Note that these tutorials use the convolutional neural
network (CNN) model we employed in our paper. **We do not believe this
model reflects the best or final architecture for feature discovery in
IDRs.** This was simply a lightweight model that made development of
the method easier due to its rapid training, and interpretation easier
due to the kinds of post-hoc interpretation techniques we were aware
of for CNNs specifically. We hope that future work extends our 
method by using architectures that may better distill distal features
(e.g. LSTM or Transformer models), but provide this tutorial as a 
reasonable approach for current analyses.

## 0. Preliminary set-up
Requirements for python packages are specified in the requirements.txt
file - make sure these are installed before running this tutorial! We
also strongly recommend a GPU configured to be used by Tensorflow 
during training. 

## 1. Preparing Training Data
To train a reverse homology model, the user needs to provide a 
directory of fasta files. Each fasta file should contain a set of
sequences considered homologous to each other. **These sequences
do not need to be aligned: any gap symbols ("-") will be removed as
part of the preprocessing.**

**A toy dataset is provided for illustrative purposes under 
tutorials/toy_homology_sets/**

Some considerations for training data:
- A large number of sets of homologues is important, because 
reverse homology requires a large number of negative samples during
training. If you are working with a smaller number of set of 
homologues, consider reducing the "comp_size" parameter under opts.py,
which controls the number of negative samples used.
- There is a trade-off in choosing what species to include in each
set of homologues: species that are too close in evolutionary 
distance may have sequences that are too similar to provide much
learning signal, but species that are too far may result in sequences
where the function has diverged (and thus challenge the base assumption. 
of our method that the sequence might diverge, but the function won't)
For our experiments, we chose yeast species for our yeast model, and
vertebrates for our human model. 
- Note that any sequences with non-canonical amino acids will be 
filtered out - this behavior is hard-coded, but we provide tips for
changing it in the Advanced Configuration section. 

## 2. Configuring parameters for training
Before training, configure options in the opts.py file by directly
editing these variables in the file:
- **comp_size**: How many sequences to use in the target set? In
our paper we set this to 400, but you may need to set this lower
if you don't have many homology families (since negative samples are
drawn from within the entire dataset provided.)
- **ref_size**: How many sequences to use for the query set? We
set this to 8, but if you don't have many species per homologue set,
you should set this lower. 
- **min_count**: Any homologue sets with fewer species than this 
number will be filtered out before training. The minimum this should
be set to is ref_size + 1, otherwise there won't be enough sequences
for training (since we need to hold out at least one homologous 
sequence from the query set during training.)
- **batch_size, learning_rate, and epochs**: Hyperparameters for 
training the neural network. Guidelines for tuning these can be 
found in most deep learning tutorials. 
- **checkpoint_path**: Directory to save the weights of the
model (and other metadata) to
- **data_path**: Directory to load training data from (this should
be the directory you prepared in the previous step.)
- **save_period**: Controls how frequently the model is saved
  (e.g. the default setting of 200 will save a weights file every
200 epochs.)

(More advanced changes to the training, like changing the sampling 
of negative examples from random uniform behavior, require direct
altering of the dataset_repeat_pad.py file - we provide an overview of some
common changes that can be made to this file in the "Advanced 
Configuration" section.)

## 3. Training a model

After these options are configured, all you need to do is run the 
training script!

> python train.py

The model will train over however many epochs you specified, saving
a weights file to the directory you specified in data_path. The 
training script will also save the loss, accuracy, precision, and recall
on the training dataset over epochs in a csv file in data_path, which 
can help in choosing which weights file to use. These weight files will
be used to extract features and produce mutation predictions for IDRs
in the subsequent tutorials. 

## Advanced Configuration

The opts.py file allows you to reproduce what we did in the paper,
but more advanced users may want to make more extensive changes. 
We provide an overview of some ways the code can be modified. 

###  Changes to the Architecture
Changes to the architecture can be made in the model_scaled_final.py
file. The "create_model" function is where the architecture should be
stored, in Keras. The architecture you specify should inputs and 
outputs as follows if you want it to be compatiable with the rest of
the code:

**Inputs:**
- **comp_in**: The target set. Must be in shape (1, comp_size, seq_length, 20)
- **ref_in**: The query set. Must be in shape (batch_size, ref_size, seq_length, 20)
- **comp_mask, ref_mask**: These are not used in the current implementation,
but provided in case you want to preprocess the sequences by adding padding
tokens instead of just repeating the amino acid sequence as we did in our paper.
These are binary masks that will be multiplied against the positions of the final
convolutional layer, so if you set any padded positions to 0, this will prevent them from
contributing activations before pooling. 

**Output:**
- Needs to be in shape (1, bsize, comp_size) - will output a vector of size
comp_size for each element in the batch (bsize), which reflects the probability
of each element in the target set (comp_size) being the homologue.

### Changes to Sampling the Contrastive Sets

By default, the model samples negative IDRs for the contrastive set 
with uniform probability, but this can be changed in line 160 of the 
dataset_repeat_pad.py file (under the load_list function). For 
computational efficiency, the code currently calculates how many 
samples to select from each of the fasta files as a vector (so it 
only needs to open up the files we are actually sampling sequences
from). Thus, to more selectively choose IDRs from specific fasta
files, you simply need to modify the vector passed to the negative_files
variable in line 165. 

### Non-Canonical Amino Acids and Size Filters

Amino acids are hard-coded in the Dataset object. To add non-canonical
amino acids, simply modify the list passed to the scikit-learn LabelEncoder
in line 11 (in the function initializing the object). You will also need to 
change any np.eye(20) references to however number of amino acids you have now
throughout the code. 

Currently the model removes any sequences with undetermined amino acids ("X")
or below 5 amino acids long. This can be modified in line 92 (i.e. the if statement
filtering these sequences out).

