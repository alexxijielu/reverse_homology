# Tutorial 2: Extracting and Visualizing Features

## Overview

This step of the tutorial is intended to guide users on how to extract
features from pretrained models. Starting with a weights file from a 
trained model (e.g. in the previous tutorial step), a directory of fasta
files containing sequence, and a target species to extract features for,
this tutorial will output a tsv file of features for all of the sequences
in the fasta directory for the given species. Step 3 will further guide on
how to produce a sequence logo for each feature. 

**NOTE:** Currently this tutorial can only be run if you have a tensorflow-enabled
GPU - unfortunately, one of the steps in the architecture has a channel
order that will only work on GPU :(

## 1. Configure parameters in the feature extractor

Under the /feature_extraction/ subdirectory, open the 
"extract_batch_human_features_conv.py" file (or the equivalent yeast
file) in a text editor or an IDE. Change the following variables:
- **datapath (line 23)**: Regular expression for the fasta files containing
sequences you want to extract features from.
- **outfile (line 24)**: tsv to write features into (this will be created
by the Python script)
- **weights_file (line 25)**: Where to load the pretrained weights from
- **speciesname (line 26)**: Which species to extract features from. The code
will look for any fasta sequences that have an identifier (e.g. following the 
'>') with a speciesname as a subsequence. 


## 2. Extracting features
Simply run:
> python extract_batch_human_features_conv.py

This will extract features from the final convolutional layer of the target
sequence encoder, as we did in the paper. To extract features from different layers
or the fully connected layers, see the "Variations" section. 

## 3. Visualizing features as sequence logos

As described in the paper, each feature in the reverse homology model
can be summarized as a sequence logo, that summarize the types of 
subsequences that activate each feature. To generate these logos, 
first edit these variables in the create_average_sequence_logos.py
(for average-pooled features) and the create_max_sequence_logos.py
(for max-pooled features) files in the /visualization/ subdirectory:
- **datapath (line 20)**: Directory for the fasta files to be used in 
calculating these sequence logos (should be the same one you used for
training unless you're doing something special here.)
- **outdir (line 21)**: Directory to save sequence logos into
- **weights_file (line 22)**: Where to load the pretrained weights from
- **species (line 31)**: Which species to use. The code will look for any
fasta sequences that have an identifier (e.g. following the
'>') with a speciesname as a subsequence. We restrict the sequence logo
calculation to one species to avoid biasing the sequence logos by closely
related homologues. 

## Variations

### Extracting features from the fully connected layer

Variations of the scripts that extract features from the final fully connected
layer are in the /feature_extraction/ subdirectory as "extract_batch_human_features_fc.py".
These features are not interpretable with the scripts we provide, but can perform
better for downstream classification/regression/analysis tasks than the convolutional
features.

### Extracting features from other layers

To extract features from non-final layers (e.g. an intermediate layer of either
the convolutional or fully connected layers), modify the "outputs" argument in 
the "trimmed_model" variable (line 43 in the convolutional script, line 46 in 
the fully connected script). This specifies which layer to extract an output from,
and needs to match the name of a layer given in the model_scaled_final.py file. 