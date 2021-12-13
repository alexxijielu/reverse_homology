# Tutorial 3.2: Predicting Mutations for your own IDRs

## Overview

The previous step of the tutorial (3.1) showed how to visualize mutation
predictions for p27-kid; this step will show how to produce the files 
necessary for any given IDR of interest.

Users will need a pretrained weights file (e.g. from Tutorial 1), and 
can specify any given IDR of interest by giving a fasta file. Unlike
the code from 3.2, this code is intended to be run on a GPU-enabled server
and not on your local server (unless they're one and the same).

**NOTE:** Currently this tutorial can only be run if you have a tensorflow-enabled
GPU - unfortunately, one of the steps in the architecture has a channel
order that will only work on GPU :(

## Step 1: Configure options 
Open the `create_mutational_scanning.py` file found under
the `/human_model/visualization` directory in a text editor or an IDE
(or it's equivalent version in the yeast directory).

Configure the following options:

**Line 28** - `input_fasta` should be set to a fasta file containing the
sequence of interest as the first sequence:
> input_fasta = '../PATH_TO_FASTA.fasta'

**Line 29** - `weight_path` should be set to the pretrained weights
for the model:
> weight_path = "../human_idr_model/1000_weights.h5"

**Line 30** - `outdir` should be set to where you want to save the
output to:
> outdir = './mutational-scanning-output-for-my-idr/'

**Line 31** - `z_score_file` should be the path of where to find a
file of pre-extracted features to calculate significance from. For
your convenience, we have provided features for the entire proteome
in the repository, but you can sub in your own if you'd like. 
> z_score_file = "../human_idr_model/human_idr_features.txt"

Additional "advanced" options are explained at the end of this tutorial.

## Step 2: Run the code

Run:
> python create_mutational_scanning.py

This should take a few minutes depending on your hardware.

The outputs from this model can be downloaded as a directory to your
local machine and used to run the create_website.py code used in Tutorial
3.1 to visualize the sequences. Happy exploring!

## Advanced Options

Besides the basic options:
* **species** (line 35): Lets you specify which species from a fasta
file to use instead of just using the first sequence by default. This
is useful if you want to use the data we provide you with in the Zenodo
archive, instead of having to create your own fasta file.
* **start, remove_M** (line 36, 37): 
By default, the code labels the amino acids starting from 0. If you want
to start the count at another number (e.g. if the sequence is in the middle
of a protein and you want the heat map labels to reflect that), just put
the position here. Also, if you want to remove a methonine at the start
of a sequence (like we did in the paper), you can specify it here assuming
you didn't already do it manually in the fasta file. 
* **autoselect_features** (line 39): Set to true if you want the code
to automatically decide which features are significant enough to visualize,
set to false if you want to manually specify.
* **minimum_significant** (line 40): If autoselect_features is true, if
there are less than this number of features above a z-score of 3 relative
to the proteome (or whatever file you gave to z_score_file), then the code
will pick the top `minimum_significant` number of features instead.
* **filters_of_interest** (line 41): If autoselect_features is false, you
must specify which features insetad. Directly specify which features you want to use here by index.
Note that average features are indiced as their filter number + 256 (so
Average Feature 0 is 256, 1 is 257, etc...)
* **save_webapp** (line 43): If you hate interactive webapps, you can 
have the code just save everything as png files instead. Set this to false.
The other options (line 43-45) scale the appearance of the heat map, as
explained by the comments. 
