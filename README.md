# Reverse Homology

Reverse homology is a self-supervised protein representation learning method that purposes principles in comparative proteomics as a learning signal for our models. Given a subset of homologous IDRs (obtained automatically), our model is asked to pick out a held-out homolog from the same family, from a large set of sequences where the other sequences are not homologous. This task, which we call reverse homology, requires our model to learn conserved features of IDRs directly from their sequences, in order to distinguish them from non-homologous background sequences.

![Architecture](architecture.png)

A summary of the architecture and training task is shown above. At the top, we show the multiple sequence alignment for the yeast protein Prx1 (as an example - during training, we iterate over all IDRs in all proteins) across 15 yeast species. Conserved resides are highlighted in blue. The yellow dotted line box shows the boundaries of an IDR in Prx1. B) By taking IDRs from different species in the yellow dotted box in A, we construct a set of homologous IDRs, H (shown as yellow lines) C) We sample a subset of IDRs (blue dotted box) from H and use this to construct the query set (blue box). We also sample a single IDR (purple dotted box) from H not used in the query set and add this to the target set (purple box). Finally, we populate the target set with non-homologous IDRs (green), sampled at random from other IDRs from other proteins in the proteome. D) The query set is encoded by the query set encoder g_1. The target set is encoded by the target set encoder g_2. In our implementation, we use a five-layer convolutional neural network architecture. We label convolutional layers with the number of kernels x the number of filters in each layer. Fully connected layers are labeled with the number of filters. E) The output of g_1 is a single representation for the entire query set. In our implementation, we pool the sequences in the query set using a simple average of their representations. The output of g_2 is a representation for each sequence in the target set. The training goal of reverse homology is to learn encoders g_1  and g_2  that produce a large score between the query set representation and the homologous target representation, but not non-homologous targets.

# Data Download

We trained two models, one for yeast IDRs, and another for human IDRs. The data used to train these models (and subsequentially extract features and interpretations from) can be found at the Zenodo link here:

zenodo.org/record/5146063

# Training and Dataset Loading

Training scripts for the respective models are provided in each folder. 

The opts.py file lets you specify training parameters, as well as where to save the model (checkpoint_path) and where to retrive the data (data_path). 

The dataset_repeat_pad.py file contains a Dataset object that loads fasta files into memory, and preprocesses them into one-hot encoding and batches for neural network training. The model_scaled_final.py file contains the architecture, as shown in the above figure. The train.py file includes the training loop.

# Feature Extraction

Scripts for extracting features from trained models are included in the "feature_extraction" folder. For the yeast model, we provide scripts to extract features from the final fully connected and convolutional target encoder layers. For the human model, we provide scripts to extract features from the final convolutional target encoder layer.

# Visualization

Scripts for generating sequence logos and mutational scanning heat/letter maps from trained models are included in the "visualization" folders. 
