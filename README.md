# Reverse Homology

Reverse homology is a self-supervised protein representation learning method that purposes principles in comparative genomics as a learning signal for our models. Given a subset of homologous IDRs (obtained automatically), our model is asked to pick out a held-out homolog from the same family, from a large set of sequences where the other sequences are not homologous. This task, which we call reverse homology, requires our model to learn conserved features of IDRs directly from their sequences, in order to distinguish them from non-homologous background sequences.

![Architecture](architecture.png)

A summary of the architecture and training task is shown above. A) At the top, we show the multiple sequence alignment for the yeast protein Prx1 (as an example - during training, we iterate over all IDRs in all proteins) across 15 yeast species. Conserved residues are highlighted in blue. The yellow dotted line box shows the boundaries of an IDR in Prx1. B) By taking IDRs from different species in the yellow dotted box in A, we construct a set of homologous IDRs, H (shown as yellow lines) C) We sample a subset of IDRs (blue dotted box) from H and use this to construct the query set (blue box). We also sample a single IDR (purple dotted box) from Hnot used in the query set and add this to the target set (purple box). Finally, we populate the target set with non-homologous IDRs (green), sampled at random from other IDRs from other proteins in the proteome. D) This panel includes detail s that are more specific to our implementation (highlighted in grey). The query set is encoded by the query set encoder g<sub>1</sub>. The target set is encoded by the target set encoder g<sub>2</sub>. In our implementation, we use a five-layer convolutional neural network architecture. We label convolutional layers with the number of kernels x the number of filters in each layer. Fully connected layers are labeled with the number of filters. E) The output of g<sub>1</sub> is a single representation for the entire query set. In our implementation, we pool the sequences in the query set using a simple average of their representations. The output of g<sub>2</sub> is a representation for each sequence in the target set. The training goal of reverse homology is to learn encoders g<sub>1</sub> and g<sub>2</sub> that produce a large score between the query set representation and the homologous target representation, but not non-homologous targets. In our implementation, this is the dot product: g<sub>1</sub> (S<sub>q</sub>) ∙ g<sub>2</sub> (s<sub>t+</sub>) > g<sub>1</sub>(S<sub>q</sub>) ∙ g<sub>2</sub>(s<sub>t-</sub>). After training, we extract features using the target sequence encoder. For this work, we extract the pooled features of the final convolutional layer, as shown by the arrow in D.

# Data Download

We trained two models, one for yeast IDRs, and another for human IDRs. The data used to train these models (and subsequentially extract features and interpretations from) can be found at the Zenodo link here: zenodo.org/record/5146063

# Tutorials

Tutorials on how to train a model, extract and visualize features as sequence logos, and generate specific mutagenesis predictions (which we visualize as "mutational scanning maps") can be found in the /tutorials/ subdirectory. Some toy human IDR data can also be found in this subdirectory, although we recommend you download the full data from the Zenodo link for serious applications.

# Companion Web App

To make browsing of our mutational scanning outputs (see Tutorial 3) more intuitive/convenient, we provide an interactive local webapp in the /mutational_scanning_webapp/ subdirectory. A sample output (the mutational scanning maps for p27-KID) is provided in this directory as well. See the below gif for a preview of how the app works! Note that the app is entirely local on your machine and thus requires you to install additional dependencies listed in /mutational_scanning_webapp/webapp_requirements.txt

![Webapp](webapp_animation.png)
