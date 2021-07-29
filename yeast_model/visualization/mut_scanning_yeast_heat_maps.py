import os
import opts as opt
import numpy as np
import random
import csv

import sys
sys.path.append("..")
from model_scaled_final import RHModel
from dataset_repeat_pad import Dataset

import tensorflow as tf
import keras as K
from tensorflow.contrib.keras import layers
from sklearn.preprocessing import LabelEncoder

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["font.size"] = 12
fontsize = 12


def heatmap(data, ax=None, **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    ax.set_yticks([])

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Create color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.5)
    cbar = ax.figure.colorbar(im, cax=cax)

    return im, cbar


if __name__ == "__main__":
    # Specify the IDRs of interest (as a list)
    idrs_of_interest = ['YPL055C_idr_1']
    weight_path = "../scer_idr_model/1000_weights.h5"
    outdir = './mutational_scanning_maps/RG_repeats/'   # Directory to save maps into

    autoselect_filters = False      # Will automatically take significant (z-score > 3) filters, or else top n
    minimum_significant = 5         # Minimum number of filters to use
    filters_of_interest = [321]     # If autoselect_filters = False, take the specified filters instead (for all IDRs)

    # File of the z-scores to use to auto-retrieve significant features from
    # This is just the feature file, but normalized with mean/stdev
    z_score_file = "../scer_idr_model/c_conv3_features_1000_zscores.txt"
    # For the yeast file, provide a conversion file of start/ends for labeling protein position
    conversion_file = "../final_list_idrs.csv"

    set_hscale = True  # False to automatically scale heat maps to max values
    hscale_value = 0.3   # If set_hscale = False, the max value to use for the heat map

    # Create data loader
    datapath = '../sc_idr_alignments/'     # Point this to a directory of fasta IDR files
    n_features = 256
    seq_length = 256

    print("Preparing the dataset...")
    ds = Dataset(opt.map_path, min_count=1)
    ds.add_dataset(datapath, clip_M=True)
    ds.prepare()

    # Preload the model
    model = RHModel().create_model(comp_size=opt.comp_size, ref_size=opt.ref_size, seq_length=ds.seq_length,
                                   bsize=opt.batch_size, training=False)
    model.load_weights(weight_path)
    trimmed_model = tf.keras.Model(inputs=model.get_layer("comp_in").input, outputs=model.get_layer("c_conv3").output)
    reshape_layer = layers.Lambda(lambda t: K.backend.reshape(t, (1, opt.comp_size, seq_length, n_features)))(
        trimmed_model.output)
    intermediate_model = tf.keras.Model(inputs=trimmed_model.input, outputs=[reshape_layer])

    # Open z-score and conversion files
    zscore_f = csv.reader(open(z_score_file), delimiter="\t")
    zscores = np.array([row for row in zscore_f])
    conversion_f = csv.reader(open(conversion_file), delimiter=",")
    conversion = np.array([row for row in conversion_f])

    # Get the correct file IDs for the proteins
    print("Getting IDRs of interest from the dataset object...")
    idrs = tuple(idrs_of_interest)
    file_ids = []
    idr_filters = []
    idr_positions = []
    idr_zscores = []
    for i in range(0, ds.num_files):
        if ds.file_info[i]['name'].split(".")[0] in idrs:
            idr_name = ds.file_info[i]['name'].split(".")[0]

            # Get the start/end positions for the IDR
            conv_index = np.where(conversion[:, 1] == idr_name)[0][0]
            start = conversion[conv_index][2]
            end = conversion[conv_index][3]

            # Get the z-score vector for the IDR
            zscore_index = np.where(zscores[:, 0] == idr_name)[0][0]
            z_vector = zscores[zscore_index][1:].astype(np.float32)
            n_significant = np.sum(z_vector > 3)

            # Get the filters to visualize for that IDR if auto-select is on
            if autoselect_filters:
                if n_significant >= minimum_significant:
                    z_sig = np.argwhere(z_vector > 3).flatten()
                else:
                    z_sig = z_vector.argsort()[-minimum_significant:][::-1]
                idr_filters.append(z_sig)
            else:
                idr_filters.append(filters_of_interest)
            file_ids.append(i)
            idr_positions.append((start, end))

            if autoselect_filters:
                idr_zscores.append(z_vector[z_sig])
            else:
                idr_zscores.append(z_vector[filters_of_interest])

    # Iterate over files
    for i in range(0, len(file_ids)):
        curr_file_id = file_ids[i]
        curr_start = np.int(idr_positions[i][0])
        curr_end = np.int(idr_positions[i][1])
        curr_filters = idr_filters[i]
        curr_zscores = idr_zscores[i]
        curr_name = ds.file_info[curr_file_id]['name'].split(".")[0]
        print ("Working on", curr_name)

        # Create directory to store files into
        curr_outdir = outdir + curr_name + "/"
        if not os.path.exists(curr_outdir):
            os.makedirs(curr_outdir)

        enc = LabelEncoder().fit(['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S',
                                  'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T'])
        amino_acids = enc.inverse_transform(np.arange(0, 20))
        # Sort the amino acids by biochemical properties
        sorted_indices = [5, 15, 16, 19, 1, 13, 11, 8, 14, 6, 2, 3, 0, 17, 9, 7, 12, 18, 4, 10]
        amino_acids = amino_acids[sorted_indices]

        # Get wild-type sequence
        curr_matrix, curr_masks, _, curr_lengths, curr_species = ds.load_all_sequences_with_lengths_and_species(
            curr_file_id)
        species_index = [idx for idx, s in enumerate(curr_species) if "Scer" in s]
        if len(species_index) != 1:
            print("COULDN'T FIND SCER ENTRY FOR", curr_name)
        else:
            curr_encoding = curr_matrix[species_index[0]]
            curr_len = curr_lengths[species_index[0]]
            curr_sequence = enc.inverse_transform(np.argmax(curr_encoding[:curr_len], axis=-1).flatten())
            curr_mask = curr_masks[species_index[0]]

        # Get the wild-type representation
        curr_seq_in = np.expand_dims(curr_encoding, axis=0)
        curr_seq_in = np.expand_dims(curr_seq_in, axis=0)
        curr_seq_in = np.expand_dims(curr_seq_in, axis=0)
        curr_seq_in = np.pad(curr_seq_in, ((0, 0), (0, 0), (0, 399), (0, 0), (0, 0)), mode='constant')
        representations = intermediate_model.predict(curr_seq_in)[0, 0, :, :]

        # Systematically mutate every single amino acid position and populate the change per filter
        avg_map = np.zeros((n_features, 256, 20))
        max_map = np.zeros((n_features, 256, 20))
        for p in range(0, 256):
            curr_window = 0
            while p not in np.arange(curr_len * curr_window, curr_len * curr_window + curr_len):
                curr_window += 1

            curr_seq_in = np.expand_dims(curr_encoding, axis=0)
            curr_seq_in = np.repeat(curr_seq_in, 400, axis=0)

            # Mutate each amino acid at that position
            curr_seq_in[:, p, :] = 0
            for a in range(0, 20):
                curr_seq_in[a, p, a] = 1

            curr_seq_in = np.reshape(curr_seq_in, (1, 1, 400, 256, 20))
            curr_encodings = intermediate_model.predict(curr_seq_in)
            curr_encodings = np.reshape(curr_encodings, (400, 256, 256))

            for a in range(0, 20):
                mut_max = np.max(curr_encodings[a, curr_len * curr_window:curr_len * curr_window + curr_len], axis=0) \
                          - np.max(representations[curr_len * curr_window:curr_len * curr_window + curr_len], axis=0)
                mut_avg = np.mean(curr_encodings[a, curr_len * curr_window:curr_len * curr_window + curr_len], axis=0) \
                          - np.mean(representations[curr_len * curr_window:curr_len * curr_window + curr_len], axis=0)
                max_map[:, p, a] = mut_max.T
                avg_map[:, p, a] = mut_avg.T

        # Create heat maps for the specific indices that we need
        filter_averages = []        # Store the averages across position for each filter
        for f in curr_filters:
            if f < n_features:
                filter_type = "Max"
                filter_idx = str(f)
                curr_map = max_map[f, :, sorted_indices].T
            else:
                filter_type = "Average"
                filter_idx = str(f - n_features)
                curr_map = avg_map[f - n_features, :, sorted_indices].T

            if filter_type == "Max":
                if set_hscale:
                    hscale = hscale_value
                else:
                    hscale = np.max(np.abs(curr_map[0:curr_len, :]))

                xlabels = []
                aa_count = curr_start
                if curr_start == 1:
                    aa_count += 1
                for x in curr_sequence[0:curr_len]:
                    xlabels.append(x + "\n" + str(aa_count))
                    aa_count += 1

                plot = plt.subplots(figsize=(0.5*curr_len, 10))
                fig, ax = plot[0], plot[1]
                im, cbar = heatmap(curr_map[0:curr_len, :].T, ax=ax, cmap="bwr", vmin=-hscale, vmax=hscale, aspect=1.0)
                ax.set_yticks(np.arange(len(amino_acids)))
                ax.set_yticklabels(amino_acids)
                ax.set_xticks(np.arange(curr_len))
                ax.set_xticklabels(xlabels)
                ax.tick_params(axis='both', which='major', labelsize=fontsize)
                plt.savefig(curr_outdir + "F" + filter_idx + "_MAX_mutation_map.png")
                plt.close()

                filter_averages.append([filter_type + " " + filter_idx, np.mean(curr_map[0:curr_len, :].T, axis=0)])

            if filter_type == "Average":
                xlabels = []
                aa_count = curr_start
                for x in curr_sequence[0:curr_len]:
                    xlabels.append(x + "\n" + str(aa_count))
                    aa_count += 1

                if set_hscale:
                    hscale = hscale_value
                else:
                    hscale = np.max(np.abs(curr_map[0:curr_len, :]))
                plot = plt.subplots(figsize=(0.5*curr_len, 10))
                fig, ax = plot[0], plot[1]
                im, cbar = heatmap(curr_map[0:curr_len, :].T, ax=ax, cmap="bwr", vmin=-hscale, vmax=hscale, aspect=1.0)
                ax.set_yticks(np.arange(len(amino_acids)))
                ax.set_yticklabels(amino_acids)
                ax.set_xticks(np.arange(curr_len))
                ax.set_xticklabels(xlabels)
                ax.tick_params(axis='both', which='major', labelsize=fontsize)
                plt.savefig(curr_outdir + "F" + filter_idx + "_AVERAGE_mutation_map.png")
                plt.close()
                filter_averages.append([filter_type + " " + filter_idx, np.mean(curr_map[0:curr_len, :].T, axis=0)])

        # Create a second summary plot of the averages
        xlabels = []
        aa_count = curr_start
        for x in curr_sequence[0:curr_len]:
            xlabels.append(x + "\n" + str(aa_count))
            aa_count += 1

        fig, ax = plt.subplots(figsize=(0.5*curr_len, 10))
        for f in filter_averages:
            ax.plot(np.arange(len(xlabels)), f[1] / np.max(np.abs(f[1])), alpha=0.5, label=f[0])
        ax.legend()
        ax.set_ylim(bottom=-1.0, top=1.0)
        ax.set_xticks(np.arange(curr_len))
        ax.set_xticklabels(xlabels)
        plt.savefig(curr_outdir + "summary.png")
        plt.close()

        zscore_output = open(curr_outdir + "zscores.txt", "w")
        zscore_output.write("Feature\tZ-Score\n")
        for f in range (0, len(curr_filters)):
            if curr_filters[f] < n_features:
                filter_type = "Max"
                filter_idx = str(curr_filters[f])
            else:
                filter_type = "Average"
                filter_idx = str(curr_filters[f] - n_features)
            zscore_output.write(filter_type + " " + filter_idx + "\t" + str(curr_zscores[f]) + "\n")
        zscore_output.close()



