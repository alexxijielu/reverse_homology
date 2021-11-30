import sys
sys.path.append("..")
import os
import opts as opt
import numpy as np
import csv

from model_scaled_final import RHModel

import tensorflow as tf
import keras as K
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logomaker
import pandas as pd

matplotlib.rcParams.update({'font.size': 22})
from Bio import SeqIO

if __name__ == "__main__":
    save_webapp = True

    input_fasta = '../hs_idr_alignments/HUMAN06525_182to372.fasta'
    remove_M = False
    species = 'HUMAN'
    start = 182
    end = 372

    weight_path = "../human_idr_model/1000_weights.h5"
    outdir = './HUMAN06525_182to372-mutational-scanning/'  # Directory to save maps into

    autoselect_filters = True  # Will automatically take significant (z-score > 3) filters, or else top n
    minimum_significant = 5  # Minimum number of filters to use
    filters_of_interest = [482]  # If autoselect_filters = False, take the specified filters instead (for all IDRs)

    # File of features from proteome to generate z-scores to select significant features from
    z_score_file = "../human_idr_model/human_idr_features.h5"

    # Options only for if save_webapp is False (scales the PNGs)
    set_hscale = False  # False to automatically scale heat maps to max values
    hscale_value = 5  # If set_hscale = False, the max value to use for the heat map
    wscale = 1

    # Preload the model
    n_features = 256
    seq_length = 256
    model = RHModel().create_model(comp_size=opt.comp_size, ref_size=opt.ref_size, seq_length=seq_length,
                                   bsize=opt.batch_size, training=False)
    model.load_weights(weight_path)
    trimmed_model = tf.keras.Model(inputs=model.get_layer("comp_in").input, outputs=model.get_layer("c_conv3").output)
    reshape_layer = layers.Lambda(lambda t: K.backend.reshape(t, (1, opt.comp_size, seq_length, n_features)))(
        trimmed_model.output)
    intermediate_model = tf.keras.Model(inputs=trimmed_model.input, outputs=[reshape_layer])

    # Open z-score and conversion files
    zscore_f = csv.reader(open(z_score_file), delimiter="\t")
    zscores = np.array([row for row in zscore_f])[:, 1:].astype(np.float32)

    # Get the correct file IDs for the proteins
    print("Extracting features from input fasta...")
    enc = LabelEncoder().fit(['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S',
                              'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T'])
    idr_filters = []  # Store filters to use
    idr_zscores = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        if species in record.id:
            curr_name = input_fasta.split("/")[-1].split(".")[0]
            curr_seq = str(record.seq)
            curr_seq = curr_seq.replace("-", "")
            curr_len = len(curr_seq)

            # Preprocess sequence
            if remove_M:
                if len(curr_seq) > 0:
                    if curr_seq[0] == 'M':
                        curr_seq = curr_seq[1:]

            if len(curr_seq) > seq_length:
                curr_seq = curr_seq[:seq_length // 2] + curr_seq[len(curr_seq) - seq_length // 2:]
            else:
                while len(curr_seq) < seq_length:
                    curr_seq = curr_seq + curr_seq
                if len(curr_seq) > seq_length:
                    curr_seq = curr_seq[:seq_length]
            curr_seq_int = enc.transform(list(curr_seq))
            curr_seq_onehot = np.eye(20)[curr_seq_int]
            input_sequence = np.expand_dims(curr_seq_onehot, axis=0)
            input_sequence = np.expand_dims(input_sequence, axis=0)
            input_sequence = np.expand_dims(input_sequence, axis=0)
            input_sequence = np.pad(input_sequence, ((0, 0), (0, 0), (0, opt.comp_size - 1),
                                                     (0, 0), (0, 0)), mode='constant')
            representation = np.squeeze(intermediate_model.predict(input_sequence))[0, :, :]
            max_representation = np.max(representation, axis=0)
            avg_representation = np.average(representation, axis=0)
            representation = np.concatenate([max_representation, avg_representation])

            z_vector = (representation - np.mean(zscores, axis=0)) / np.std(zscores, axis=0)
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

            if autoselect_filters:
                idr_zscores.append(z_vector[z_sig])
            else:
                idr_zscores.append(z_vector[filters_of_interest])

            print("Systematically mutating the IDR...")
            curr_filters = idr_filters[0]
            curr_zscores = idr_zscores[0]

            # Create directory to store files into
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            amino_acids = ['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S', 'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N',
                           'D', 'T']
            enc = LabelEncoder().fit(amino_acids)
            amino_acids = enc.inverse_transform(np.arange(0, 20))
            # Sort the amino acids by biochemical properties
            sorted_indices = [5, 15, 16, 19, 1, 13, 11, 8, 14, 6, 2, 3, 0, 17, 9, 7, 12, 18, 4, 10]
            amino_acids = amino_acids[sorted_indices]

            # Get wild-type sequence
            curr_encoding = curr_seq_onehot
            curr_sequence = enc.inverse_transform(np.argmax(curr_seq_onehot, axis=-1).flatten())

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
                    mut_max = np.max(curr_encodings[a, curr_len * curr_window:curr_len * curr_window + curr_len],
                                     axis=0) \
                              - np.max(representations[curr_len * curr_window:curr_len * curr_window + curr_len],
                                       axis=0)
                    mut_avg = np.mean(curr_encodings[a, curr_len * curr_window:curr_len * curr_window + curr_len],
                                      axis=0) \
                              - np.mean(representations[curr_len * curr_window:curr_len * curr_window + curr_len],
                                        axis=0)
                    max_map[:, p, a] = mut_max.T
                    avg_map[:, p, a] = mut_avg.T

            if save_webapp:
                # Create heat maps for the specific indices that we need
                filter_averages = []  # Store the averages across position for each filter
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
                    aa_count = start
                    if start == 1 and remove_M == True:
                        aa_count += 1
                    for x in curr_sequence[0:curr_len]:
                        xlabels.append(x + "\n" + str(aa_count))
                        aa_count += 1

                    if filter_type == "Max":
                        np.save(outdir + "F" + filter_idx + "_max_heat_map.npy", curr_map[0:curr_len, :])
                    if filter_type == "Average":
                        np.save(outdir + "F" + filter_idx + "_average_heat_map.npy", curr_map[0:curr_len, :])
                    np.save(outdir + "aa_labels.npy", np.array(xlabels))

                    curr_heatmap = curr_map[0:curr_len, :]
                    magnitudes = np.sum(-curr_heatmap, axis=1)
                    probs = np.zeros_like(curr_heatmap)
                    for p in range(0, len(magnitudes)):
                        m = magnitudes[p]
                        if m >= 0:
                            curr_pos = curr_heatmap[p]
                            curr_pos = (curr_pos - np.min(curr_pos)) / (np.max(curr_pos) - np.min(curr_pos)) * 10
                            probs[p] = np.exp(curr_pos) / np.sum(np.exp(curr_pos))
                        else:
                            curr_pos = curr_heatmap[p]
                            curr_pos = (curr_pos - np.min(curr_pos)) / (np.max(curr_pos) - np.min(curr_pos)) * 10
                            probs[p] = np.exp(-curr_pos) / np.sum(np.exp(-curr_pos))
                    probs = np.nan_to_num(probs)
                    # Calculate the final size of each letter
                    letter_scale = probs * magnitudes[:, None]
                    df = pd.DataFrame(letter_scale, columns=amino_acids)

                    fig, ax = plt.subplots()
                    fig.set_size_inches(curr_len * 0.25, 5)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.set_yticklabels([str(-x) for x in ax.get_yticks()])
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax.set_ylabel('Favorability (A.U)')
                    ax.set_xlabel('Position')
                    logo = logomaker.Logo(df, ax=ax, font_name='Verdana', color_scheme="dmslogo_custom",
                                          stack_order="small_on_top", flip_below=False)
                    plt.savefig(outdir + "F" + filter_idx + "_" + filter_type.lower() + "_letter_map.png",
                                bbox_inches="tight")
                    plt.close()

                # Create a second summary plot of the averages
                zscore_output = open(outdir + "zscores.txt", "w")
                zscore_output.write("Feature\tZ-Score\n")
                for f in range(0, len(curr_filters)):
                    if curr_filters[f] < n_features:
                        filter_type = "Max"
                        filter_idx = str(curr_filters[f])
                        zscore_output.write("F" + filter_idx + "_max\t" + str(curr_zscores[f]) + "\n")
                    else:
                        filter_type = "Average"
                        filter_idx = str(curr_filters[f] - n_features)
                        zscore_output.write("F" + filter_idx + "_average\t" + str(curr_zscores[f]) + "\n")
                zscore_output.close()
            else:
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

                # Create heat maps for the specific indices that we need
                filter_averages = []  # Store the averages across position for each filter
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
                        aa_count = start
                        if start == 1:
                            aa_count += 1
                        for x in curr_sequence[0:curr_len]:
                            xlabels.append(x + "\n" + str(aa_count))
                            aa_count += 1

                        plot = plt.subplots(figsize=(0.5 * curr_len, 10))
                        fig, ax = plot[0], plot[1]
                        im, cbar = heatmap(curr_map[0:curr_len, :].T, ax=ax, cmap="bwr", vmin=-hscale, vmax=hscale,
                                           aspect=1.0)
                        ax.set_yticks(np.arange(len(amino_acids)))
                        ax.set_yticklabels(amino_acids)
                        ax.set_xticks(np.arange(curr_len))
                        ax.set_xticklabels(xlabels)
                        ax.tick_params(axis='both', which='major', labelsize=12)
                        plt.savefig(outdir + "F" + filter_idx + "_MAX_letter_map.png")
                        plt.close()

                        filter_averages.append(
                            [filter_type + " " + filter_idx, np.mean(curr_map[0:curr_len, :].T, axis=0)])

                    if filter_type == "Average":
                        xlabels = []
                        aa_count = start
                        for x in curr_sequence[0:curr_len]:
                            xlabels.append(x + "\n" + str(aa_count))
                            aa_count += 1

                        if set_hscale:
                            hscale = hscale_value
                        else:
                            hscale = np.max(np.abs(curr_map[0:curr_len, :]))
                        plot = plt.subplots(figsize=(0.5 * curr_len, 10))
                        fig, ax = plot[0], plot[1]
                        im, cbar = heatmap(curr_map[0:curr_len, :].T, ax=ax, cmap="bwr", vmin=-hscale, vmax=hscale,
                                           aspect=1.0)
                        ax.set_yticks(np.arange(len(amino_acids)))
                        ax.set_yticklabels(amino_acids)
                        ax.set_xticks(np.arange(curr_len))
                        ax.set_xticklabels(xlabels)
                        ax.tick_params(axis='both', which='major', labelsize=12)
                        plt.savefig(outdir + "F" + filter_idx + "_AVERAGE_letter_map.png")
                        plt.close()
                        filter_averages.append(
                            [filter_type + " " + filter_idx, np.mean(curr_map[0:curr_len, :].T, axis=0)])

                zscore_output = open(outdir + "zscores.txt", "w")
                zscore_output.write("Feature\tZ-Score\n")
                for f in range(0, len(curr_filters)):
                    if curr_filters[f] < n_features:
                        filter_type = "Max"
                        filter_idx = str(curr_filters[f])
                    else:
                        filter_type = "Average"
                        filter_idx = str(curr_filters[f] - n_features)
                    zscore_output.write(filter_type + " " + filter_idx + "\t" + str(curr_zscores[f]) + "\n")
                zscore_output.close()

                filter_averages = []  # Store the averages across position for each filter
                for f in curr_filters:
                    if f < n_features:
                        filter_type = "Max"
                        filter_idx = str(f)
                        curr_map = max_map[f, :, sorted_indices].T
                    else:
                        filter_type = "Average"
                        filter_idx = str(f - n_features)
                        curr_map = avg_map[f - n_features, :, sorted_indices].T

                    # Calculate the magnitude of the letters at each position
                    window = curr_map[0:curr_len, :]
                    magnitudes = np.sum(-window, axis=1)

                    # Calculate the probability according to whether it improves or decreases at each position
                    probs = np.zeros_like(window)
                    for p in range(0, len(magnitudes)):
                        m = magnitudes[p]
                        if m >= 0:
                            curr_pos = window[p]
                            curr_pos = (curr_pos - np.min(curr_pos)) / (np.max(curr_pos) - np.min(curr_pos)) * 10
                            probs[p] = np.exp(curr_pos) / np.sum(np.exp(curr_pos))
                        else:
                            curr_pos = window[p]
                            curr_pos = (curr_pos - np.min(curr_pos)) / (np.max(curr_pos) - np.min(curr_pos)) * 10
                            probs[p] = np.exp(-curr_pos) / np.sum(np.exp(-curr_pos))
                    probs = np.nan_to_num(probs)

                    # Calculate the final size of each letter
                    letter_scale = probs * magnitudes[:, None]
                    df = pd.DataFrame(letter_scale, columns=amino_acids)

                    fig, ax = plt.subplots()
                    fig.set_size_inches(curr_len * 0.5, 10)
                    logo = logomaker.Logo(df, ax=ax, font_name='Verdana',
                                          stack_order="small_on_top", flip_below=False)
                    plt.savefig(outdir + "F" + filter_idx + "_" + filter_type + "_mutation_map.png")
                    plt.close()
                    filter_averages.append([filter_type + " " + filter_idx, np.sum(curr_map[0:curr_len, :].T, axis=0)])

                zscore_output = open(outdir + "zscores.txt", "w")
                zscore_output.write("Feature\tZ-Score\n")
                for f in range(0, len(curr_filters)):
                    if curr_filters[f] < n_features:
                        filter_type = "Max"
                        filter_idx = str(curr_filters[f])
                    else:
                        filter_type = "Average"
                        filter_idx = str(curr_filters[f] - n_features)
                    zscore_output.write(filter_type + " " + filter_idx + "\t" + str(curr_zscores[f]) + "\n")
                zscore_output.close()



