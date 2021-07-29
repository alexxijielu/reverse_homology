import os
import opts as opt
import numpy as np
import pandas as pd
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
import logomaker


if __name__ == "__main__":
    # Specify the IDRs of interest (as a list)
    idrs_of_interest = ['YGR159C_idr_1', 'YLR197W_idr_2']    
    weight_path = "../scer_idr_modell/1000_weights.h5"
    outdir = './mutational_scanning/charge_transition/'   # Directory to save maps into

    autoselect_filters = False       # Will automatically take significant (z-score > 3) filters, or else top n
    minimum_significant = 5       # Minimum number of filters to use
    filters_of_interest = [244]     # If autoselect_filters = False, take the specified filters instead (for all IDRs)

    # File of the z-scores to use to auto-retrieve significant features from
    # This is just the feature file, but normalized with mean/stdev
    z_score_file = "../scer_idr_model/c_conv3_features_1000_zscores.txt"
    # For the yeast file, provide a conversion file of start/ends for labeling protein position
    conversion_file = "../final_list_idrs.csv"

    set_hscale = False  # False to automatically scale heat maps to max values
    hscale_value = 5    # If set_hscale = False, the max value to use for the heat map

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
            plt.savefig(curr_outdir + "F" + filter_idx + "_" + filter_type + "_mutation_map.png")
            plt.close()
            filter_averages.append([filter_type + " " + filter_idx, np.sum(curr_map[0:curr_len, :].T, axis=0)])

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

        output = open(curr_outdir + "sums.txt", "w")
        for f in filter_averages:
            output.write(f[0])
            for n in f[1]:
                 output.write("\t" + str(n))
            output.write("\n")



