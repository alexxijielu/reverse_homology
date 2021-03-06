import opts as opt
import numpy as np
import csv
import os

import tensorflow as tf
import sys
sys.path.append("..")
from model_scaled_final import RHModel
from dataset_repeat_pad import Dataset
from sklearn.preprocessing import LabelEncoder
from tensorflow.contrib.keras import layers

import seqlogo
import pandas as pd
import keras as K

if __name__ == "__main__":
    datapath = '../sc_idr_alignments/'      # Directory of yeast fasta files
    outdir = './max_features-yeast/'        # Where to save sequence logos
    weights_file = "../scer_idr_modell/1000_weights.h5"
    seq_length = 256
    window_size = 15                        # Receptive field of neural network
    percent_filter = 0.7                    # How much of maximum activation percent to include in PFM
    threshold = 20                          # Minimum number of sequences to use if threshold is too stringent
    n_filters = 256
    b_size = 64
    n_features = 256
    scer_only = True                        # Calculate using S. cer sequences only and not other species

    '''For features that have lots of sequences at the start/end, some positions can end up having only a few residues 
    PFM compared to other positions. This parameter prevents these positions from artificially having a lot of 
    information content, by adding even PFM counts if a position has fewer letters than other positions '''
    scale_filters = True

    print("Preparing the dataset...")
    ds = Dataset(opt.map_path)
    ds.min_count = 1        # Set min count to one to include sequences that were filtered out for training
    ds.add_dataset(datapath, clip_M=True)
    ds.prepare()
    print(len(ds.file_info), ds.min_count)

    print("Loading the model...")
    model = RHModel().create_model(comp_size=opt.comp_size, ref_size=opt.ref_size, seq_length=seq_length,
                                   bsize=opt.batch_size, training=False)
    model.load_weights(weights_file)
    trimmed_model = tf.keras.Model(inputs=model.get_layer("comp_in").input,
                                   outputs=model.get_layer("c_conv3").output)
    reshape_layer = layers.Lambda(lambda t: K.backend.reshape(t, (1, 1, opt.comp_size, seq_length, n_features)))(
        trimmed_model.output)
    intermediate_model = tf.keras.Model(inputs=trimmed_model.input, outputs=[reshape_layer])

    enc = LabelEncoder().fit(
        ['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S', 'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T'])

    # First pass: Iterate through all sequences and store maximum activations
    maximums = np.zeros(n_filters)
    values = [[] for x in range(0, n_filters)]
    for j in range(0, len(ds.file_info)):
        curr_matrix, curr_masks, name = ds.load_all_sequences(j)
        if scer_only:
            curr_matrix = np.expand_dims(curr_matrix[0], axis=0)
            curr_masks = np.expand_dims(curr_masks[0], axis=0)

        print("Calculating maximums with ", name)
        n_chunks = np.int(np.floor(curr_matrix.shape[0] // opt.comp_size) + 1)

        # Iterate through all sequence in fasta file
        cells = []
        sequences = []
        for i in range(0, n_chunks):
            if i == n_chunks - 1:
                current_batch = curr_matrix[i * opt.comp_size:]
                current_sequences = enc.inverse_transform(np.argmax(current_batch, axis=-1).flatten()). \
                    reshape(current_batch.shape[0], seq_length)

                current_batch_masks = curr_masks[i * opt.comp_size:]
                pad_length = opt.comp_size - current_batch.shape[0]
                cell_length = current_batch.shape[0]
                current_batch = np.pad(current_batch, ((0, pad_length), (0, 0), (0, 0)), mode='constant')
                current_batch_masks = np.pad(current_batch_masks, ((0, pad_length), (0, 0)), mode='constant')
                current_batch = np.expand_dims(current_batch, axis=0)
                current_batch = np.expand_dims(current_batch, axis=0)
                current_cells = np.squeeze(intermediate_model.predict(current_batch))
                current_cells = current_cells[:cell_length]

                if cells == []:
                    cells = np.array(current_cells)
                    sequences = np.array(current_sequences)
                else:
                    cells = np.vstack((cells, current_cells))
                    sequences = np.vstack((sequences, current_sequences))
            else:
                current_batch = curr_matrix[i * opt.comp_size: i * opt.comp_size + opt.comp_size]
                current_sequences = enc.inverse_transform(np.argmax(current_batch, axis=-1).flatten()). \
                    reshape(current_batch.shape[0], seq_length)

                current_batch_masks = curr_masks[i * opt.comp_size: i * opt.comp_size + opt.comp_size]
                current_batch = np.expand_dims(current_batch, axis=0)
                current_batch = np.expand_dims(current_batch, axis=0)
                current_cells = np.squeeze(intermediate_model.predict(current_batch))

                if cells == []:
                    cells = np.array(current_cells)
                    sequences = np.array(current_sequences)
                else:
                    cells = np.vstack((cells, current_cells))
                    sequences = np.vstack((sequences, current_sequences))

        # For each sequence and filter, record the maximum activation
        for s in range(0, cells.shape[0]):
            curr_seq = sequences[s]
            curr_activation = cells[s]

            stored_seqs = []
            stored_activations = []

            for a in range(0, cells.shape[2]):
                max_activation = np.max(curr_activation[:, a])
                values[a].append(max_activation)
                if max_activation > maximums[a]:
                    maximums[a] = max_activation

    normalized_values = []      # Store the relative activation to the maximum possible activation
    for a in range(0, n_filters):
        normalized_values.append(np.array(values[a]) / maximums[a])

    # Second pass: Iterate through all sequences and store sequences that meet threshold
    all_sequence_names = [[] for x in range(0, n_filters)]
    scer_sequence_names = [[] for x in range(0, n_filters)]
    all_sequences = [[] for x in range(0, n_filters)]
    count = 0
    for j in range(0, len(ds.file_info)):
        curr_matrix, curr_masks, name = ds.load_all_sequences(j)
        if scer_only:
            curr_matrix = np.expand_dims(curr_matrix[0], axis=0)
            curr_masks = np.expand_dims(curr_masks[0], axis=0)
        count += len(curr_matrix)

        print("Storing sequences with ", name)
        n_chunks = np.int(np.floor(curr_matrix.shape[0] // opt.comp_size) + 1)

        # Iterate through all sequence in fasta file
        cells = []
        sequences = []
        for i in range(0, n_chunks):
            if i == n_chunks - 1:
                current_batch = curr_matrix[i * opt.comp_size:]
                current_sequences = enc.inverse_transform(np.argmax(current_batch, axis=-1).flatten()). \
                    reshape(current_batch.shape[0], seq_length)

                current_batch_masks = curr_masks[i * opt.comp_size:]
                pad_length = opt.comp_size - current_batch.shape[0]
                cell_length = current_batch.shape[0]
                current_batch = np.pad(current_batch, ((0, pad_length), (0, 0), (0, 0)), mode='constant')
                current_batch_masks = np.pad(current_batch_masks, ((0, pad_length), (0, 0)), mode='constant')
                current_batch_masks = np.expand_dims(current_batch_masks, axis=0)
                current_batch_masks = np.expand_dims(current_batch_masks, axis=-1)
                current_batch = np.expand_dims(current_batch, axis=0)
                current_batch = np.expand_dims(current_batch, axis=0)
                current_cells = np.squeeze(intermediate_model.predict(current_batch))
                current_cells = current_cells[:cell_length]

                if cells == []:
                    cells = np.array(current_cells)
                    sequences = current_sequences
                else:
                    cells = np.vstack((cells, current_cells))
                    sequences = np.vstack((sequences, current_sequences))
            else:
                current_batch = curr_matrix[i * opt.comp_size: i * opt.comp_size + opt.comp_size]
                current_sequences = enc.inverse_transform(np.argmax(current_batch, axis=-1).flatten()). \
                    reshape(current_batch.shape[0], seq_length)

                current_batch_masks = curr_masks[i * opt.comp_size: i * opt.comp_size + opt.comp_size]
                current_batch = np.expand_dims(current_batch, axis=0)
                current_batch = np.expand_dims(current_batch, axis=0)
                current_batch_masks = np.expand_dims(current_batch_masks, axis=0)
                current_batch_masks = np.expand_dims(current_batch_masks, axis=-1)
                current_cells = np.squeeze(intermediate_model.predict(current_batch))

                if cells == []:
                    cells = np.array(current_cells)
                    sequences = current_sequences
                else:
                    cells = np.vstack((cells, current_cells))
                    sequences = np.vstack((sequences, current_sequences))

        # For each sequence and filter, record the maximum activation if it passes the filter
        for s in range(0, cells.shape[0]):
            curr_seq = sequences[s]
            curr_activation = cells[s]

            stored_seqs = []
            stored_activations = []

            for a in range(0, cells.shape[2]):
                max_activation = np.max(curr_activation[:, a])
                max_position = np.argmax(curr_activation[:, a])

                rel_activation = np.max(curr_activation[:, a]) / maximums[a]
                activation_list = np.array(normalized_values[a])
                num_activations = len(np.where(activation_list >= percent_filter)[0])

                if num_activations >= threshold:
                    if max_activation >= maximums[a] * percent_filter:
                        start_position = max_position - (window_size - 1) // 2
                        start_padding = 0
                        if start_position < 0:
                            start_padding = 0 - start_position
                            start_position = 0

                        end_position = max_position + (window_size - 1) // 2
                        end_padding = 0
                        if end_position > seq_length - 1:
                            end_padding = end_position - seq_length + 1
                            end_position = seq_length

                        max_sequence = "-" * start_padding + ''.join(curr_seq[start_position:end_position + 1].tolist()) \
                                       + "-" * end_padding

                        all_sequences[a].append(max_sequence)
                        all_sequence_names[a].append([name.split(".")[0], max_activation, rel_activation, max_sequence, max_position])
                        if s == 0:
                            scer_sequence_names[a].append(
                                [name.split(".")[0], max_activation, rel_activation, max_sequence, max_position])
                else:
                    indices = np.argsort(-activation_list)[:threshold]
                    if count - (cells.shape[0] - s) in indices:
                        start_position = max_position - (window_size - 1) // 2
                        start_padding = 0
                        if start_position < 0:
                            start_padding = 0 - start_position
                            start_position = 0

                        end_position = max_position + (window_size - 1) // 2
                        end_padding = 0
                        if end_position > seq_length - 1:
                            end_padding = end_position - seq_length + 1
                            end_position = seq_length

                        max_sequence = "-" * start_padding + ''.join(curr_seq[start_position:end_position + 1].tolist()) \
                                       + "-" * end_padding

                        all_sequences[a].append(max_sequence)
                        all_sequence_names[a].append([name.split(".")[0], max_activation, rel_activation, max_sequence, max_position])
                        if s == 0:
                            scer_sequence_names[a].append(
                                [name.split(".")[0], max_activation, rel_activation, max_sequence, max_position])

    enc2 = LabelEncoder().fit(
        ['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S', 'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T', '-'])

    # Iterate through filters and take max X% activations to construct visualization
    for a in range(0, n_filters):
        filter_seqs = np.array(all_sequences[a])
        curr_names = all_sequence_names[a]
        scer_names = scer_sequence_names[a]

        # Iterate through filtered sequences and generate PFM
        pfm = np.zeros((window_size, 20))
        for s in filter_seqs:
            transformed = np.eye(21)[enc2.transform(list(s))][:, 1:]
            pfm += transformed

        # If any rows are all 0, pad them
        pfm_sums = np.sum(pfm, axis=1)
        max_sum = np.max(pfm_sums)
        for i in range(0, window_size):
            if pfm_sums[i] == 0:
                pfm[i, :] += 0.01

            if scale_filters:
                pfm[i, :] += (max_sum - pfm_sums[i]) / 20

        # Visualize and save
        if not os.path.exists(outdir + "/"):
            os.makedirs(outdir + "/")

        pfm = pd.DataFrame(pfm)
        cpm = seqlogo.CompletePm(pfm=pfm, alphabet_type='AA')
        image = seqlogo.seqlogo(cpm, ic_scale=True,
                                filename=outdir + "/" + "filter_" + str(a) + ".png",
                                format='png', size='medium')

        output = open(outdir + "/filter_" + str(a) + "_idr_activations.txt", "w")
        for n in range(0, len(curr_names)):
            output.write(curr_names[n][0].split("_")[0] + "\t" + str(curr_names[n][0]) + "\t" + str(
                curr_names[n][1]) + "\t" + str(curr_names[n][2]) + "\t" + str(curr_names[n][3]) + "\t"
                         + str(curr_names[n][4]) + "\n")




