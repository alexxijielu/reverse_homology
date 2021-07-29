import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import opts as opt
import glob
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv

import sys
sys.path.append("..")
import tensorflow as tf
from model_scaled_final import RHModel
from tensorflow.contrib.keras import layers
import keras as K

if __name__ == "__main__":
    datapath = './sc_idr_alignments/*.fasta'    # Folder of fasta files to extract features from
    outfile = "./rh_scaled_final_features/"     # Output directory to store features
    seq_length = 256
    n_features = 256
    enc = LabelEncoder().fit(['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S',
                              'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T'])

    map = open(opt.map_path)
    assignments = csv.reader(map, delimiter=',')
    assignments = np.array([row for row in assignments])
    orfs = assignments[1:, 1]
    starts = assignments[1:, 2]

    epochs = ['1000']       # Which epochs of pretrained model to extract features from
    for epoch in epochs:
        print("Loading the model...")
        model = RHModel().create_model(comp_size=opt.comp_size, ref_size=opt.ref_size, seq_length=seq_length,
                                       bsize=opt.batch_size, training=False)
        model.load_weights("../scer_idr_model/" + epoch + "_weights.h5")
        trimmed_model = tf.keras.Model(inputs=[model.get_layer("comp_in").input, model.get_layer("comp_mask").input], outputs=model.get_layer("c_fc_2").output)
        reshape_layer = layers.Lambda(lambda t: K.backend.reshape(t, (1, opt.comp_size, n_features)))(trimmed_model.output)
        intermediate_model = tf.keras.Model(inputs=trimmed_model.input, outputs=[reshape_layer])

        print("Evaluating fasta files...")
        buffer = 0
        buffer_sequences = []
        buffer_names = []
        for file_name in glob.glob(datapath):
            count = 0
            for record in SeqIO.parse(file_name, "fasta"):
                if "Scer" in record.id:
                    curr_name = file_name.split("/")[-1].split(".")[0]
                    curr_seq = str(record.seq)
                    curr_seq = curr_seq.replace("-", "")
                    print ("Working on ", curr_name)

                    remove_M = False
                    try:
                        curr_start = starts[np.where(orfs == curr_name)[0][0]]
                        if curr_start == '1':
                            remove_M = True
                    except IndexError:
                        print("Couldn't find ", curr_name, " in the IDR mapping")
                        remove_M = True

                    if "X" in curr_seq or len(curr_seq) < 5:
                        pass
                    else:
                        # preprocess sequence
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
                        buffer_sequences.append(curr_seq_onehot)
                        buffer_names.append(curr_name)
                        buffer += 1

                        if buffer == opt.comp_size:
                            input_sequence = np.expand_dims(np.array(buffer_sequences), axis=0)
                            input_sequence = np.expand_dims(input_sequence, axis=0)
                            pad_length = opt.comp_size - input_sequence.shape[2]
                            if pad_length > 0:
                                input_sequence = np.pad(input_sequence, ((0, 0), (0, 0), (0, pad_length), (0, 0), (0, 0)))
                            mask = np.ones((input_sequence.shape[0], input_sequence.shape[1], input_sequence.shape[2],
                                            input_sequence.shape[3], 1))

                            representation = np.squeeze(intermediate_model.predict([input_sequence, mask]))
                            output = open(outfile + epoch + "_features.txt", "a")
                            for r in range(0, len(representation)):
                                output.write(buffer_names[r])
                                for f in representation[r]:
                                    output.write("\t" + str(f))
                                output.write("\n")

                            buffer = 0
                            buffer_sequences = []
                            buffer_names = []

        # Flush out the remainder of the buffer
        input_sequence = np.expand_dims(np.array(buffer_sequences), axis=0)
        input_sequence = np.expand_dims(input_sequence, axis=0)
        pad_length = opt.comp_size - input_sequence.shape[2]
        if pad_length > 0:
            input_sequence = np.pad(input_sequence, ((0, 0), (0, 0), (0, pad_length), (0, 0), (0, 0)), mode='constant')
        mask = np.ones((input_sequence.shape[0], input_sequence.shape[1], input_sequence.shape[2],
                        input_sequence.shape[3], 1))

        representation = np.squeeze(intermediate_model.predict([input_sequence, mask]))[:opt.comp_size - pad_length]
        output = open(outfile + epoch + "_features.txt", "a")
        for r in range(0, len(representation)):
            output.write(buffer_names[r])
            for f in representation[r]:
                output.write("\t" + str(f))
            output.write("\n")
