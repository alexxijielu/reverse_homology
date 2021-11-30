import sys
sys.path.append("..")
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

import tensorflow as tf
from model_scaled_final import RHModel
from tensorflow.contrib.keras import layers
import keras as K

if __name__ == "__main__":
    datapath = '../hs_idr_alignments_filtered/*.fasta'      # Folder of fasta files to extract features from
    outfile = "./rh_human_features_1000.tsv"                         # Where to store features
    weights_file = "../human_idr_model/1000_weights.h5"
    speciesname = "HUMAN"

    seq_length = 256
    n_features = 256
    enc = LabelEncoder().fit(['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S',
                              'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T'])

    map = open(opt.map_path)
    assignments = csv.reader(map, delimiter=',')
    assignments = np.array([row for row in assignments])
    orfs = assignments[1:, 1]
    starts = assignments[1:, 2]

    print("Loading the model...")
    model = RHModel().create_model(comp_size=opt.comp_size, ref_size=opt.ref_size, seq_length=seq_length,
                                   bsize=opt.batch_size, training=False)
    model.load_weights(weights_file)   # Where to load model from
    trimmed_model = tf.keras.Model(inputs=model.get_layer("comp_in").input, outputs=model.get_layer("c_conv3").output)
    reshape_layer = layers.Lambda(lambda t: K.backend.reshape(t, (1, opt.comp_size, seq_length, n_features)))(trimmed_model.output)
    intermediate_model = tf.keras.Model(inputs=trimmed_model.input, outputs=[reshape_layer])

    print("Evaluating fasta files...")
    buffer = 0
    buffer_sequences = []
    buffer_names = []
    for file_name in glob.glob(datapath):
        count = 0
        for record in SeqIO.parse(file_name, "fasta"):
            if speciesname in record.id:
                curr_name = file_name.split("/")[-1].split(".")[0]
                curr_seq = str(record.seq)
                curr_seq = curr_seq.replace("-", "")
                print ("Working on ", curr_name)

                remove_M = False
                if "HUMAN" in curr_name:
                    curr_start = curr_name.split("_")[1].split("to")[0]
                    if curr_start == '1':
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

                    input_sequence = np.expand_dims(curr_seq_onehot, axis=0)
                    input_sequence = np.expand_dims(input_sequence, axis=0)
                    input_sequence = np.expand_dims(input_sequence, axis=0)
                    input_sequence = np.pad(input_sequence, ((0, 0), (0, 0), (0, opt.comp_size - 1),
                                                             (0, 0), (0, 0)), mode='constant')

                    representation = np.squeeze(intermediate_model.predict(input_sequence))[0, :, :]
                    max_representation = np.max(representation, axis=0)
                    avg_representation = np.average(representation, axis=0)
                    representation = np.concatenate([max_representation, avg_representation])
                    output = open(outfile, "a")
                    output.write(curr_name)
                    for f in representation:
                        output.write("\t" + str(f))
                    output.write("\n")
