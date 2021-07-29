from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

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
import re

import tensorflow as tf
from model_scaled_final import RHModel
from tensorflow.contrib.keras import layers
import keras as K

# Set dictionary of regular expressions for motifs
regex_dict = {"CLV_Separin_Fungi": "S[IVLMH]E[IVPFMLYAQR]GR.",
              "DEG_APCC_KENBOX_2": ".KEN.",
              "DEG_APCC_TPR_1": ".[ILM]R",
              "DOC_CKS1_1": "[MPVLIFWYQ].(T)P..",
              "DOC_MAPK_DCC_7": "[RK].{2,4}[LIVP]P.[LIV].[LIVMF]|[RK].{2,4}[LIVP].P[LIV].[LIVMF]",
              "DOC_MAPK_gen_1": "[KR]{0,2}[KR].{0,2}[KR].{2,4}[ILVM].[ILVF]",
              "DOC_MAPK_HePTP_8": "([LIV][^P][^P][RK]....[LIVMP].[LIV].[LIVMF])|([LIV][^P][^P][RK][RK]G.{4,7}[LIVMP].[LIV].[LIVMF])",
              "DOC_PP1_RVXF_1": "..[RK].{0,1}[VIL][^P][FW].",
              "DOC_PP2B_PxIxI_1": ".P[^P]I[^P][IV][^P]",
              "LIG_APCC_Cbox_2": "DR[YFH][ILFVM][PA]..",
              "LIG_AP_GAE_1": "[DE][DES][DEGAS]F[SGAD][DEAP][LVIMFD]",
              "LIG_CaM_IQ_9": "[ACLIVTM][^P][^P][ILVMFCT]Q[^P][^P][^P][RK][^P]{4,5}[RKQ][^P][^P]",
              "LIG_EH_1": ".NPF.",
              "LIG_eIF4E_1": "Y....L[VILMF]",
              "LIG_GLEBS_BUB3_1": "[EN][FYLW][NSQ].EE[ILMVF][^P][LIVMFA]",
              "LIG_LIR_Gen_1": "[EDST].{0,2}[WFY]..[ILV]",
              "LIG_PCNA_PIPBox_1": "((^.{0,3})|(Q)).[^FHWY][ILM][^P][^FHILVWYP][HFM][FMY]..",
              "LIG_SUMO_SIM_par_1": "[DEST]{0,5}.[VILPTM][VIL][DESTVILMA][VIL].{0,1}[DEST]{1,10}",
              "MOD_CDK_SPxK_1": "...([ST])P.[KR]",
              "MOD_LATS_1": "H.[KR]..([ST])[^P]",
              "MOD_SUMO_for_1": "[VILMAFP](K).E",
              "TRG_ER_FFAT_1": "[DE].{0,4}E[FY][FYK]D[AC].[ESTD]",
              "TRG_Golgi_diPhe_1": "Q.{6,6}FF.{6,7}",
              "TRG_NLS_MonoExtN_4": "(([PKR].{0,1}[^DE])|([PKR]))((K[RK])|(RK))(([^DE][KR])|([KR][^DE]))[^DE]",
              "MOD_CDK_STP": "[ST]P",
              "MOD_MEC1": "[ST]Q",
              "MOD_PRK1": "[LIVM]....TG",
              "MOD_IPL1": "[RK].[ST][LIV]",
              "MOD_PKA": "R[RK].S",
              "MOD_CKII": "[ST][DE].[DE]",
              "MOD_IME2": "RP.[ST]",
              "DOC_PRO": "P..P",
              "TRG_ER_HDEL": "HDEL",
              "TRG_MITOCHONDRIA": "[MR]L[RK]",
              "MOD_ISOMERASE": "C..C",
              "TRG_FG": "(F.FG)|(GLFG)",
              "INT_RGG": "(RGG)|(RG)",
              "AA_S": "S[^S]",
              "AA_P": "P[^P]",
              "AA_T": "T[^T]",
              "AA_A": "A[^A]",
              "AA_H": "H[^H]",
              "AA_Q": "Q[^Q]",
              "AA_N": "N[^N]",
              "AA_G": "G[^G]",
              "acidic": "[DE]",
              "basic": "[RK]",
              "aliphatic": "[ALMIV]",
              "polar_fraction": "[QNSTGCH]",
              "chain_expanding": "[EDRKP]",
              "aromatic": "[FYW]",
              "disorder_promoting": "[TAGRDHQKSEP]",
              "REP_Q2" : "Q{2,}",
              "REP_N2" : "N{2,}",
              "REP_S2": "S{2,}",
              "REP_G2": "G{2,}",
              "REP_K2": "K{2,}",
              "REP_R2": "R{2,}",
              "REP_P2": "P{2,}",
              "REP_D2": "D{2,}",
              "REP_E2": "E{2,}",
              "REP_QN2": "[QN]{2,}",
              "REP_RG2": "[RG]{2,}",
              "REP_FG2": "[FG]{2,}",
              "REP_SG2": "[SG]{2,}",
              "REP_SR2": "[SR]{2,}",
              "REP_KAP2": "[KAP]{2,}",
              "REP_PTS2": "[PTS]{2,}",
              "R_plus_Y": "[RY]",
              "DEG_APCC_DBOX_1": "R..L..[LIVM].",
              "DOC_MAPK_MEF2A_6": "[LIVMP].[LIV].[LIVMF]",
              "DOC_CYCLIN_RxL_1": "[^EDWNSG][^D][RK][^D]L",
              "DOC_MAPK_FxFP_2": "F.[FY]P",
              "DOC_PP2A_B56_1": "[LM]..[IL].E",
              "LIG_14-3-3_CanoR_1": "R..[ST].P",
              "LIG_NRBOX": "[^P]L[^P][^P]LL[^P]",
              "LIG_PDZ_Class_1": "...[ST].[ACVILF]$",
              "LIG_PROFILIN_1": "PPP[PA]P[LGP][LGP]",
              "LIG_SH2_SRC": "(Y)[QDEVAIL][DENPYHI][IPVGAHS]",
              "LIG_SH2_GRB2": "(Y).N.",
              "LIG_SH3_1": "[RKY]..P..P",
              "LIG_SH3_2": "P..P.[KR]",
              "LIG_WW_1": "PP.Y",
              "MOD_CAAXbox": "(C)[^DENQ][LIVMF].$",
              "MOD_CDK_SPxxK_3": "...([ST])P..[RK]",
              "MOD_CDC7_priming": "S[ST]P",
              "MOD_GSK3_1": "...([ST])...[ST]",
              "MOD_PIKK_1": "[ST]Q",
              "MOD_N-GLC_1": ".(N)[^P][ST]..",
              "MOD_PKA_1": "[RK][RK].([ST])[^P]..",
              "MOD_PKB_1": "R.R..([ST])[^P]..",
              "MOD_Plk_1": "[DNE][^PG][ST]",
              "MOD_ERK1": "P.[ST]P",
              "TRG_NLS_MonoCore_2": "[^DE]K[RK][KRP][KR][^DE]",
              "TRG_ER_KDEL_1": "[KRHQSAP][DENQT]EL$",
              "LIG_EVH1_1": "[FYWL]P.PP",
              }

if __name__ == '__main__':
    datapath = '../hs_idr_alignments/*.fasta'        # Directory of fasta files
    outfile = "./motif_correlations/"                # Where to save results
    seq_length = 256
    n_features = 256
    species = "HUMAN"
    enc = LabelEncoder().fit(['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S',
                              'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T'])

    print("Loading the models...")
    random_model = RHModel().create_model(comp_size=opt.comp_size, ref_size=opt.ref_size, seq_length=seq_length,
                                          bsize=opt.batch_size, training=False)
    trimmed_random_model = tf.keras.Model(inputs=random_model.get_layer("comp_in").input,
                                          outputs=random_model.get_layer("c_conv3").output)
    random_reshape_layer = layers.Lambda(lambda t: K.backend.reshape(t, (1, opt.comp_size, seq_length, n_features)))(
        trimmed_random_model.output)
    random_model = tf.keras.Model(inputs=trimmed_random_model.input, outputs=[random_reshape_layer])

    model = RHModel().create_model(comp_size=opt.comp_size, ref_size=opt.ref_size, seq_length=seq_length,
                                   bsize=opt.batch_size, training=False)
    model.load_weights("../human_idr_model/1000_weights.h5")    # Where to load model from
    trimmed_model = tf.keras.Model(inputs=model.get_layer("comp_in").input, outputs=model.get_layer("c_conv3").output)
    reshape_layer = layers.Lambda(lambda t: K.backend.reshape(t, (1, opt.comp_size, seq_length, n_features)))(
        trimmed_model.output)
    intermediate_model = tf.keras.Model(inputs=trimmed_model.input, outputs=[reshape_layer])

    # Iterate through fasta files
    regex_values = []
    activations = []
    rand_activations = []
    for file_name in glob.glob(datapath):
        count = 0
        for record in SeqIO.parse(file_name, "fasta"):
            if species in record.id:
                curr_name = file_name.split("/")[-1].split(".")[0]
                curr_seq = str(record.seq)
                curr_seq = curr_seq.replace("-", "")
                print("Working on ", curr_name)

                remove_M = False
                curr_start = curr_name.split("_")[1].split("to")[0]
                if str(curr_start) == '1':
                    remove_M = True

                if "X" in curr_seq or len(curr_seq) < 5:
                    pass
                else:
                    # preprocess sequence
                    if remove_M:
                        if len(curr_seq) > 0:
                            if curr_seq[0] == 'M':
                                curr_seq = curr_seq[1:]

                    input_length = len(curr_seq)
                    if len(curr_seq) > seq_length:
                        input_length = seq_length

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

                    representation = np.squeeze(intermediate_model.predict(input_sequence))[0, :input_length, :]
                    rand_representation = np.squeeze(random_model.predict(input_sequence))[0, :input_length, :]

                    if activations == []:
                        activations = representation
                    else:
                        activations = np.concatenate((activations, representation), axis=0)

                    if rand_activations == []:
                        rand_activations = rand_representation
                    else:
                        rand_activations = np.concatenate((rand_activations, rand_representation), axis=0)

                    tmp = []
                    for e, key in enumerate(regex_dict):
                        match_vector = np.zeros(len(curr_seq[:input_length]))
                        matches = re.finditer(regex_dict[key], curr_seq[:input_length])
                        for m in matches:
                            for i in range(m.start(), m.end()):
                                match_vector[i] = 1
                        tmp.append(match_vector)
                    tmp = np.array(tmp).T

                    if regex_values == []:
                        regex_values = tmp
                    else:
                        regex_values = np.concatenate((regex_values, tmp), axis=0)

    # Calculate and record correlations
    corrmatrix = np.zeros((activations.shape[1], regex_values.shape[1]))
    rand_corrmatrix = np.zeros((rand_activations.shape[1], regex_values.shape[1]))
    for i in range (0, activations.shape[1]):
        print ("Calculating correlations with feature", i)
        for j in range (0, regex_values.shape[1]):
            corrmatrix[i, j] = np.corrcoef(activations[:, i], regex_values[:, j])[0][1]
            rand_corrmatrix[i, j] = np.corrcoef(rand_activations[:, i], regex_values[:, j])[0][1]

    if not os.path.exists(outfile):
        os.makedirs(outfile)

    output = open(outfile + "correlations.txt", "w")
    for key in regex_dict:
        output.write(key)
        output.write("\t")
    output.write("\n")
    for i in range(0, corrmatrix.shape[0]):
        for j in range(0, corrmatrix.shape[1]):
            output.write(str(corrmatrix[i, j]))
            output.write("\t")
        output.write("\n")

    randoutput = open(outfile + "random_correlations.txt", "w")
    for key in regex_dict:
        randoutput.write(key)
        randoutput.write("\t")
    randoutput.write("\n")
    for i in range(0, rand_corrmatrix.shape[0]):
        for j in range(0, rand_corrmatrix.shape[1]):
            randoutput.write(str(rand_corrmatrix[i, j]))
            randoutput.write("\t")
        randoutput.write("\n")
