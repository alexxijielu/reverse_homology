import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.losses import categorical_crossentropy as cce

import opts as opt
from dataset_repeat_pad import Dataset
from model_scaled_final import RHModel


'''Loads a batch from the Dataset object and formats it for the neural network'''
def load_seqs_gt(ds, file_ids, ref_size=8, comp_size=400):
    ref_in, comp_in, ref_mask, comp_mask, labels = ds.load_list(file_ids, ref_size=ref_size, comp_size=comp_size)

    # Pad everything to a batch size of 1 and deal with within the Keras model
    ref_in = np.expand_dims(ref_in, axis=0)
    comp_in = np.expand_dims(comp_in, axis=0)
    ref_mask = np.expand_dims(ref_mask, axis=0)
    comp_mask = np.expand_dims(comp_mask, axis=0)
    labels = np.expand_dims(labels, axis=0)

    return ref_in, comp_in, ref_mask, comp_mask, labels

'''Data generator during training'''
def data_generator(dataset, shuffle=True, batch_size=1, ref_size=8, comp_size=400):
    b = 0  # batch item index
    file_index = 0
    file_ids = np.copy(dataset.file_ids)

    # Runs indefinitely for Keras
    while True:
        # If we've exhausted the image dataset, reshuffle the indices
        file_index = (file_index + batch_size) % len(file_ids)
        if shuffle and file_index < batch_size:
            np.random.shuffle(file_ids)

        # Get batch from data loader
        curr_ids = file_ids[file_index:file_index + batch_size]
        if len(curr_ids) != batch_size:
            pass
        else:
            ref_in, comp_in, ref_mask, comp_mask, labels \
                = load_seqs_gt(dataset, curr_ids, comp_size=comp_size, ref_size=ref_size)

            # Return batch
            inputs = [ref_in, comp_in, ref_mask, comp_mask]
            outputs = labels

            yield inputs, outputs

'''Custom metric function to work with batch size trick: same as native but handles first dimension being 1'''
def recall_m(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=0)
    y_pred = K.squeeze(y_pred, axis=0)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

'''Custom metric function to work with batch size trick: same as native but handles first dimension being 1'''
def precision_m(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=0)
    y_pred = K.squeeze(y_pred, axis=0)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

'''Custom loss function to work with batch size trick: same as native but handles first dimension being 1'''
def custom_categorical_crossentropy(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=0)
    y_pred = K.squeeze(y_pred, axis=0)
    return cce(y_true, y_pred)

if __name__ == "__main__":
    print("Preparing the dataset...")
    ds = Dataset(min_count=opt.min_count, map_path=opt.map_path)
    ds.add_dataset(opt.data_path, clip_M=True)
    ds.prepare()
    train_generator = data_generator(ds, batch_size=opt.batch_size, comp_size=opt.comp_size, ref_size=opt.ref_size)
    steps = len(ds.file_info) // opt.batch_size

    print("Training the model...")
    model = RHModel().create_model(comp_size=opt.comp_size, ref_size=opt.ref_size,
                                   seq_length=ds.seq_length, bsize=opt.batch_size)
    optimizer = tf.train.AdamOptimizer(learning_rate=opt.learning_rate, beta1=0.5)
    model.compile(optimizer=optimizer, loss=custom_categorical_crossentropy, metrics=['accuracy', precision_m, recall_m])
 
    csv_logger = CSVLogger(opt.checkpoint_path + "model_history_log.csv", append=True)
    saver = ModelCheckpoint(filepath=opt.checkpoint_path + "{epoch:02d}_weights.h5", verbose=False, period=200)
    model.fit_generator(train_generator, steps_per_epoch=steps, epochs=opt.epochs, use_multiprocessing=False, callbacks=[saver, csv_logger])

    print("Saving model weights in " + opt.checkpoint_path)
    model.save(opt.checkpoint_path + "final_weights.h5")