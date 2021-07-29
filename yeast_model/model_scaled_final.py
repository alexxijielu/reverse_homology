import tensorflow as tf
import keras as K
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import regularizers

class RHModel():
    def __init__(self):
        print ('Building model...')

    def create_model(self, comp_size, ref_size, seq_length, bsize, training=True, print_summary=True):
        # Inputs: "batch_size" will always be 1 to compensate for differing batch sizes
        comp_in = layers.Input(shape=(1, comp_size, seq_length, 20), name='comp_in', batch_size=1)
        ref_in = layers.Input(shape=(bsize, ref_size, seq_length, 20), name='ref_in', batch_size=1)
        comp_mask = layers.Input(shape=(1, comp_size, seq_length, 1), name='comp_mask', batch_size=1)
        ref_mask = layers.Input(shape=(bsize, ref_size, seq_length, 1), name='ref_mask', batch_size=1)

        # Remove the first dimension to enable differing batch sizes for target vs reference encoders
        c_in = layers.Lambda(lambda t: K.backend.reshape(t, (1, comp_size, seq_length, 20)))(comp_in)
        c_mask = layers.Lambda(lambda t: K.backend.reshape(t, (1, comp_size, seq_length, 1)))(comp_mask)
        x_in = layers.Lambda(lambda t: K.backend.reshape(t, (bsize, ref_size, seq_length, 20)))(ref_in)
        x_mask = layers.Lambda(lambda t: K.backend.reshape(t, (bsize, ref_size, seq_length, 1)))(ref_mask)

        n_conv_features = 10
        n_features = 256

        # Convolutional encoders for reference
        x_conv1_1 = layers.Conv2D(n_conv_features, (1, 1), activation='relu', padding='same', name='conv1_scale1')(x_in)
        x_conv1_3 = layers.Conv2D(n_conv_features * 2, (1, 3), activation='relu', padding='same', name='conv1_scale2')(x_in)
        x_conv1_5 = layers.Conv2D(n_conv_features * 3, (1, 5), activation='relu', padding='same', name='conv1_scale3')(x_in)
        x_conv1 = layers.Concatenate(axis=-1)([x_conv1_1, x_conv1_3, x_conv1_5])
        x_conv2 = layers.Conv2D(n_features, (1, 5), activation='relu', padding='same')(x_conv1)
        x_conv2 = layers.BatchNormalization()(x_conv2, training=training)
        x_conv3 = layers.Conv2D(n_features, (1, 7), activation='relu', padding='same', name='conv3')(x_conv2)
        x_conv3 = layers.BatchNormalization()(x_conv3, training=training)

        # Convolutional encoders for targets
        c_conv1_1 = layers.Conv2D(n_conv_features, (1, 1), activation='relu', padding='same', name='c_conv1_scale1')(c_in)
        c_conv1_3 = layers.Conv2D(n_conv_features * 2, (1, 3), activation='relu', padding='same', name='c_conv1_scale2')(c_in)
        c_conv1_5 = layers.Conv2D(n_conv_features * 3, (1, 5), activation='relu', padding='same', name='c_conv1_scale3')(c_in)
        c_conv1 = layers.Concatenate(axis=-1)([c_conv1_1, c_conv1_3, c_conv1_5])
        c_conv2 = layers.Conv2D(n_features, (1, 5), activation='relu', padding='same')(c_conv1)
        c_conv2 = layers.BatchNormalization()(c_conv2, training=training)
        c_conv3 = layers.Conv2D(n_features, (1, 7), activation='relu', padding='same', name='c_conv3')(c_conv2)
        c_conv3 = layers.BatchNormalization()(c_conv3, training=training)

        # Max pool targets
        c_max = layers.Multiply()([c_conv3, c_mask])
        c_max = layers.MaxPooling2D(data_format='channels_first', pool_size=(seq_length, 1))(c_max)
        c_max = layers.Reshape((comp_size, n_features))(c_max)

        # Average pool targets (account for padding)
        c_avg = layers.Multiply()([c_conv3, c_mask])
        c_avg = layers.AvgPool2D(data_format='channels_first', pool_size=(seq_length, 1))(c_avg)
        c_avg = layers.Reshape((comp_size, n_features))(c_avg)
        comp_mask_sum = layers.Lambda(lambda t: K.backend.sum(t, axis=2))(c_mask)
        c_avg = layers.Lambda(lambda t: t / seq_length)(c_avg)
        c_avg = layers.Multiply()([c_avg, comp_mask_sum])
        c_avg = layers.Lambda(lambda t: t * 17.06)(c_avg)

        # Concatenate average and max pool for targets
        c = layers.Concatenate(axis=-1, name='conv3_concat')([c_max, c_avg])

        # Fully connected encoder for target feature vectors
        c_fc1 = layers.Dense(n_features, activation='relu', name="c_fc_1")(c)
        c_fc1 = layers.BatchNormalization()(c_fc1, training=training)
        c_fc2 = layers.Dense(n_features, activation='relu', name="c_fc_2")(c_fc1)
        c = layers.BatchNormalization()(c_fc2, training=training)

        # Max pool references
        x_max = layers.Multiply()([x_conv3, x_mask])
        x_max = layers.MaxPooling2D(data_format='channels_first', pool_size=(seq_length, 1))(x_max)
        x_max = layers.Reshape((ref_size, n_features))(x_max)

        # Average pool references (account for padding)
        x_avg = layers.Multiply()([x_conv3, x_mask])
        x_avg = layers.AvgPool2D(data_format='channels_first', pool_size=(seq_length, 1))(x_avg)
        x_avg = layers.Reshape((ref_size, n_features))(x_avg)
        x_mask_sum = layers.Lambda(lambda t: K.backend.sum(t, axis=2))(x_mask)
        x_avg = layers.Lambda(lambda t: t / seq_length)(x_avg)
        x_avg = layers.Multiply()([x_avg, x_mask_sum])
        x_avg = layers.Lambda(lambda t: t * 17.06)(x_avg)

        # Concatenate average and max pool for reference
        x = layers.Concatenate(axis=-1)([x_max, x_avg])

        # Fully connected encoders for reference
        x_fc1 = layers.Dense(n_features, activation='relu', name="x_fc_1")(x)
        x_fc1 = layers.BatchNormalization()(x_fc1, training=training)
        x_fc2 = layers.Dense(n_features, activation='relu', name="x_fc_2")(x_fc1)
        x = layers.BatchNormalization()(x_fc2, training=training)

        # Max pool over species for reference
        x = layers.AveragePooling1D(data_format='channels_last', pool_size=ref_size)(x)

        # Concatenate and classify
        x = layers.Lambda(K.backend.tile, arguments={'n': (1, comp_size, 1)})(x)
        c = layers.Lambda(K.backend.tile, arguments={'n': (bsize, 1, 1)})(c)
        out = layers.Lambda(lambda t: K.backend.sum(t[1] * t[0], axis=-1, keepdims=True))([x, c])
        out = layers.Flatten(name="presoftmax")(out)
        out = layers.Activation('softmax', name='output')(out)
        out = layers.Lambda(lambda t: K.backend.reshape(t, (1, bsize, comp_size)))(out)

        model = models.Model(inputs=[ref_in, comp_in, ref_mask, comp_mask], outputs=out)
        if print_summary:
            print (model.summary())
        return model

if __name__ == "__main__":
    print("Training the model...")
    model = RHModel().create_model(comp_size=50, ref_size=8, seq_length=128, bsize=64)

