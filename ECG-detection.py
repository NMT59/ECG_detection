from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import os
import glob
import wfdb
import keras
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.simplefilter('ignore')
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, BatchNormalization, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers

model = Sequential()

model.add(TimeDistributed(Conv1D(32, 5, activation='elu'), input_shape=(None, 100, 1)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

model.add(TimeDistributed(Conv1D(32, 5, activation='elu'), input_shape=(None, 100, 1)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling1D(8, strides=1)))


model.add(TimeDistributed(Conv1D(64, 4, activation='elu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

model.add(TimeDistributed(Conv1D(64, 4, activation='elu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling1D(8, strides=1)))


model.add(TimeDistributed(Conv1D(128, 3, activation='elu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

model.add(TimeDistributed(Conv1D(128, 3, activation='elu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling1D(8, strides=1)))


model.add(TimeDistributed(Conv1D(256, 2, activation='elu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

model.add(TimeDistributed(Conv1D(256, 2, activation='elu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling1D(8, strides=1)))


# extract features and dropout
model.add(TimeDistributed(Flatten()))

model.add(TimeDistributed(Dense(256, activation='elu')))

# input to LSTM
model.add(LSTM(256, return_sequences=False, dropout=0.5))

# classifier with sigmoid activation for multilabel
model.add(Dense(4, activation='softmax'))
optimizer = optimizers.Adam()
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"])
model.summary()

DATASET_PATH = './mit-bih-arrhythmia-database-1.0.0'
header_path = os.path.join(DATASET_PATH, '*hea')
paths = glob.glob(header_path)

# Remove the extension and store the path
paths = [path[:-4] for path in paths]

# Remove paced beat record
remove_paced_beats = ['102', '104', '107', '217']

# Store the data path
data_paths = [path for path in paths if path[-3:] not in remove_paced_beats]

train_data = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
              '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']

train_data_paths = [path for path in data_paths if path[-3:] in train_data]

test_data = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
             '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

random.seed(42)
random.shuffle(test_data)
validation_data_paths = [path for path in data_paths if path[-3:] in test_data[:-5]]
test_data_paths = [path for path in data_paths if path[-3:] in test_data[-5:]]

# Generate 50 sequences from both sides of the sample
def sequences(symbol, signal, sample, signal_length):
    non_beat_annotations = ["[", "!", "]", "x", "(", ")", "p", "t", "u", "`", "'",
                            "^", "|", "~", "+", "s", "T", "*", "D", "=", '"', "@"]

    # Following beats are considered for the analysis
    beat_annotations = ["N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "f"]

    start = sample - 50
    end = sample + 50

    if symbol in beat_annotations and start > 0 and end < signal_length:
        signal_lead_0 = signal[start:end, 0].reshape(1, -1, 1)
        signal_lead_1 = signal[start:end, 1].reshape(1, -1, 1)

        return signal_lead_0, signal_lead_1, symbol

    else:
        return [], [], []

# Scale the data
def preprocess(signal):
    scaler = StandardScaler()
    scaled_signal = scaler.fit_transform(signal)
    return scaled_signal

# Train, Validation and Test data and labels
def generate_data(path):
    signal_channel_0 = []
    signal_channel_1 = []
    labels_channel_0 = []
    labels_channel_1 = []

    for file in path:
        # Load the ECG signal from 2 leads
        record = wfdb.rdrecord(file)

        # Check the frequency is 360
        assert record.fs == 360, 'sample frequency is not 360'
        scaled_signal = preprocess(record.p_signal)
        signal_length = scaled_signal.shape[0]
        annotation = wfdb.rdann(file, 'atr')
        samples = annotation.sample
        symbols = annotation.symbol

        N = ['.', 'N', 'L', 'R', 'A', 'a', 'J', 'S', 'e', 'j']

        for i, sample in enumerate(samples):
            signal_0, signal_1, valid_label = sequences(symbols[i], scaled_signal, sample, signal_length)
            signal_channel_0.extend(signal_0)
            signal_channel_1.extend(signal_1)

            if valid_label != []:

                if valid_label in N:
                    label = 'N'
                else:
                    label = valid_label

                labels_channel_0.append(label)
                labels_channel_1.append(label)

    signals = np.vstack((signal_channel_0, signal_channel_1))
    labels_channel_0_array = np.array([labels_channel_0]).reshape(-1, 1)
    labels_channel_1_array = np.array([labels_channel_1]).reshape(-1, 1)
    labels = np.vstack((labels_channel_0_array, labels_channel_1_array))

    return signals, labels

# Train, Validation and Test data and labels
train_signals, train_labels = generate_data(train_data_paths)
validation_signals, validation_labels = generate_data(validation_data_paths)
test_signals, test_labels = generate_data(test_data_paths)

# One hot encoding of labels
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=False)

train_labels_numerical = label_encoder.fit_transform(train_labels.reshape(-1, 1))
train_labels_encoded = one_hot_encoder.fit_transform(train_labels_numerical.reshape(-1, 1))

validation_labels_numerical = label_encoder.transform(validation_labels.reshape(-1, 1))
validation_labels_encoded = one_hot_encoder.transform(validation_labels_numerical.reshape(-1, 1))

test_labels_numerical = label_encoder.transform(test_labels.reshape(-1, 1))
test_labels_encoded = one_hot_encoder.transform(test_labels_numerical.reshape(-1, 1))

def generator(X, y, batch_size):
    num_batches = len(X) // batch_size
    while True:
        np.random.seed(100)
        shuffle_sequence = np.random.permutation(len(X))
        X = X[shuffle_sequence]
        y = y[shuffle_sequence]

        for batch in range(num_batches):
            batch_data = np.zeros((batch_size, 100, 1))
            batch_labels = np.zeros((batch_size, 4))

            for folder in range(batch_size):
                batch_data[folder, :, :] = X[folder + (batch * batch_size), :, :]
                batch_labels[folder, :] = y[folder + (batch * batch_size), :]
            yield batch_data.reshape(batch_size, 1, 100, 1), batch_labels.reshape(batch_size, 4)

        if (len(X) % batch_size != 0):
            batch_size = len(X) % batch_size

            batch_data = np.zeros((batch_size, 100, 1))
            batch_labels = np.zeros((batch_size, 4))

            for folder in range(batch_size):
                batch_data[folder, :, :] = X[folder + (num_batches * batch_size), :, :]
                batch_labels[folder, :] = y[folder + (num_batches * batch_size), :]
            yield batch_data.reshape(batch_size, 1, 100, 1), batch_labels.reshape(batch_size, 4)

#print('Shape of train data:', train_signals.shape)
#print('Shape of train labels:', train_labels_encoded.shape)
#print('Shape of validation data:', validation_signals.shape)
#print('Shape of validation labels:', validation_labels_encoded.shape)
#print('Shape of test data:', test_signals.shape)
#print('Shape of test labels:', test_labels_encoded.shape)

batch_size = 2048
train_generator = generator(train_signals, train_labels_encoded, batch_size)
val_generator = generator(validation_signals, validation_labels_encoded, batch_size)

#Training time
curr_dt_time = datetime.datetime.now()
num_train_sequences = len(train_signals)
print('# training sequences =', num_train_sequences)
num_val_sequences = len(validation_signals)
print('# validation sequences =', num_val_sequences)
num_epochs = 20 # choose the number of epochs
print ('# epochs =', num_epochs)

if (num_train_sequences % batch_size) == 0:
    steps_per_epoch = int(num_train_sequences / batch_size)
else:
    steps_per_epoch = (num_train_sequences // batch_size) + 1

if (num_val_sequences % batch_size) == 0:
    validation_steps = int(num_val_sequences / batch_size)
else:
    validation_steps = (num_val_sequences // batch_size) + 1
model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ', '').replace(':', '_') + '/'

if not os.path.exists(model_name):
    os.mkdir(model_name)

filepath = model_name + 'model-{epoch:05d}-{loss:.3f}-{categorical_accuracy:.3f}-{val_loss:.3f}-{val_categorical_accuracy:.3f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                             mode='auto', period=1)

LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, cooldown=1,
                       verbose=1)  # write the REducelronplateau code here
callbacks_list = [checkpoint, LR]
history = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1,
                              callbacks=callbacks_list, validation_data=val_generator,
                              validation_steps=validation_steps, class_weight=None, shuffle=True, workers=1,
                              initial_epoch=0, use_multiprocessing=False)

# Save the model
model.save('LSTM-ECG.h5')

# Load the model for evaluation
test_model = load_model('LSTM-ECG.h5')
# Reshape the test data for prediction
test_signals_reshaped = test_signals.reshape(test_signals.shape[0], -1, test_signals.shape[1], test_signals.shape[2])
# Predict the test data
test_results = test_model.predict(test_signals_reshaped[:,:,:,:])
test_predicted = np.argmax(test_results, axis=1)
# Confusion Matrix
confusion_matrix = metrics.confusion_matrix(test_labels_numerical, test_predicted)
print(confusion_matrix)

# Classification_report
print(metrics.classification_report(test_labels_numerical, test_predicted))


