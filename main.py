import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wfdb
import numpy as np
from torchsummary import summary
import scipy.signal as spysig
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
import torch.optim as optim
import random
# Define constants
DATA_TYPE = np.float32
FREQUENCY_SAMPLING = 360
NEIGHBOUR_POINT = 145
fs=360

#define model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            # Layer 1: Convolutional + ReLU + Dropout
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 2: MaxPooling
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Layer 3: Convolutional + ReLU + Dropout
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 2: MaxPooling
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Layer 3: Convolutional + ReLU + Dropout
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 2: MaxPooling
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Layer 3: Convolutional + ReLU + Dropout
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Flatten
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            # Layer 4: Fully-connected + ReLU + Dropout
            nn.Linear(256, 256),  # Auto-adjust dimensions
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 5: Fully-connected + ReLU + Dropout
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 6: Output
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

summary(CNN(), input_size=(3, 224, 224))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def smooth(x, window_len=300, window='hanning'):
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window_len % 2 == 0:
        window_len += 1
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len / 2):-int(window_len / 2)]

def baseline_wander_remove(signal, fs=360, f1=0.2, f2=0.6):
    window1 = int(f1 * fs / 2) + 1 if int(f1 * fs / 2) % 2 == 0 else int(f1 * fs / 2)
    window2 = int(f2 * fs / 2) + 1 if int(f2 * fs / 2) % 2 == 0 else int(f2 * fs / 2)
    out1 = smooth(signal, window1)
    out2 = smooth(out1, window2)
    bwr_signal = signal - out2
    return bwr_signal

def normalize(raw, window_len, samp_from=-1, samp_to=-1):
    # The window size is the number of samples that corresponds to the time analogue of 2e = 0.5s
    if window_len % 2 == 0:
        window_len += 1
    abs_raw = abs(raw)
    # Remove outlier
    while True:
        g = maximum_filter1d(abs_raw, size=window_len)
        if np.max(abs_raw) < 5.0:
            break
        abs_raw[g > 5.0] = 0
    g_smooth = smooth(g, window_len, window='hamming')
    g_mean = max(np.mean(g_smooth) / 3.0, 0.1)
    g_smooth = np.clip(g_smooth, g_mean, None)
    # Avoid cases where the value is 0
    g_smooth[g_smooth < 0.01] = 1
    nor_signal = np.divide(raw, g_smooth)
    return nor_signal

# Define the function to preprocess the data
def preprocess_data(file_paths, separate=None,sample_size=109446 ):
    data = []
    labels = []
    positive_range = int(0.04 * FREQUENCY_SAMPLING)+1  # ±40ms distance

    for file_path in file_paths:
        file_path = file_path[:-4]
        info = wfdb.rdheader(file_path)
        signal_length = info.sig_len
        if separate == 1:
            signal, _ = wfdb.rdsamp(file_path, channels=[0], sampfrom=0, sampto=signal_length // 2)
            annotation = wfdb.rdann(file_path, 'atr', sampfrom=0, sampto=signal_length // 2)
            signal_length = signal_length // 2
        elif separate == 2:
            signal, _ = wfdb.rdsamp(file_path, channels=[0], sampfrom=signal_length // 2, sampto=signal_length)
            annotation = wfdb.rdann(file_path, 'atr', sampfrom=signal_length // 2, sampto=signal_length)
            annotation.sample = annotation.sample - (info.sig_len - info.sig_len // 2)
            signal_length = signal_length - signal_length // 2
        else:
            signal, _ = wfdb.rdsamp(file_path, channels=[0])
            annotation = wfdb.rdann(file_path, 'atr')
        signal.astype(DATA_TYPE)
        signal = np.squeeze(signal)

        if info.fs != FREQUENCY_SAMPLING:
            signal_length = int(FREQUENCY_SAMPLING / info.fs * signal_length)
            signal = spysig.resample(signal, signal_length)
            annotation.sample = annotation.sample * FREQUENCY_SAMPLING / info.fs
            annotation.sample = annotation.sample.astype('int')

        signal.astype(DATA_TYPE)
        signal = butter_bandpass_filter(signal, 1, 30, 360, order=5)
        signal = baseline_wander_remove(signal, 360, 0.2, 0.6)
        signal = normalize(signal, int(0.5 * 360))

        input_scale = 0.003920781891793013
        input_zero_point = 0
        _signal = signal / input_scale + input_zero_point
        _signal = np.maximum(_signal, 0)
        _signal = _signal.astype(np.uint8)

        data_sample = []
        for i in range(signal_length - (NEIGHBOUR_POINT - 1)):
            data_sample.append(signal[i:i + NEIGHBOUR_POINT])
        data_sample = np.array(data_sample, dtype='float32')

        # Reshape data_sample to have a channel dimension
        data_sample = np.expand_dims(data_sample, axis=1)
        data_sample = np.expand_dims(data_sample, axis=-1)

        label = np.zeros((signal_length, 2), dtype='int8')
        for i in range(annotation.ann_len):
            if annotation.symbol[i] in ['+', '~', '|', '[', '!', ']', '"', 's', 'x']:
                continue
            # Set the label to 1 (positive class) at the positions of the annotations
            label[annotation.sample[i] - positive_range:annotation.sample[i] + positive_range + 1] = 1
        label = np.array(label[int(0.1 * FREQUENCY_SAMPLING):signal_length - int(0.3 * FREQUENCY_SAMPLING)],dtype='int8')
        label = torch.FloatTensor(label)

        indices = np.linspace(0, signal_length - (NEIGHBOUR_POINT - 1), sample_size, dtype=int)
        indices = [i for i in indices if i < len(data_sample)]
        data_sample = data_sample[indices]
        label = label[indices]
        data.append(data_sample)
        labels.append(label)

    data = np.concatenate(data, axis=0)
    labels = torch.cat(labels, dim=0)

    return data, labels

# Set the random seed for reproducibility
torch.manual_seed(42)

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Load data and preprocess
train_file_paths = ["mit-bih-arrhythmia-database-1.0.0/101.hea", "mit-bih-arrhythmia-database-1.0.0/106.hea",
                    "mit-bih-arrhythmia-database-1.0.0/108.hea", "mit-bih-arrhythmia-database-1.0.0/109.hea",
                    "mit-bih-arrhythmia-database-1.0.0/112.hea", "mit-bih-arrhythmia-database-1.0.0/114.hea",
                    "mit-bih-arrhythmia-database-1.0.0/115.hea", "mit-bih-arrhythmia-database-1.0.0/116.hea",
                    "mit-bih-arrhythmia-database-1.0.0/118.hea", "mit-bih-arrhythmia-database-1.0.0/119.hea",
                    "mit-bih-arrhythmia-database-1.0.0/122.hea",
                    "mit-bih-arrhythmia-database-1.0.0/124.hea", "mit-bih-arrhythmia-database-1.0.0/201.hea",
                    "mit-bih-arrhythmia-database-1.0.0/203.hea", "mit-bih-arrhythmia-database-1.0.0/205.hea",
                    "mit-bih-arrhythmia-database-1.0.0/207.hea", "mit-bih-arrhythmia-database-1.0.0/208.hea",
                    "mit-bih-arrhythmia-database-1.0.0/209.hea", "mit-bih-arrhythmia-database-1.0.0/215.hea",
                    "mit-bih-arrhythmia-database-1.0.0/220.hea", "mit-bih-arrhythmia-database-1.0.0/223.hea",
                    "mit-bih-arrhythmia-database-1.0.0/230.hea"]
test_file_paths = ["mit-bih-arrhythmia-database-1.0.0/100.hea", "mit-bih-arrhythmia-database-1.0.0/103.hea",
                   "mit-bih-arrhythmia-database-1.0.0/105.hea", "mit-bih-arrhythmia-database-1.0.0/111.hea",
                   "mit-bih-arrhythmia-database-1.0.0/113.hea", "mit-bih-arrhythmia-database-1.0.0/117.hea",
                   "mit-bih-arrhythmia-database-1.0.0/121.hea", "mit-bih-arrhythmia-database-1.0.0/123.hea",
                   "mit-bih-arrhythmia-database-1.0.0/200.hea", "mit-bih-arrhythmia-database-1.0.0/202.hea",
                   "mit-bih-arrhythmia-database-1.0.0/210.hea", "mit-bih-arrhythmia-database-1.0.0/212.hea",
                   "mit-bih-arrhythmia-database-1.0.0/213.hea", "mit-bih-arrhythmia-database-1.0.0/214.hea",
                   "mit-bih-arrhythmia-database-1.0.0/219.hea", "mit-bih-arrhythmia-database-1.0.0/221.hea",
                   "mit-bih-arrhythmia-database-1.0.0/222.hea", "mit-bih-arrhythmia-database-1.0.0/228.hea",
                   "mit-bih-arrhythmia-database-1.0.0/231.hea", "mit-bih-arrhythmia-database-1.0.0/232.hea",
                   "mit-bih-arrhythmia-database-1.0.0/233.hea", "mit-bih-arrhythmia-database-1.0.0/234.hea"]
train_data, train_labels = preprocess_data(train_file_paths)
test_data, test_labels = preprocess_data(test_file_paths)

# Convert data to Tensor
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)

# Modify the model instantiation to include the required changes
model = CNN().to(device)

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer with SGD and momentum
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# Define the number of epochs and mini-batch size
epochs = 10
mini_batch_size = 128

# Calculate the training set mean
train_mean = torch.mean(train_data, dim=0, keepdim=True)

# Modify the preprocessing step to subtract the training set mean
train_data -= train_mean
test_data -= train_mean

# Create DataLoader for training and testing datasets with the updated batch size
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False)

best_checkpoint_path = "best_checkpoint.pth"  # Đường dẫn đến checkpoint tốt nhất
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    best_accuracy = 0.0  # Khởi tạo biến best_accuracy
    for epoch in range(epochs):
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        model.train()  # Chuyển sang chế độ huấn luyện

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Tính toán accuracy
            predicted_labels = torch.sigmoid(outputs) >= 0.5
            total_correct += (predicted_labels.squeeze() == labels.squeeze()).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}/{epochs}: Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Lưu checkpoint nếu accuracy tốt hơn
        if accuracy > best_accuracy:
            if os.path.exists('best_checkpoint.pth'):
                os.remove('best_checkpoint.pth')
            best_accuracy = accuracy
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
            }
            torch.save(checkpoint, 'best_checkpoint.pth')

# Train the model with the updated configurations
train_model(model, train_loader, criterion, optimizer, device, epochs)

def clustering(test_labels):
    positive_point = np.where(test_labels == 1)[0]
    beat = []
    if len(positive_point) > 5:
        cluster = np.array([positive_point[0]])
        for i in range(1, len(positive_point)):
            if positive_point[i] - cluster[-1] > 0.08 * FREQUENCY_SAMPLING or i == len(positive_point) - 1:
                if i == len(positive_point) - 1:
                    cluster = np.append(cluster, positive_point[i])
                if cluster.shape[0] > 5:
                    beat.append(int(np.mean(cluster)))
                cluster = np.array([positive_point[i]])
            else:
                cluster = np.append(cluster, positive_point[i])

    return np.asarray(beat)

# Load the best checkpoint
checkpoint = torch.load('best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print("Best checkpoint loaded.")

def eval_model(model, test_loader, criterion, device):
    model.eval()  # Chuyển sang chế độ đánh giá (evaluation mode)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    test_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Tính toán accuracy
            predicted_labels = torch.sigmoid(outputs) >= 0.5
            total_correct += (predicted_labels.squeeze() == labels.squeeze()).sum().item()
            total_samples += labels.size(0)
            test_predictions.append(predicted_labels)

    test_predictions = torch.cat(test_predictions)
    eval_loss = total_loss / len(test_loader)
    eval_accuracy = total_correct / total_samples
    print(f"Eval Accuracy: {eval_accuracy:.4f}")
    print(f"Eval Loss: {eval_loss:.4f}")
    return test_predictions

# Đánh giá mô hình và nhận kết quả dự đoán
test_predictions = eval_model(model, test_loader, criterion, device)

def evaluate(test_file_paths, test_predictions):
    for file_path in test_file_paths:
        file_path = file_path[:-4]
        info = wfdb.rdheader(file_path)
        signal_length = info.sig_len
        annotation = wfdb.rdann(file_path, 'atr')

    fs = info.fs
    if fs != FREQUENCY_SAMPLING:
        signal_length = int(FREQUENCY_SAMPLING / fs * signal_length)
        annotation.sample = annotation.sample * FREQUENCY_SAMPLING / fs
        annotation.sample = annotation.sample.astype('int')

    condition = np.isin(annotation.symbol,
                        ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', '\'',
                         '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@'], invert=True)
    sample = np.extract(condition, annotation.sample)
    cluster = clustering(np.expand_dims(test_predictions[0:, 1], axis=1)) + int(0.1 * FREQUENCY_SAMPLING)
    window = int(0.075 * FREQUENCY_SAMPLING)

    recording = np.zeros(signal_length, dtype='int32')
    detection = np.zeros(signal_length, dtype='int32')

    np.put(recording, sample, 1)
    np.put(detection, cluster, 1)

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(sample)):
        if sum(detection[sample[i] - window:sample[i] + window + 1]) > 0:
            TP += 1
        else:
            FN += 1
    for i in range(len(cluster)):
        if sum(recording[cluster[i] - window:cluster[i] + window + 1]) == 0:
            FP += 1

    sensitivity = TP/ (TP + FN)
    positive_value = TP / (TP + FP)

    return TP, FN, FP, sensitivity, positive_value
TP, FN, FP, sensitivity, positive_value= evaluate(test_file_paths,test_predictions)
print(f"True Positive (TP): {TP}")
print(f"False Negative (FN): {FN}")
print(f"False Positive (FP): {FP}")
print(f"Sensitivity (SE): {sensitivity:.4f}")
print(f"Positive Predictivity (+P): {positive_value:.4f}")






