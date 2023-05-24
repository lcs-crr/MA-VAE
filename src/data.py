# Lucas Correia
# Mercedes-Benz AG, Stuttgart, Germany

"""
This is the script used to pre-process the data.
The raw data comes in the MF4 file format
"""

import pickle
import random
import tensorflow as tf
import numpy as np
import sklearn
from asammdf import MDF
import scipy
import os

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Set desired sampling frequency and filter order
freq_target = 2
filter_order = 1

# Specify sampling frequencies of the chosen channels
sampling_freqs = [freq_A, freq_B, freq_C, freq_D, freq_E, freq_F, freq_G, freq_H, freq_I,
                  freq_J, freq_K, freq_L, freq_M, freq_N, freq_O, freq_P, freq_Q, freq_R]


def butter_lowpass_filter(data, cutoff_freq, order):
    # Find filter parameters
    b, a = scipy.signal.butter(order, cutoff_freq, btype='low', analog=False)
    # Filter data
    filtered_data = scipy.signal.lfilter(b, a, data)
    return filtered_data


def mf4_parser(path):
    data = []
    # For MF4 file in directory
    for file in [f for f in os.listdir(path) if f.upper().endswith('.MF4')]:
        # Load MF4 file
        mdf = MDF(path + '\\' + file)

        # MF4 files are not sampled exactly equidistantly,
        # resample with the highest channel frequency to remedy
        mdf = mdf.resample(1/max(sampling_freqs))

        # Obtain desired signals from mdf object
        A = mdf.get('A')
        B = mdf.get('B')
        C = mdf.get('C')
        D = mdf.get('D')
        E = mdf.get('E')
        F = mdf.get('F')
        G = mdf.get('G')
        H = mdf.get('H')
        I = mdf.get('I')
        J = mdf.get('J')
        K = mdf.get('K')
        L = mdf.get('L')
        M = mdf.get('M')
        N = mdf.get('N')
        O = mdf.get('O')
        P = mdf.get('P')
        Q = mdf.get('Q')
        R = mdf.get('R')

        # Create list of signals to iterate later
        mf4_channel_list = [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R]

        temp = []
        for idx, channel in enumerate(mf4_channel_list):
            # Make sure dtype is float32
            channel = channel.astype("float32")
            channel.samples = channel.samples[::int(max(sampling_freqs) // sampling_freqs[idx])]
            channel.timestamps = channel.timestamps[::int(max(sampling_freqs) // sampling_freqs[idx])]

            if freq_target < sampling_freqs[idx]:  # Sampling frequency is higher than target frequency
                # Padding is applied to cut out oscillations caused by low-pass filtering
                pad_amount = 1000
                channel_pad = np.pad(channel.samples, (pad_amount, pad_amount), 'constant',
                                     constant_values=(channel.samples[0], channel.samples[-1]))
                # Apply low pass filter
                signal_new = butter_lowpass_filter(
                    channel_pad, freq_target // 2, sampling_freqs[idx], filter_order)
                # Trim padding form signal
                signal_new = signal_new[pad_amount:-pad_amount]
                # Resample signal
                signal_new = signal_new[::int(sampling_freqs[idx] // freq_target)]
                temp.append(signal_new)

            elif freq_target == sampling_freqs[idx]:  # Sampling frequency is equal to target frequency
                # Do nothing
                temp.append(channel.samples)

            elif freq_target > sampling_freqs[idx]:  # Sampling frequency is lower than target frequency
                # Create time array of original signal to interpolate
                time = np.round(channel.timestamps - channel.timestamps[0], 3)
                # Parametrise interpolation function
                f = scipy.interpolate.interp1d(time, channel.samples)
                # Create time array of new signal to interpolate
                time_new = np.arange(0, time[-1], 1 / freq_target)
                # Interpolate
                signal_new = f(time_new)
                temp.append(signal_new)

            # Truncate longer signals in case they don't match in size
            min_len = len(min(temp, key=len))
            for j in range(len(temp)):
                temp[j] = temp[j][:min_len]

            # Ensure no NaNs are present in data
            temp_array = np.vstack(temp).T
            if not np.any(np.isnan(temp_array)):
                data.append(temp_array)
    return data


# Parse normal data
normal_data = mf4_parser('NORMAL_LOAD_PATH')
# -> List of 1875 multivariate time-series of varying length

# Parse anomalous data
anomalous_data = mf4_parser('ANOMALOUS_LOAD_PATH')
# -> List of 60 multivariate time-series of varying length

# Shuffle normal data
random.shuffle(normal_data)

# Split normal_data into training and test data
train_list, test_list = sklearn.model_selection.train_test_split(
    normal_data, random_state=seed, test_size=0.3)
# -> List of 1312, 563 multivariate time-series of varying length

# Split training data into training and validation subsets
train_list, val_list = sklearn.model_selection.train_test_split(
    train_list, random_state=seed, test_size=0.2)
# -> List of 1049, 263 multivariate time-series of varying length


## Process training and validation data
# Find statistical metrics, such as mean and standard deviation
def find_stat_metrics(data):
    mean = np.mean(np.vstack(data), axis=0)
    std_dev = np.std(np.vstack(data), axis=0)
    return [mean, std_dev]


stat_metrics = find_stat_metrics(train_list)
# -> List of metrics


def standardise(data, stat_metrics):
    mean = stat_metrics[0]
    std_dev = stat_metrics[1]
    data_scaled = []
    for time_series in data:
        data_scaled.append((time_series - mean) / std_dev)
    return data_scaled


# Standardise data, i.e. transform it such that mean=0 and std_dev=1
scaled_train_list = list(standardise(train_list, stat_metrics))
# -> List of 1049 multivariate time-series of varying length
scaled_val_list = list(standardise(val_list, stat_metrics))
# -> List of 263 multivariate time-series of varying length


def window_list(data, window_size, window_shift):
    window_list_temp = []
    # For time series in data
    for time_series in data:
        # Find number of resulting windows given the window size and shift
        set_window_count = int((time_series.shape[0] - window_size) / window_shift + 1)
        # Pre-allocate output array
        window_data = np.zeros((set_window_count, window_size, time_series.shape[1]))
        # For number of resulting windows
        for window in range(set_window_count):
            # For features in time series
            for feature in range(time_series.shape[1]):
                window_data[window, :, feature] = \
                    time_series[window * window_shift:window_size + window * window_shift, feature]
        window_list_temp.append(window_data)
    # Concatenate list into 3D array of shape (number of windows, window size, features)
    windows = np.concatenate(window_list_temp[:], axis=0)
    return windows


# Specify window size, shift and features
window_size = 256
window_shift = 128
features = 13

# Window training and validation data with given window size and shift
scaled_train_window = window_list(scaled_train_list, window_size, window_shift)
# -> 3D array of shape (4937, window size, features)
scaled_val_window = window_list(scaled_val_list, window_size, window_shift)
# -> 3D array of shape (1216, window size, features)


def generator(data):
    for i in range(len(data)):
        yield data[i, :, :], data[i, :, :]


# Create tf.data objects
tf_train = tf.data.Dataset.from_generator(lambda: generator(scaled_train_window),
                                          output_types=(tf.as_dtype('float32'),
                                                        tf.as_dtype('float32')),
                                          output_shapes=(tf.TensorShape([window_size, features]),
                                                         tf.TensorShape([window_size, features])))

tf_val = tf.data.Dataset.from_generator(lambda: generator(scaled_val_window),
                                        output_types=(tf.as_dtype('float32'),
                                                      tf.as_dtype('float32')),
                                        output_shapes=(tf.TensorShape([window_size, features]),
                                                       tf.TensorShape([window_size, features])))

# Find number of windows in each array to enable shuffling later
n_train_data = len(scaled_train_window)
n_val_data = len(scaled_val_window)

# Define desired batch size
batch_size = 256

# Shuffle and batch data
tf_train = tf_train.shuffle(n_train_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)
tf_val = tf_val.shuffle(n_val_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Save training and validation data
tf.data.experimental.save(tf_train, 'SAVE_PATH')
tf.data.experimental.save(tf_val, 'SAVE_PATH')

## Process test data
# Standardise normal test data, i.e. transform it such that mean=0 and std_dev=1
scaled_list = list(standardise(test_list, stat_metrics))
# Pre-allocate list where each item represents the windowed time series
scaled_windowed_list = []
# For time series in test list
for time_series in scaled_list:
    # Window training and validation data with given window size and shift
    windowed_time_series_temp = window_list(list(time_series), window_size, window_size)
    # -> 3D array of shape (number of windows, window size, features)
    scaled_windowed_list.append(windowed_time_series_temp)
    # -> List of 3D arrays, each shape (number of windows, window size, features)
# Concatenate all list items into one array
scaled_normal_test_window = np.vstack(scaled_windowed_list)
# -> 3D array of shape (2606, window size, features)

# Standardise anomalous test data, i.e. transform it such that mean=0 and std_dev=1
scaled_list = list(standardise(anomalous_data, stat_metrics))
# Pre-allocate list where each item represents the windowed time series
scaled_windowed_list = []
# For time series in test list
for time_series in scaled_list:
    # Window training and validation data with given window size and shift
    windowed_time_series_temp = window_list(list(time_series), window_size, window_size)
    # -> 3D array of shape (number of windows, window size, features)
    scaled_windowed_list.append(windowed_time_series_temp)
    # -> List of 3D arrays, each shape (number of windows, window size, features)
# Concatenate all list items into one array
scaled_anomalous_test_window = np.vstack(scaled_windowed_list)
# -> 3D array of shape (225, window size, features)

# Save normal and anomalous test data as pickle files
with open('SAVE_PATH', 'wb') as f:
    pickle.dump(scaled_normal_test_window, f)

with open('SAVE_PATH', 'wb') as f:
    pickle.dump(scaled_anomalous_test_window, f)