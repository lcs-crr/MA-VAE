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
sampling_freqs = [freq_1, freq_2, freq_3, freq_4, freq_5, freq_6, freq_7,
                  freq_8, freq_9, freq_10, freq_11, freq_12, freq_13]


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
        channel1 = mdf.get('Channel1')
        channel2 = mdf.get('Channel2')
        channel3 = mdf.get('Channel3')
        channel4 = mdf.get('Channel4')
        channel5 = mdf.get('Channel5')
        channel6 = mdf.get('Channel6')
        channel7 = mdf.get('Channel7')
        channel8 = mdf.get('Channel8')
        channel9 = mdf.get('Channel9')
        channel10 = mdf.get('Channel10')
        channel11 = mdf.get('Channel11')
        channel12 = mdf.get('Channel12')
        channel13 = mdf.get('Channel13')


        # Create list of signals to iterate later
        mf4_channel_list = [channel1, channel2, channel3, channel4, channel5, channel6, channel7,
                            channel8, channel9, channel10, channel11, channel12, channel13]

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
# -> List of multivariate time-series of varying length

# Parse anomalous data
anomalous_data = mf4_parser('ANOMALOUS_LOAD_PATH')
# -> List of multivariate time-series of varying length

meas_time = []
for sequence in normal_data:
    meas_time.append(len(sequence) / (2*60))  # 2*60 converts samples to minutes
# Find total dynamic testing time
cumsum = np.cumsum(meas_time)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# Find index of sequence that comes closest to 1h of testing time
_, idx_1 = find_nearest(cumsum, 1)
# Find index of sequence that comes closest to 8h of testing time
_, idx_8 = find_nearest(cumsum, 8)
# Find index of sequence that comes closest to 64h of testing time
_, idx_64 = find_nearest(cumsum, 64)
# Find index of sequence that comes closest to 512h of testing time
_, idx_512 = find_nearest(cumsum, 512)

train_list_1h = normal_data[:idx_1]
train_list_8h = normal_data[:idx_8]
train_list_64h = normal_data[:idx_64]
train_list_512h = normal_data[:idx_512]
train_list_list = [train_list_1h, train_list_8h, train_list_64h, train_list_512h]

test_list = normal_data[idx_512:]

versions = ['1h', '8h', '64h', '512h']

for i, train_list in enumerate(train_list_list):
    train_list, val_list = sklearn.model_selection.train_test_split(
        train_list, random_state=seed, test_size=0.2)
    # -> Two lists multivariate time-series of varying length

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
    # -> List multivariate time-series of varying length
    scaled_val_list = list(standardise(val_list, stat_metrics))
    # -> List multivariate time-series of varying length


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
    # -> 3D array of shape (number of windows, window size, features)
    scaled_val_window = window_list(scaled_val_list, window_size, window_shift)
    # -> 3D array of shape (number of windows, window size, features)


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
    batch_size = 512

    # Shuffle and batch data
    tf_train = tf_train.shuffle(n_train_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    tf_val = tf_val.shuffle(n_val_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Save training and validation data
    tf.data.experimental.save(tf_train, 'SAVE_PATH')
    tf.data.experimental.save(tf_val, 'SAVE_PATH')

    ## Process test data
    # Standardise normal test data, i.e. transform it such that mean=0 and std_dev=1
    scaled_normal_test_list = list(standardise(test_list, stat_metrics))
    # -> List multivariate time-series of varying length

    # Standardise anomalous test data, i.e. transform it such that mean=0 and std_dev=1
    scaled_anomalous_test_list = list(standardise(anomalous_data, stat_metrics))
    # -> List multivariate time-series of varying length

    # Save normal and anomalous test data as pickle files
    with open(os.path.join('SAVE_PATH', versions[i], 'FILE_NAME', 'wb')) as f:
        pickle.dump(scaled_normal_test_list, f)

    with open(os.path.join('SAVE_PATH', versions[i], 'FILE_NAME', 'wb')) as f:
        pickle.dump(scaled_anomalous_test_list, f)
