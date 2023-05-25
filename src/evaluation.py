# Lucas Correia
# Mercedes-Benz AG, Stuttgart, Germany

"""
This is the script used to evaluate the model discussed.
"""

import numpy as np
import pickle
import random
import tensorflow as tf
import tensorflow_probability as tfp
import sklearn

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Load validation data
with open('LOAD_PATH', 'rb') as f:
    validation_data = pickle.load(f)
    # -> List multivariate time-series of varying length

# Load normal test data
with open('LOAD_PATH', 'rb') as f:
    normal_test_data = pickle.load(f)
    # -> List multivariate time-series of varying length

# Load anomalous test data
with open('LOAD_PATH', 'rb') as f:
    anomalous_test_data = pickle.load(f)
    # -> List multivariate time-series of varying length


def window(data, window_size, shift):
    # Find number of resulting windows given the window size and shift
    set_window_count = (data.shape[0] - window_size) // shift + 1
    # Pre-allocate output array
    window_data = np.zeros((set_window_count, window_size, data.shape[1]))
    # For number of resulting windows
    for j in range(set_window_count):
        window_data[j] = data[j * shift:window_size + j * shift]
    return window_data


def rev_window(windows, shift, mode):
    if mode == 'last':  # Whole first window, then all last time steps of following windows
        data = np.zeros(((windows.shape[0] - 1) * shift + windows.shape[1], windows.shape[2]))  # Pre-allocate array
        data[:windows.shape[1], :] = windows[0, :, :]  # First whole window, then all last time steps of windows
        for i in range(windows.shape[0] - 1):
            data[windows.shape[1] + shift * i: shift * (i + 1) + windows.shape[1], :] = windows[(i + 1), -1 * shift:, :]
    elif mode == 'first':
        data = np.zeros(((windows.shape[0] - 1) * shift + windows.shape[1], windows.shape[2]))  # Pre-allocate array
        data[-windows.shape[1]:, :] = windows[-1, :, :]  # Last window
        for i in range(windows.shape[0] - 1):  # All first time steps of windows, then whole last window
            data[shift * i: shift * (i + 1), :] = windows[i, :shift, :]
    elif mode == 'mean':
        data = np.zeros((windows.shape[0], (windows.shape[0] - 1) * shift + windows.shape[1],
                         windows.shape[2])) + np.nan  # Pre-allocate array
        for i in range(windows.shape[0]):
            data[i, i:i + windows.shape[1], :] = windows[i]
        data = np.nanmean(data, axis=0)
    return data


def evaluate_vmsa_vae(model, data, rev_mode, window_size):
    # Window data
    shift = 1
    test_data = window(data, window_size, shift)
    # Inference
    reconstruction = model.predict(data,
                                   batch_size=1024,
                                   verbose=0,
                                   steps=None,
                                   callbacks=None)

    # Extract mean and log variance from output
    mean = reconstruction[0]
    log_var = reconstruction[1]
    # Convert log variance to variance
    var = tf.math.exp(log_var)
    # Reverse-window
    mean = rev_window(mean, 1, rev_mode)
    var = rev_window(var, 1, rev_mode)
    # Convert variance to standard deviation
    std = tf.sqrt(var)
    # Create distribution with parameters
    output_dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
    # Calculate negative log likelihood
    log_probs = tf.expand_dims(-output_dist.unnormalized_log_prob(test_data), axis=1).numpy()
    return log_probs, [mean, std]


# Load model
model = tf.keras.models.load_model('LOAD_PATH')

rev_mode = 'mean'
window_size = 256

val_score = []
for sequence in validation_data:
    # Inference on validation data
    score, recon = evaluate_vmsa_vae(model, sequence, rev_mode, window_size)
    # Reduce windows to scalars through percentile
    val_score.append(np.percentile(score, 100))
val_score = np.vstack(val_score)
# Set unsupervised threshold
threshold = np.percentile(val_score, 100)

# Evaluate normal test data to obtain TN and FP
normal_test_score = []
for sequence in normal_test_data:
    # Inference on sequence
    score, recon = evaluate_vmsa_vae(model, sequence, rev_mode, window_size)
    score = np.percentile(score, 100)
    # Find maximum error in anomaly score sequence
    normal_test_score.append(score)
# Obtain predicted labels
predicted_labels_normal = normal_test_score >= threshold


# Evaluate anomalous test data to obtain TP and FN
anomalous_test_score = []
for sequence in anomalous_test_data:
    # Inference on sequence
    score, recon = evaluate_vmsa_vae(model, sequence, rev_mode, window_size)
    # Find maximum error in anomaly score sequence
    score = np.percentile(score, 100)
    anomalous_test_score.append(score)
# Obtain predicted labels
predicted_labels_anomaly = anomalous_test_score >= threshold

# Concatenate predicted label vectors
predicted_labels_all = np.concatenate((predicted_labels_normal, predicted_labels_anomaly), axis=0)
# Create true label vectors
true_labels_normal = np.zeros_like(predicted_labels_normal)
true_labels_anomaly = np.ones_like(predicted_labels_anomaly)
true_labels_all = np.concatenate((true_labels_normal, true_labels_anomaly), axis=0)

# Calculate precision, recall and F1 metrics for unsupervised threshold
precision = sklearn.metrics.precision_score(true_labels_all, predicted_labels_all)
recall = sklearn.metrics.recall_score(true_labels_all, predicted_labels_all)
f1 = sklearn.metrics.f1_score(true_labels_all, predicted_labels_all)

precision_list = []
recall_list = []
f1_list = []
# Specify percentile range to run for loop over
percentile_array = np.arange(0, 100.1, 0.1)
for percentile in percentile_array:
    # Set temporary threshold
    threshold_temp = np.percentile(np.concatenate((normal_test_score, anomalous_test_score)), percentile)
    # Obtain temporary predicted labels
    predicted_labels_normal = normal_test_score >= threshold_temp
    predicted_labels_anomaly = anomalous_test_score >= threshold_temp
    # Concatenate temporary predicted label vectors
    predicted_labels_all = np.concatenate((predicted_labels_normal, predicted_labels_anomaly), axis=0)
    # Calculate temporary precision, recall and F1 metrics
    precision_temp = sklearn.metrics.precision_score(true_labels_all, predicted_labels_all)
    recall_temp = sklearn.metrics.recall_score(true_labels_all, predicted_labels_all)
    f1_temp = sklearn.metrics.f1_score(true_labels_all, predicted_labels_all)
    precision_list.append(precision_temp)
    recall_list.append(recall_temp)
    f1_list.append(f1_temp)
precision_array = np.vstack(precision_list)
recall_array = np.vstack(recall_list)
f1_array = np.vstack(f1_list)
# Calculate area under the precision-recall curve
auc = sklearn.metrics.auc(recall_array[:, 0], precision_array[:, 0])
# Set threshold at theoretical maximum F1 score
threshold_best = np.percentile(np.concatenate((normal_test_score, anomalous_test_score)),
                               percentile_array[np.argmax(f1_array)])

# Re-evaluate normal test data with ideal threshold
predicted_labels_normal = normal_test_score >= threshold_best

# Re-evaluate anomalous test data with ideal threshold
predicted_labels_anomaly = anomalous_test_score >= threshold_best
# Concatenate predicted label vectors
predicted_labels_all = np.concatenate((predicted_labels_normal, predicted_labels_anomaly), axis=0)

# Calculate theoretical maximum F1 score
f1_best = sklearn.metrics.f1_score(true_labels_all, predicted_labels_all)
