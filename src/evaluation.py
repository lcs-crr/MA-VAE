# Lucas Correia
# Mercedes-Benz AG, Stuttgart, Germany

"""
This is the script used to evaluate the models discussed.
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

tf_val = tf.data.experimental.load(r'LOAD_PATH')

# Load normal test data
with open('LOAD_PATH', 'rb') as f:
    normal_test_data = pickle.load(f)
    # -> 3D array of shape (2606, window size, features)

# Load anomalous test data
with open('LOAD_PATH', 'rb') as f:
    anomalous_test_data = pickle.load(f)
    # -> 3D array of shape (225, window size, features)


def evaluate_vmsa_vae(model, data, samples):
    # If data is tf-data (validation data) convert to numpy array
    if type(data) is not np.ndarray:
        test_data_list = []
        for batch in data:
            test_data_list.append(batch[0].numpy())
        test_data = np.vstack(test_data_list)

    sample_error_list = []
    # Run model L times and average
    for L in range(samples):
        # Inference
        output = model.predict(test_data, verbose=0)
        # Extract mean and log variance from output
        mean = output[0]
        log_var = output[1]
        # Transform log variance to standard deviation
        std_dev = tf.math.exp(0.5 * log_var)
        # Create distribution with parameters
        output_dist = tfp.distributions.Normal(loc=mean, scale=std_dev)
        # Calculate negative log likelihood
        log_lik = tf.reduce_mean(-output_dist.unnormalized_log_prob(test_data), axis=-1)
        sample_error_list.append(np.expand_dims(log_lik, axis=0))
    sample_error_array = np.vstack(sample_error_list)
    return tf.reduce_mean(sample_error_array, axis=0)


# Load model
model = tf.keras.models.load_model('LOAD_PATH')
# Inference on validation data
score_normal_val = evaluate_vmsa_vae(model, tf_val, 5)
# Specify window percentile
window_percentile = 85
# Reduce windows to scalars through percentile
score_normal_val = np.percentile(score_normal_val, window_percentile, axis=-1)
# Set threshold
threshold = np.percentile(score_normal_val, 99)

# Create label vectors
true_labels_normal = np.zeros(len(normal_test_data))
# -> 1D array of shape (2606,)
true_labels_anomaly = np.zeros(len(anomalous_test_data))
# -> 1D array of shape (225,)

# Concatenate label vectors
true_labels_all = np.concatenate((true_labels_normal, true_labels_anomaly), axis=0)

# Inference on test data
score_normal = evaluate_vmsa_vae(model, normal_test_data, 5)
# -> 3D array of shape (2606, window size, 1)
score_anomaly = evaluate_vmsa_vae(model, anomalous_test_data, 5)
# -> 3D array of shape (225, window size, 1)

# Reduce windows to scalars through percentile
score_normal = np.percentile(score_normal, window_percentile, axis=1)
score_anomaly = np.percentile(score_anomaly, window_percentile, axis=1)

# Obtain predicted label vectors
predicted_labels_normal = score_normal >= threshold
predicted_labels_anomaly = score_anomaly >= threshold

# Concatenate label vectors
predicted_labels_all = np.concatenate((predicted_labels_normal, predicted_labels_anomaly), axis=0)

# Calculate precision, recall and F1 metrics
precision = sklearn.metrics.precision_score(true_labels_all, predicted_labels_all)
recall = sklearn.metrics.recall_score(true_labels_all, predicted_labels_all)
f1 = sklearn.metrics.f1_score(true_labels_all, predicted_labels_all)

# Find precision and recall values at different thresholds
precision_list = []
recall_list = []
# Specify percentile range to run for loop over
percentile_array = np.arange(0, 100.1, 0.1)
for percentile in percentile_array:
    # Obtain threshold
    threshold_temp = np.percentile(np.concatenate((score_normal, score_anomaly)), percentile)
    # Obtain predicted label vectors
    predicted_labels_normal = score_normal >= threshold_temp
    predicted_labels_anomaly = score_anomaly >= threshold_temp
    # Concatenate label vectors
    predicted_labels_all = np.concatenate((predicted_labels_normal, predicted_labels_anomaly), axis=0)
    # Calculate precision and recall metrics
    precision_temp = sklearn.metrics.precision_score(true_labels_all, predicted_labels_all)
    recall_temp = sklearn.metrics.recall_score(true_labels_all, predicted_labels_all)
    precision_list.append(precision_temp)
    recall_list.append(recall_temp)
precision_list = np.vstack(precision_list)
recall_list = np.vstack(recall_list)
# Calculate area under the precision recall curve
a_prc = sklearn.metrics.auc(recall_list[:, 0], precision_list[:, 0])