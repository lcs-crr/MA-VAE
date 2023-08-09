# Lucas Correia
# Mercedes-Benz AG, Stuttgart, Germany

"""
This is the script used to train the OmniAnomaly model.
10.1145/3292500.3330672
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random
import tf.keras.layers as tfkl

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


class OmniAnomaly(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(OmniAnomaly, self).__init__()

        # Model
        self.encoder = encoder
        self.decoder = decoder

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.log_probs_loss_tracker = tf.keras.metrics.Mean(name="log_probs_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.lgssm_loss_tracker = tf.keras.metrics.Mean(name="lgssm_loss")

    @tf.function
    def loss_fn(self, X, Xhat, Xhat_mean, Xhat_log_var, z, z_mean, z_log_var):
        # Calculate log probability of data belonging to parametrised distribution
        log_probs_loss = self.evaluate_log_prob(X, loc=Xhat_mean, scale=tf.sqrt(tf.math.exp(Xhat_log_var)))
        # Reduce log probability to single value per batch
        log_probs_loss = tf.reduce_sum(log_probs_loss, axis=1)
        # Calculate KL Divergence between latent distribution and Gaussian distribution
        kl_loss = tfp.distributions.kl_divergence(
            tfp.distributions.Normal(loc=0., scale=1.),
            tfp.distributions.Normal(loc=z_mean, scale=tf.sqrt(tf.math.exp(z_log_var))))
        # Reduce KL divergence to single value
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        # Calculate log probability of sample belongs parametrised distribution
        lgssm_loss = self.evaluate_lgssm(z)
        # Reduce LGSSM loss to single value
        lgssm_loss = tf.reduce_mean(lgssm_loss)
        return -log_probs_loss, kl_loss, -lgssm_loss

    @tf.function
    def evaluate_log_prob(self, sample, loc, scale):
        # Configure distribution with parameters output from decoder
        output_dist = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        # Calculate log probability of sample belongs parametrised distribution
        log_probs = output_dist.unnormalized_log_prob(sample)
        return log_probs

    @tf.function
    def evaluate_lgssm(self, sample):
        # Configure distribution with parameters output from encoder
        lgssm_dist = tfp.distributions.LinearGaussianStateSpaceModel(
            num_timesteps=sample.shape[1],
            transition_matrix=tf.linalg.LinearOperatorIdentity(sample.shape[-1]),
            transition_noise=tfp.distributions.MultivariateNormalDiag(scale_diag=tf.ones([sample.shape[-1]])),
            observation_matrix=tf.linalg.LinearOperatorIdentity(sample.shape[-1]),
            observation_noise=tfp.distributions.MultivariateNormalDiag(scale_diag=tf.ones([sample.shape[-1]])),
            initial_state_prior=tfp.distributions.MultivariateNormalDiag(scale_diag=tf.ones([sample.shape[-1]]))
        )
        # Calculate log probability of sample belongs parametrised distribution
        lgssm_loss = lgssm_dist.unnormalized_log_prob(sample)
        return lgssm_loss

    @tf.function
    def train_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        with tf.GradientTape() as tape:
            # Forward pass through encoder
            z_mean, z_log_var, z = self.encoder(X, training=True)
            # Forward pass through decoder
            Xhat_mean, Xhat_log_var, Xhat = self.decoder(z, training=True)
            # Calculate losses from parameters
            log_probs_loss, kl_loss, lgssm_loss = self.loss_fn(
                X,
                Xhat,
                Xhat_mean,
                Xhat_log_var,
                z,
                z_mean,
                z_log_var
            )
            # Calculate total loss from different losses
            total_loss = log_probs_loss + kl_loss + lgssm_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.log_probs_loss_tracker.update_state(log_probs_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.lgssm_loss_tracker.update_state(lgssm_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "log_probs_loss": self.log_probs_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "lgssm_loss": self.lgssm_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        # Forward pass through encoder
        z_mean, z_log_var, z = self.encoder(X, training=False)
        # Forward pass through decoder
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(z, training=False)
        # Calculate losses from parameters
        log_probs_loss, kl_loss, lgssm_loss = self.loss_fn(
            X,
            Xhat,
            Xhat_mean,
            Xhat_log_var,
            z,
            z_mean,
            z_log_var
        )
        total_loss = log_probs_loss + kl_loss + lgssm_loss
        self.total_loss_tracker.update_state(total_loss)
        self.log_probs_loss_tracker.update_state(log_probs_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.lgssm_loss_tracker.update_state(lgssm_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.log_probs_loss_tracker,
            self.kl_loss_tracker,
            self.lgssm_loss_tracker,
        ]

    @tf.function
    def call(self, inputs, **kwargs):
        # Forward pass through encoder
        z_mean, z_log_var, z = self.encoder(inputs, training=False)
        # Forward pass through decoder
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(z, training=False)
        return Xhat_mean, Xhat_log_var, Xhat, z_mean, z_log_var, z


class Encoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features):
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()

    def build_encoder(self):
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        x = tfkl.GRU(500, return_sequences=True, activity_regularizer=tf.keras.regularizers.L2(l2=1e-4))(enc_input)
        h = tfkl.TimeDistributed(
            tfkl.Dense(500, activation="relu", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(x)
        z_mean = tfkl.TimeDistributed(
            tfkl.Dense(self.latent_dim, name="z_mean", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        z_log_var = tfkl.TimeDistributed(
            tfkl.Dense(self.latent_dim, name="z_log_var", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        eps = output_dist.sample(tf.shape(z_mean))
        z = z_mean + tf.sqrt(tf.math.exp(z_log_var)) * eps + 1e-4
        K = 20
        # planar normalizing flow
        for k in range(K):
            z = z + tfkl.Dense(self.latent_dim, use_bias=False)(tfkl.Dense(self.latent_dim, activation='tanh')(z))
        return tf.keras.Model(enc_input, [z_mean, z_log_var, z], name="encoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.encoder(inputs, **kwargs)


class Decoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.decoder = self.build_decoder()

    def build_decoder(self):
        dec_input = tfkl.Input(shape=(self.seq_len, self.latent_dim))
        x = tfkl.GRU(500, return_sequences=True, activity_regularizer=tf.keras.regularizers.L2(l2=1e-4))(dec_input)
        h = tfkl.TimeDistributed(
            tfkl.Dense(500, activation="relu", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(x)
        Xhat_mean = tfkl.TimeDistributed(
            tfkl.Dense(self.features, name="Xhat_mean", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        Xhat_log_var = tfkl.TimeDistributed(
            tfkl.Dense(self.features, name="Xhat_log_var", activity_regularizer=tf.keras.regularizers.L2(l2=1e-4)))(h)
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        eps = output_dist.sample(tf.shape(Xhat_mean))
        Xhat = Xhat_mean + tf.sqrt(tf.math.exp(Xhat_log_var)) * eps
        return tf.keras.Model(dec_input, [Xhat_mean, Xhat_log_var, Xhat], name="decoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)


# Establish callbacks
es = tf.keras.callbacks.EarlyStopping(monitor='val_log_probs_loss',
                                      mode='min',
                                      verbose=1,
                                      patience=250,
                                      restore_best_weights=True,
                                      )

tf_train = tf.data.Dataset.load('LOAD_PATH')
tf_val = tf.data.Dataset.load('LOAD_PATH')

tf_train = tf_train.unbatch().batch(512)
tf_val = tf_val.unbatch().batch(512)

encoder = Encoder(seq_len=256, latent_dim=3, features=13)
decoder = Decoder(seq_len=256, latent_dim=3, features=13)
model = OmniAnomaly(encoder, decoder)
optimiser = tf.keras.optimizers.Adam(amsgrad=True, clipnorm=10)
model.compile(optimizer=optimiser)

# Fit vae model
history = model.fit(tf_train,
                    epochs=10000,
                    callbacks=es,
                    validation_data=tf_val,
                    verbose=1
                    )
