# Lucas Correia
# Mercedes-Benz AG, Stuttgart, Germany

"""
This is the script used to train the SISVAE model.
10.1109/TNNLS.2020.2980749
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


class SISVAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(SISVAE, self).__init__()

        # Model
        self.encoder = encoder
        self.decoder = decoder

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.log_probs_loss_tracker = tf.keras.metrics.Mean(name="log_probs_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.smooth_loss_tracker = tf.keras.metrics.Mean(name="smooth_loss")

    @tf.function
    def loss_fn(self, X, Xhat, Xhat_mean, Xhat_log_var, z_mean, z_log_var):
        # Calculate log probability of data belonging to parametrised distribution
        log_probs_loss = self.evaluate_log_prob(X, loc=Xhat_mean, scale=tf.sqrt(tf.math.exp(Xhat_log_var)))
        # Reduce log probability to single value per batch
        log_probs_loss = tf.reduce_sum(log_probs_loss, axis=1)
        # Calculate KL Divergence between latent distribution and Gaussian distribution
        kl_loss = tfp.distributions.kl_divergence(
            tfp.distributions.Normal(loc=0., scale=1.),
            tfp.distributions.Normal(loc=z_mean, scale=tf.sqrt(tf.math.exp(z_log_var))))
        # Reduce KL divergence to single value per batch
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        # Calculate KL Divergence between current latent distribution and t-1 latent distribution
        smooth_loss = [tfp.distributions.kl_divergence(
            tfp.distributions.Normal(loc=z_mean[:, 0], scale=tf.sqrt(tf.math.exp(z_log_var[:, 0]))),
            tfp.distributions.Normal(loc=z_mean[:, 0], scale=tf.sqrt(tf.math.exp(z_log_var[:, 0]))))]
        for time_step in range(1, kl_loss.shape[1]):
            smooth_loss.append(tfp.distributions.kl_divergence(
                tfp.distributions.Normal(loc=z_mean[:, time_step - 1],
                                         scale=tf.sqrt(tf.math.exp(z_log_var[:, time_step - 1]))),
                tfp.distributions.Normal(loc=z_mean[:, time_step],
                                         scale=tf.sqrt(tf.math.exp(z_log_var[:, time_step])))))
        smooth_loss = tf.transpose(tf.stack(smooth_loss), perm=[1, 0, 2])
        # Reduce smooth loss to single value per batch
        smooth_loss = tf.reduce_mean(tf.reduce_sum(smooth_loss, axis=1))
        return -log_probs_loss, kl_loss, smooth_loss

    @tf.function
    def evaluate_log_prob(self, sample, loc, scale):
        # Configure distribution with parameters output from decoder
        output_dist = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        # Calculate log probability of sample belongs parametrised distribution
        log_probs = output_dist.unnormalized_log_prob(sample)
        return log_probs

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
            log_probs_loss, kl_loss, smooth_loss = self.loss_fn(
                X,
                Xhat,
                Xhat_mean,
                Xhat_log_var,
                z_mean,
                z_log_var
            )
            # Calculate total loss from different losses
            total_loss = log_probs_loss + kl_loss + 0.5 * smooth_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.log_probs_loss_tracker.update_state(log_probs_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.smooth_loss_tracker.update_state(smooth_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "log_probs_loss": self.log_probs_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "smooth_loss": self.smooth_loss_tracker.result(),
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
        log_probs_loss, kl_loss, smooth_loss = self.loss_fn(
            X,
            Xhat,
            Xhat_mean,
            Xhat_log_var,
            z_mean,
            z_log_var
        )
        total_loss = log_probs_loss + kl_loss + 0.5 * smooth_loss
        self.total_loss_tracker.update_state(total_loss)
        self.log_probs_loss_tracker.update_state(log_probs_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.smooth_loss_tracker.update_state(smooth_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.log_probs_loss_tracker,
            self.kl_loss_tracker,
            self.smooth_loss_tracker,
        ]

    @tf.function
    def call(self, X, **kwargs):
        # Encoder is fed with input window
        z_mean, z_log_var, z = self.encoder(X, **kwargs)
        # Decoder is fed with the sampled latent matrix from encoder
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(z, **kwargs)
        return Xhat_mean, Xhat_log_var, Xhat, z_mean, z_log_var, z


class VAE_Encoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features):
        super(VAE_Encoder, self).__init__()

        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()

    def build_encoder(self):
        # Input
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        # GRU layer
        gru = tfkl.GRU(200, return_sequences=True)(enc_input)
        # Transform deterministic BiLSTM output into distribution parameters z_mean and z_log_var
        z_mean = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, name="z_mean"))(gru)
        z_log_var = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, name="z_log_var"))(gru)
        # Create distribution object for reparametrisation trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        # Get epsilon for reparametrisation trick
        eps = output_dist.sample(tf.shape(z_mean))
        # Reparametrisation trick
        z = z_mean + tf.sqrt(tf.math.exp(z_log_var)) * eps
        return tf.keras.Model(enc_input, [z_mean, z_log_var, z], name="encoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.encoder(inputs, **kwargs)


class VAE_Decoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features):
        super(VAE_Decoder, self).__init__()
        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.decoder = self.build_decoder()

    def build_decoder(self):
        # Latent vector input
        latent_input = tfkl.Input(shape=(self.seq_len, self.latent_dim,))
        # GRU layer
        gru = tfkl.GRU(200, return_sequences=True)(latent_input)
        # Transform deterministic GRU output into distribution parameters z_mean and z_log_var
        Xhat_mean = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat_mean")(gru)
        Xhat_log_var = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat_log_var")(gru)
        # Create distribution object for reparametrisation trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        # Get epsilon for reparametrisation trick
        eps = output_dist.sample(tf.shape(Xhat_mean))
        # Reparametrisation trick
        Xhat = Xhat_mean + tf.sqrt(tf.math.exp(Xhat_log_var)) * eps
        return tf.keras.Model(latent_input, [Xhat_mean, Xhat_log_var, Xhat], name="decoder")

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

encoder = VAE_Encoder(seq_len=256, latent_dim=40, features=13)
decoder = VAE_Decoder(seq_len=256, latent_dim=40, features=13)
model = SISVAE(encoder, decoder)
optimiser = tf.keras.optimizers.Adam(amsgrad=True)
model.compile(optimizer=optimiser)

# Fit vae model
history = model.fit(tf_train,
                    epochs=10000,
                    callbacks=[es],
                    validation_data=tf_val,
                    verbose=1
                    )
