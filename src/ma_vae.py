# Lucas Correia
# Mercedes-Benz AG, Stuttgart, Germany

"""
This is the script used to train the MA-VAE model.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Bidirectional, LSTM, \
    TimeDistributed, MultiHeadAttention

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


class VAE_kl_annealing(tf.keras.callbacks.Callback):
    def __init__(self, annealing_epochs=30, type="normal", grace_period=20, start=0.0001, end=0.1,
                 lower_initial_betas=False
                 ):
        super(VAE_kl_annealing, self).__init__()
        self.annealing_epochs = annealing_epochs
        self.type = type
        self.grace_period = grace_period
        self.grace_period_idx = np.maximum(0, grace_period - 1)  # Starting from 0
        self.start = start
        self.end = end
        if type in ["cyclical", "monotonic"]:
            self.beta_values = np.linspace(start, end, annealing_epochs)
            if lower_initial_betas:
                self.beta_values[:annealing_epochs // 2] = self.beta_values[:annealing_epochs // 2] / 2

    def on_epoch_begin(self, epoch, logs=None):
        shifted_epochs = tf.math.maximum(0.0, epoch - self.grace_period_idx)
        if epoch < self.grace_period_idx or type == "normal":
            step_size = (self.start / self.grace_period)
            new_value = step_size * (epoch % self.grace_period)
            self.model.beta.assign(new_value)
        elif self.type == "monotonic":
            new_value = self.beta_values[min(epoch, self.annealing_epochs - 1)]
            self.model.beta.assign(new_value)
        elif self.type == "cyclical":
            new_value = self.beta_values[int(shifted_epochs % self.annealing_epochs)]
            self.model.beta.assign(new_value)
        print(f"Beta value: {self.model.beta.numpy():.10f}, cycle epoch {int(shifted_epochs % self.annealing_epochs)}")


class MA_VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, ma, beta=1e-8):
        super(MA_VAE, self).__init__()

        # Model
        self.encoder = encoder
        self.decoder = decoder
        self.ma = ma

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.log_probs_loss_tracker = tf.keras.metrics.Mean(name="log_probs_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")

        # Modifiable weight for KL-loss
        self.beta = tf.Variable(beta, trainable=False)  # Weight for KL-Loss, can be modified with a callback

    @tf.function
    def stoch_vae_loss_fn(self, X, Xhat, Xhat_mean, Xhat_log_var, z_mean, z_log_var):
        # Calculate log probability of data belonging to parametrised distribution
        log_probs_loss = self.evaluate_log_prob(X, loc=Xhat_mean, scale=tf.math.exp(0.5 * Xhat_log_var))
        # Calculate KL Divergence between latent distribution and Gaussian distribution
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        # Reduce KL divergence to single value
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        # Calculate mean-squared error between encoder input and sampled decoder output, not used
        recon_loss = tf.keras.losses.MeanSquaredError()(X, Xhat)
        return -log_probs_loss, kl_loss, recon_loss

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
            z_mean, z_log_var, z, states = self.encoder(X, training=True)
            # Forward pass through MA
            A = self.ma([X, z], training=True)
            # Forward pass through decoder
            Xhat_mean, Xhat_log_var, Xhat = self.decoder(A, training=True)
            # Calculate losses from parameters
            log_probs_loss, kl_loss, recon_loss = self.stoch_vae_loss_fn(
                X,
                Xhat,
                Xhat_mean,
                Xhat_log_var,
                z_mean,
                z_log_var
            )
            # Calculate total loss from different losses
            total_loss = log_probs_loss + self.beta * kl_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.log_probs_loss_tracker.update_state(log_probs_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "log_probs_loss": self.log_probs_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "rec_loss": self.reconstruction_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        # # Forward pass
        Xhat_mean, Xhat_log_var, Xhat, z_mean, z_log_var, z, A = self(X, training=False)
        # Calculate losses from parameters
        log_probs_loss, kl_loss, recon_loss = self.stoch_vae_loss_fn(
            X,
            Xhat,
            Xhat_mean,
            Xhat_log_var,
            z_mean,
            z_log_var
        )
        total_loss = log_probs_loss + self.beta * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.log_probs_loss_tracker.update_state(log_probs_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.log_probs_loss_tracker,
            self.kl_loss_tracker,
            self.reconstruction_loss_tracker,
        ]

    @tf.function
    def call(self, inputs, **kwargs):
        # Encoder is fed with input window
        z_mean, z_log_var, z, states = self.encoder(inputs, **kwargs)
        # Mean matrix of latent distribution is passed to MA mechanism
        A = self.ma([inputs, z_mean], **kwargs)
        # Decoder is fed with the attention matrix from MA mechanism
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(A, **kwargs)
        return Xhat_mean, Xhat_log_var, Xhat, z_mean, z_log_var, z, A


class VAE_Encoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features):
        super(VAE_Encoder, self).__init__()

        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.encoder = self.build_BiLSTM_encoder()

    def build_BiLSTM_encoder(self):
        # Input
        enc_input = Input(shape=(self.seq_len, self.features))
        # Add small ammount of Gaussian noise
        enc_input = GaussianNoise(0.01)(enc_input)
        # Pass input through BiLSTM layers
        bilstm = Bidirectional(LSTM(512, return_sequences=True))(enc_input)
        bilstm = Bidirectional(LSTM(256, return_sequences=True))(bilstm)
        # Transform deterministic BiLSTM output into distribution parameters z_mean and z_log_var
        z_mean = TimeDistributed(Dense(self.latent_dim, name="z_mean"))(bilstm)
        z_log_var = TimeDistributed(Dense(self.latent_dim, name="z_log_var"))(bilstm)
        # Create distribution object for reparametrisation trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        # Get epsilon for reparametrisation trick
        eps = output_dist.sample(tf.shape(z_mean))
        # Reparametrisation trick
        z = z_mean + tf.math.exp(0.5 * z_log_var) * eps
        return tf.keras.Model(enc_input, [z_mean, z_log_var, z, bilstm], name="encoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.encoder(inputs, **kwargs)


class VAE_Decoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features):
        super(VAE_Decoder, self).__init__()
        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.decoder = self.build_BiLSTM_decoder()

    def build_BiLSTM_decoder(self):
        # Latent vector input
        attention_input = Input(shape=(self.seq_len, self.latent_dim))
        # Pass input through BiLSTM layers
        bilstm = Bidirectional(LSTM(256, return_sequences=True))(attention_input)
        bilstm = Bidirectional(LSTM(512, return_sequences=True))(bilstm)
        # Transform deterministic BiLSTM output into distribution parameters Xhat_mean and Xhat_log_var
        Xhat_mean = TimeDistributed(Dense(self.features), name="Xhat_mean")(bilstm)
        Xhat_log_var = TimeDistributed(Dense(self.features), name="Xhat_log_var")(bilstm)
        # Create distribution object for reparametrisation trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        # Get epsilon for reparametrisation trick
        eps = output_dist.sample(tf.shape(Xhat_mean))
        # Reparametrisation trick
        Xhat = Xhat_mean + tf.math.exp(0.5 * Xhat_log_var) * eps
        return tf.keras.Model(attention_input, [Xhat_mean, Xhat_log_var, Xhat], name="decoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)


class MA(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features):
        super(MA, self).__init__()

        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.ma = self.build_MA()

    def build_MA(self):
        attention = MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            output_shape=self.latent_dim,
            name="A_det"
        )

        ma_input = Input(shape=(self.seq_len, self.features))
        latent_input = Input(shape=(self.seq_len, self.latent_dim))
        A = attention(ma_input, ma_input, latent_input)
        return tf.keras.Model([ma_input, latent_input], A, name="MA")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.ma(inputs, **kwargs)


# Establish callbacks
es = tf.keras.callbacks.EarlyStopping(monitor='val_log_probs_loss',
                                      mode='min',
                                      verbose=1,
                                      patience=250,
                                      restore_best_weights=True,
                                      )

annealing = VAE_kl_annealing(
    annealing_epochs=25,
    type="cyclical",
    grace_period=25,
    start=1e-8,
    end=1e-2,
    lower_initial_betas=False,
)

seq_len = 256
latent_dim = 64
features = 13

encoder = VAE_Encoder(seq_len=seq_len, latent_dim=latent_dim, features=features)
decoder = VAE_Decoder(seq_len=seq_len, latent_dim=latent_dim, features=features)
ma = MA(seq_len=seq_len, latent_dim=latent_dim, features=features)


tf_train = tf.data.Dataset.load('LOAD_PATH')
tf_val = tf.data.Dataset.load('LOAD_PATH')

tf_train = tf_train.unbatch().batch(512)
tf_val = tf_val.unbatch().batch(512)

vae = MA_VAE(encoder, decoder, ma, beta=1e-8)
optimiser = tf.keras.optimizers.Adam(amsgrad=True)
vae.compile(optimizer=optimiser)

# Fit vae model
history = vae.fit(tf_train,
                  epochs=10000,
                  callbacks=[es,
                             annealing,
                             ],
                  validation_data=tf_val,
                  verbose=1
                  )

