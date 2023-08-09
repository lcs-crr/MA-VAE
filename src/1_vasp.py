# Lucas Correia
# Mercedes-Benz AG, Stuttgart, Germany

"""
This is the script used to train the VASP model.
10.1016/j.engappai.2021.104354
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


class VASP(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(VASP, self).__init__()

        # Model
        self.encoder = encoder
        self.decoder = decoder

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @tf.function
    def loss_fn(self, X, Xhat, z_mean, z_log_var):
        # Calculate reconstruction error
        rec_loss = (X-Xhat)**2
        rec_loss = tf.reduce_mean(tf.reduce_sum(rec_loss, axis=(1, 2)))
        # Calculate KL Divergence between latent distribution and Gaussian distribution
        kl_loss = tfp.distributions.kl_divergence(
            tfp.distributions.Normal(loc=0., scale=1.),
            tfp.distributions.Normal(loc=z_mean, scale=tf.sqrt(tf.math.exp(z_log_var))))
        # Reduce KL divergence to single value per batch
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        return rec_loss, kl_loss

    @tf.function
    def train_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        with tf.GradientTape() as tape:
            # Forward pass through encoder
            z_mean, z_log_var, z = self.encoder(X, training=True)
            # Forward pass through decoder
            Xhat = self.decoder(z, training=True)
            # Calculate losses from parameters
            rec_loss, kl_loss = self.loss_fn(
                X,
                Xhat,
                z_mean,
                z_log_var
            )
            # Calculate total loss from different losses
            total_loss = rec_loss + 0.5 * kl_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        # Forward pass through encoder
        z_mean, z_log_var, z = self.encoder(X, training=False)
        # Forward pass through decoder
        Xhat = self.decoder(z, training=False)
        # Calculate losses from parameters
        rec_loss, kl_loss = self.loss_fn(
            X,
            Xhat,
            z_mean,
            z_log_var
        )
        total_loss = rec_loss + 0.5 * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def call(self, X, **kwargs):
        # Encoder is fed with input window
        z_mean, z_log_var, z = self.encoder(X, training=False)
        # Decoder is fed with the sampled latent matrix from encoder
        Xhat = self.decoder(z, training=False)
        return Xhat, z_mean, z_log_var, z


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
        # FC layer
        fc = tfkl.TimeDistributed((tfkl.Dense(80)))(enc_input)
        # LSTM layer
        lstm = tfkl.LSTM(65, return_sequences=False, dropout=0.2, recurrent_dropout=0.1)(fc)
        # FC Layer
        fc = tfkl.Dense(13)(lstm)
        # Transform deterministic BiLSTM output into distribution parameters z_mean and z_log_var
        z_mean = tfkl.Dense(self.latent_dim, name="z_mean")(fc)
        z_log_var = tfkl.Dense(self.latent_dim, name="z_log_var")(fc)
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
        latent_input = tfkl.Input(shape=(self.latent_dim,))
        # FC Layer
        fc = (tfkl.Dense(13))(latent_input)
        # FC Layer
        fc = (tfkl.Dense(65))(fc)
        # Repeat vector as many times as the window length
        rep = tfkl.RepeatVector(self.seq_len)(fc)
        # LSTM layer
        lstm = tfkl.LSTM(65, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(rep)
        #  FC layer
        fc = tfkl.TimeDistributed((tfkl.Dense(80)))(lstm)
        # FC Layer
        Xhat = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat_mean")(fc)
        return tf.keras.Model(latent_input, Xhat, name="decoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)


# Establish callbacks
es = tf.keras.callbacks.EarlyStopping(monitor='val_rec_loss',
                                      mode='min',
                                      verbose=1,
                                      patience=250,
                                      restore_best_weights=True,
                                      )

tf_train = tf.data.Dataset.load('LOAD_PATH')
tf_val = tf.data.Dataset.load('LOAD_PATH')

tf_train = tf_train.unbatch().batch(512)
tf_val = tf_val.unbatch().batch(512)

encoder = VAE_Encoder(seq_len=256, latent_dim=8, features=13)
decoder = VAE_Decoder(seq_len=256, latent_dim=8, features=13)
model = VASP(encoder, decoder)
optimiser = tf.keras.optimizers.Adam(amsgrad=True)
model.compile(optimizer=optimiser)

# Fit vae model
history = model.fit(tf_train,
                    epochs=10000,
                    callbacks=[es,
                               ],
                    validation_data=tf_val,
                    verbose=1
                    )
