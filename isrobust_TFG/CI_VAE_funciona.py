import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import tensorflow.keras.backend as K
from keras import ops, regularizers
from keras.layers import Dense, Input, Layer
from keras.models import Model
from keras.optimizers import Adam

from isrobust_TFM.layers import InformedBiasConstraint, InformedConstraint
from isrobust_TFM.utils import set_all_seeds


class Sampling(Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.seed_generator = keras.random.SeedGenerator(42)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(
            shape=(batch, dim), mean=0.0, stddev=0.1
        )  # , seed=self.seed_generator
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VAE_Loss(Layer):
    """
    Layer that adds VAE total loss to the model.
    """

    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim

    def call(self, inputs):
        inputs, outputs, z_mean, z_log_sigma = inputs
        reconstruction_loss = K.mean(K.square(inputs - outputs))
        reconstruction_loss *= self.input_dim
        kl_loss = 1 + z_log_sigma - ops.square(z_mean) - ops.exp(z_log_sigma)
        kl_loss = ops.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = ops.mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return outputs


def build_kegg_layers(circuits, pathways, act="tanh"):
    """
    Biologist priors to use using kegg information
    Args:
        circuits : type of circuits that we are gonna use.
        pathways  : type of pathways that we wanna use.
        act : Activation function. Defaults to "tanh".

    Returns:
       layers: layers of biologist prior choosen
    """

    layers = []
    if circuits is not None:  # and (circuits.shape[1] > 0):
        circuit_layer = Dense(
            circuits.shape[1],
            activation=act,
            activity_regularizer=regularizers.L2(1e-5),
            kernel_constraint=InformedConstraint(circuits),
            bias_constraint=InformedBiasConstraint(circuits),
            name="circuits",
        )
        layers.append(circuit_layer)
        #
    if pathways is not None:  # and (pathways.shape[1] > 0):
        pathway_layer = Dense(
            pathways.shape[1],
            activation=act,
            activity_regularizer=regularizers.L2(1e-5),
            kernel_constraint=InformedConstraint(pathways),
            bias_constraint=InformedBiasConstraint(pathways),
            name="pathways",
        )
        layers.append(pathway_layer)
    return layers


def build_reactome_layers(adj, act="tanh"):
    """
    Biologist priors to use using reactome information

    Args:
        circuits : type of circuits that we are gonna use.
        pathways  : type of pathways that we wanna use.
        act : Activation function. Defaults to "tanh".

    Returns:
       layers: layers of biologist prior choosen
    """
    if adj is not None:  # and (adj.shape[1] > 0):
        return [
            Dense(
                adj.shape[1],
                activation=act,
                activity_regularizer=regularizers.L2(1e-5),
                kernel_constraint=InformedConstraint(adj),
                bias_constraint=InformedBiasConstraint(adj),
                name="pathways",
            )
        ]
    else:
        return []


def build_reactome_vae(adj):
    """
    Build de VAE based in the reactome priors
    """
    layers = build_reactome_layers(adj)
    return build_vae(layers=layers, seed=42, learning_rate=1e-5)


def build_kegg_vae(circuits, pathways):
    """_summary_
    Build de VAE based in the KEEG priors
    """
    layers = build_kegg_layers(circuits, pathways)
    return build_vae(layers=layers, seed=42, learning_rate=1e-5)


def build_vae(layers, seed, learning_rate):
    """
    Build the variational autoencoder model

    Args:
        layers (_type_): _description_
        seed (_type_): _description_
        learning_rate (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        vae: model of the variational autoencoder
        encoder: model that goes to the latent dimension the data.
        decoder: model that goes from the latent dimension to the original one.
    """
    set_all_seeds(seed)

    # if not layers:
    #   raise ValueError("The layers list cannot be empty.")

    latent_dim = layers[-1].kernel_constraint.adj.shape[1] // 2
    input_dim = layers[0].kernel_constraint.adj.shape[0]

    inputs = Input(shape=(input_dim,))

    # build recursevely the hidden layers of the encoder
    for i, layer in enumerate(layers):
        if i == 0:
            inner_encoder = layer(inputs)
        else:
            inner_encoder = layer(inner_encoder)

    z_mean = Dense(latent_dim)(inner_encoder)
    z_log_sigma = Dense(latent_dim)(inner_encoder)

    z = Sampling()([z_mean, z_log_sigma])

    # Create encoder
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name="encoder")

    # Create decoder
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")

    # build recursevely the hidden layers of the decoder
    for i, layer in enumerate(layers[::-1]):
        if i == 0:
            inner_decoder = Dense(
                layer.kernel_constraint.adj.shape[1], activation="tanh"
            )(latent_inputs)
        else:
            inner_decoder = Dense(
                layer.kernel_constraint.adj.shape[1], activation="tanh"
            )(inner_decoder)

    outputs = Dense(input_dim, activation="linear")(inner_decoder)
    decoder = Model(latent_inputs, outputs, name="decoder")

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae_outputs = VAE_Loss(input_dim)([inputs, outputs, z_mean, z_log_sigma])
    vae = Model(inputs, vae_outputs, name="vae_mlp")

    vae.compile(optimizer=Adam(learning_rate=learning_rate), metrics=["mse"])

    return vae, encoder, decoder
