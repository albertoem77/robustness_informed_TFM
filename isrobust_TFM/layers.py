import numpy as np
import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable()
class Informed(keras.layers.Dense):
    def __init__(self, adj, **kwargs):
        """Informed layer using the Keras API. An Informed layer is defined by an adjacency
        matrix that indicates which inputs from the previous layer are going to be set to 0.
        This defines the layer's Kernel.

        Args:
            adj (n_inputs, n_ouputs): Adjacency matrix to inform the layer kernel.
        """

        self.adj = adj
        if adj is not None:
            units = adj.shape[1]
        super(Informed, self).__init__(units=units, **kwargs)

    def build(self, input_shape):
        """Build the layer once its parameters are known.

        Args:
            input_shape (_type_): Layer  dimensions.
        """

        connection_vector = 1 * (np.sum(self.adj, axis=0) > 0)

        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=InformedInitializer(self.adj, self.kernel_initializer),
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.kernel_adj = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.Constant(self.adj),
            name="kernel_adj",
            trainable=False,
        )

        self.bias = self.add_weight(
            shape=(self.units,),
            initializer=InformedInitializer(connection_vector, self.bias_initializer),
            name="bias",
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

        self.bias_adj = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(connection_vector),
            name="bias_adj",
            trainable=False,
        )

        self.built = True

    def call(self, inputs):
        """Forward pass."""

        output = tf.keras.backend.dot(inputs, self.kernel * self.kernel_adj)
        output = tf.keras.backend.bias_add(output, self.bias_adj * self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def count_params(self):
        """Redefine the number of parameters by taking into account the
        induced sparsity.
        """

        n_bias_weights = self.bias_adj.numpy().sum()
        n_kernel_weights = self.kernel_adj.numpy().sum()

        return n_bias_weights + n_kernel_weights

    def get_config(self):
        """We need this to make it serializable."""

        config = {"adj": self.adj.tolist()}
        base_config = super().get_config()
        base_config.pop("units", None)
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        """We need this to make it serializable."""

        adj_as_list = config["adj"]
        config["adj"] = np.array(adj_as_list)
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class InformedInitializer(keras.initializers.Initializer):
    def __init__(self, adj, initializer):
        """Propagate the sparsity induced by the adjacency matrix `adj` to
        the provided `initializer`.

        Args:
            adj (_type_): Adjacency matrix.
            initializer (_type_): Keras initalizer.
        """

        self.adj = adj
        self.initializer = initializer

    def __call__(self, shape, dtype=None):
        """Sparsity induction."""

        return tf.constant(self.adj, dtype=dtype) * self.initializer(shape, dtype=dtype)

    def get_config(self):
        """We need this to make it serializable."""

        return {"adj": self.adj, "initializer": self.initializer}
