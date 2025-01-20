import keras
import numpy as np
import tensorflow as tf


class InformedConstraint(keras.constraints.Constraint):
    def __init__(self, adj):
        self.adj = adj

    def __call__(self, w):
        return w * tf.constant(self.adj, dtype=w.dtype)

    def get_config(self):
        return {"adj": self.adj.tolist()}


class InformedBiasConstraint(keras.constraints.Constraint):
    def __init__(self, adj):
        self.adj = 1 * (np.sum(adj, axis=0) > 0)

    def __call__(self, w):
        return w * tf.constant(self.adj, dtype=w.dtype)

    def get_config(self):
        return {"adj": self.adj.tolist()}


# class InformedInitializer(keras.initializers.Initializer):
#     def __init__(self, adj, initializer):
#         self.adj = adj
#         self.initializer = initializer

#     def __call__(self, shape, dtype=None):
#         return tf.constant(self.adj, dtype=dtype) * self.initializer(shape, dtype=dtype)

#     def get_config(self):
#         return {"adj": self.adj, "initializer": self.initializer}
# class InformedInitializer(keras.initializers.Initializer):
#     def __init__(self, adj, base_initializer=keras.initializers.GlorotUniform()):
#         self.adj = adj
#         self.base_initializer = base_initializer

#     def __call__(self, shape, dtype=None):
#         base_values = self.base_initializer(shape, dtype)
#         return tf.where(tf.constant(self.adj, dtype=tf.bool), base_values, tf.zeros_like(base_values))

#     def get_config(self):
#         return {
#             'adj': self.adj,
#             'base_initializer': keras.initializers.serialize(self.base_initializer)
#         }

#     @classmethod
#     def from_config(cls, config):
#         config['base_initializer'] = keras.initializers.deserialize(config['base_initializer'])
#         return cls(**config)

# class BiasInitializer(keras.initializers.Initializer):
#     def __init__(self, adj):
#         self.adj = adj

#     def __call__(self, shape, dtype=None):
#         bias = 1 * (np.sum(self.adj, axis=0) > 0)
#         return tf.constant(bias, shape=shape, dtype=dtype)

#     def get_config(self):
#         return {'adj': self.adj}

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
