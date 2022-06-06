import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape
import warnings


class GatherIndices(Layer):
    """
    Gathers slices from a data tensor, based on an indices tensors (``tf.gather`` in Layer form).

    Args:
        axis (int or Tensor): the data axis to gather from.
        batch_dims (int): the number of batch dimensions in the data and indices.
    """

    def __init__(self, axis=None, batch_dims=0, **kwargs):
        super().__init__(**kwargs)
        self._axis = axis
        self._batch_dims = batch_dims

    def get_config(self):
        config = super().get_config()
        config.update(axis=self._axis, batch_dims=self._batch_dims)
        return config

    def compute_output_shape(self, input_shapes):
        data_shape, indices_shape = input_shapes
        axis = self._batch_dims if self._axis is None else self._axis
        # per https://www.tensorflow.org/api_docs/python/tf/gather
        return (
            data_shape[:axis]
            + indices_shape[self._batch_dims :]
            + data_shape[axis + 1 :]
        )

    def call(self, inputs):
        """
        Args:
            inputs (list): a pair of tensors, corresponding to the ``params`` and ``indices``
                parameters to ``tf.gather``.
        """
        data, indices = inputs
        return tf.gather(data, indices, axis=self._axis, batch_dims=self._batch_dims)


class GraphConvolution(Layer):

    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on https://github.com/tkipf/keras-gcn.

    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn

    Notes:
      - The batch axis represents independent graphs to be convolved with this GCN kernel (for
        instance, for full-batch node prediction on a single graph, its dimension should be 1).

      - If the adjacency matrix is dense, both it and the features should have a batch axis, with
        equal batch dimension.

      - If the adjacency matrix is sparse, it should not have a batch axis, and the batch
        dimension of the features must be 1.

      - There are two inputs required, the node features,
        and the normalized graph Laplacian matrix

      - This class assumes that the normalized Laplacian matrix is passed as
        input to the Keras methods.

    .. seealso:: :class:`.GCN` combines several of these layers.

    Args:
        units (int): dimensionality of output feature vectors
        activation (str or func): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
        final_layer (bool): Deprecated, use ``tf.gather`` or :class:`.GatherIndices`
        kernel_initializer (str or func, optional): The initialiser to use for the weights.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights.
        kernel_constraint (str or func, optional): The constraint to use for the weights.
        bias_initializer (str or func, optional): The initialiser to use for the bias.
        bias_regularizer (str or func, optional): The regulariser to use for the bias.
        bias_constraint (str or func, optional): The constraint to use for the bias.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        final_layer=None,
        input_dim=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        if final_layer is not None:
            raise ValueError(
                "'final_layer' is not longer supported, use 'tf.gather' or 'GatherIndices' separately"
            )

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by Keras model serialization.

        Returns:
            A dictionary that contains the config of the layer
        """

        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:

        Args:
            input_shapes (tuple of int)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        feature_shape, *As_shapes = input_shapes

        batch_dim = feature_shape[0]
        out_dim = feature_shape[1]

        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)

        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        """
        Applies the layer.

        Args:
            inputs (list): a list of 3 input tensors that includes
                node features (size 1 x N x F),
                graph adjacency matrix (size N x N),
                where N is the number of nodes in the graph, and
                F is the dimensionality of node features.

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        features, *As = inputs

        # Calculate the layer operation of GCN
        A = As[0]
        if K.is_sparse(A):
            # FIXME(#1222): batch_dot doesn't support sparse tensors, so we special case them to
            # only work with a single batch element (and the adjacency matrix without a batch
            # dimension)
            if features.shape[0] != 1:
                raise ValueError(
                    f"features: expected batch dimension = 1 when using sparse adjacency matrix in GraphConvolution, found features batch dimension {features.shape[0]}"
                )
            if len(A.shape) != 2:
                raise ValueError(
                    f"adjacency: expected a single adjacency matrix when using sparse adjacency matrix in GraphConvolution (tensor of rank 2), found adjacency tensor of rank {len(A.shape)}"
                )

            features_sq = K.squeeze(features, axis=0)
            h_graph = K.dot(A, features_sq)
            h_graph = K.expand_dims(h_graph, axis=0)
        else:
            h_graph = K.batch_dot(A, features)
        output = K.dot(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)

        return output

class SqueezedSparseConversion(Layer):
    """
    Converts Keras tensors containing indices and values to a tensorflow sparse
    tensor. The input tensors are expected to have a batch dimension of 1 which
    will be removed before conversion to a matrix.

    This only works with a tensorflow Keras backend.

    Example:
        ```
        A_indices = Input(batch_shape=(1, None, 2), dtype="int64")
        A_values = Input(batch_shape=(1, None))
        Ainput = TFSparseConversion(shape=(N, N))([A_indices, A_values])
        ```

    Args:
        shape (list of int): The shape of the sparse matrix to create
        dtype (str or tf.dtypes.DType): Data type for the created sparse matrix
    """

    def __init__(self, shape, axis=0, dtype=None):
        super().__init__(dtype=dtype)

        self.trainable = False
        self.supports_masking = True
        self.matrix_shape = shape
        # self.dtype = dtype
        self.axis = axis

        # Check backend
        if K.backend() != "tensorflow":
            raise RuntimeError(
                "SqueezedSparseConversion only supports the TensorFlow backend"
            )

    def get_config(self):
        config = {"shape": self.matrix_shape, "dtype": self.dtype}
        return config

    def compute_output_shape(self, input_shapes):
        return tuple(self.matrix_shape)

    def call(self, inputs):
        """
        Creates a TensorFlow `SparseTensor` from the inputs

        Args:
            inputs (list): Two input tensors contining
                matrix indices (size 1 x E x 2) of type int64, and
                matrix values (size (size 1 x E),
                where E is the number of non-zero entries in the matrix.

        Returns:
            TensorFlow SparseTensor that represents the converted sparse matrix.
        """
        # Here we squeeze the specified axis
        if self.axis is not None:
            indices = K.squeeze(inputs[0], self.axis)
            values = K.squeeze(inputs[1], self.axis)
        else:
            indices = inputs[0]
            values = inputs[1]

        if self.dtype is not None:
            values = K.cast(values, self.dtype)

        # Import tensorflow here so that the backend check will work without
        # tensorflow installed.
        import tensorflow as tf

        # Build sparse tensor for the matrix
        output = tf.SparseTensor(
            indices=indices, values=values, dense_shape=self.matrix_shape
        )
        return output
