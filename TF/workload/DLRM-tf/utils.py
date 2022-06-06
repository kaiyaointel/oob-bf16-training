import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding, Layer

def MLP(units_list, use_bias=True, activation='relu', out_activation=None):
    
    mlp = Sequential()
    
    for units in units_list[:-1]:
        mlp.add(Dense(units, 
                        activation=activation, 
                        use_bias=use_bias))
    
    mlp.add(Dense(units_list[-1], 
                activation=out_activation, 
                use_bias=use_bias))
    
    return mlp


class LatentFactor(Embedding):
    
    def __init__(self, num_instances, dim, zero_init=False, name=None):
        
        if zero_init:
            initializer = 'zeros'
        else:
            initializer = 'uniform'
        super(LatentFactor, self).__init__(input_dim=num_instances, 
                                           output_dim=dim, 
                                           embeddings_initializer=initializer,
                                           name=name)
    
    def censor(self, censor_id):
        
        unique_censor_id, _ = tf.unique(censor_id)
        embedding_gather = tf.gather(self.variables[0], indices=unique_censor_id)
        norm = tf.norm(embedding_gather, axis=1, keepdims=True)
        return self.variables[0].scatter_nd_update(indices=tf.expand_dims(unique_censor_id, 1), 
                                                   updates=embedding_gather / tf.math.maximum(norm, 0.1))


class SecondOrderFeatureInteraction(Layer):
    
    def __init__(self, self_interaction=False):
        
        self._self_interaction = self_interaction
        
        super(SecondOrderFeatureInteraction, self).__init__()
    
    def call(self, inputs):
        
        '''
        inputs: list of features with shape [batch_size, feature_dim]
        '''
        
        batch_size = tf.shape(inputs[0])[0]
        
        concat_features = tf.stack(inputs, axis=1)
        dot_products = tf.linalg.LinearOperatorLowerTriangular(tf.matmul(concat_features, concat_features, transpose_b=True)).to_dense()

        ones = tf.ones_like(dot_products)
        mask = tf.linalg.band_part(ones, 0, -1)
        
        if not self._self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)
            out_dim = int(len(inputs) * (len(inputs)-1) / 2)
        else:
            out_dim = int(len(inputs) * (len(inputs)+1) / 2)
        
        flat_interactions = tf.reshape(tf.boolean_mask(dot_products, mask), (batch_size, out_dim))
            
        return flat_interactions