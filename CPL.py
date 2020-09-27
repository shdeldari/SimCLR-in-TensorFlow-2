from keras.backend import concatenate
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
import tensorflow as tf
from tcn import TCN


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out

def m_activation(activation, model):
    if activation == 'norm_relu':

        model = keras.layers.Activation('relu')(model)
        model = keras.layers.Lambda(channel_normalization, name="encoder_norm_{}".format(100))(model)
    elif activation == 'wavenet':
        model = WaveNet_activation(model)
    else:
        model = keras.layers.Activation(activation)(model)
    return model

def WaveNet_activation(x):
    tanh_out = keras.layers.Activation('tanh')(x)
    sigm_out = keras.layers.Activation('sigmoid')(x)
    return keras.layers.Concatenate(mode='mul')([tanh_out, sigm_out])

def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings
        Any Network encoder can be used here : MTCN, RNN-based, dilated-TCN, ...
        Here we used multiple dilated TCN for multi dimenstion input.
    '''
    x = keras.layers.Flatten()(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x


def mResTCN_encoder (x, code_size, nb_filters=64, kernel_size = 4, nb_stacks=1, dilations=[1, 2, 4], padding='causal',
        use_skip_connections=True, dropout_rate=0,activation='relu'):

    x = K.expand_dims(x)
    n_steps_in = x.shape[2]
    n_features = x.shape[1]
    #input_h = Input(shape=(n_steps_in, n_features), name="Input_History")

    xrep_dtcn = list()

    for i in range(0, n_features):
        # Slicing the ith channel:
        #x_h = Lambda(lambda xh: xh[:,:, i])(input_h)
        x_h =  x[:,i,:]
        #x_h = K.expand_dims(x_h)
        # TCN - rep learning
        ### this is 2-dimensional TCN I need to use 1-dimensional
        xrep  = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
                    use_skip_connections, dropout_rate, return_sequences=True, activation='relu',
                    kernel_initializer='he_normal', use_batch_norm=True)(x_h)

        xrep = keras.layers.Flatten()(xrep)
        xrep = keras.layers.Dense(n_steps_in)(xrep)
        xrep = m_activation(activation, xrep)
        xrep = keras.layers.Dense(code_size)(xrep)

        xrep_dtcn.append(xrep)

    encoded = concatenate(xrep_dtcn, axis=-1)
    return encoded

def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''
    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)
    return x


def network_prediction(context, pred_shape, l_name="pred"):
    n,m = pred_shape[0], pred_shape[1]
    ''' Define the network mapping context to multiple embeddings '''

    pred = keras.layers.Dense(n*m*2, activation='relu')(context)
    flat_encoded = keras.layers.Dropout(0.01)(pred)
    pred = keras.layers.Dense(n * m, activation='relu', name=l_name)(pred)
    return pred

class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''
    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension
        # Keras loss functions take probabilities
        cosine = K.sigmoid(dot_product)
        return cosine

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

def cosine_dist(a, b):
    # calculate distnace based on cosine similarity
    cosine = K.sum(a * b,axis=1, keepdims=True) / (K.sqrt(K.sum(a * a,axis=1, keepdims=True)) * K.sqrt(K.sum(b * b,axis=1, keepdims=True)))
    return  cosine

def infoNCEsim(history, future): #batch_size : 12
    temp = tf.cast(0.5, dtype=tf.float32)

    sim = tf.matmul(history, future, transpose_b=True) # output shape: bs*bs
    history_2 = K.sqrt(K.sum(history * history, axis=1, keepdims=True))
    future_2 = K.sqrt(K.sum(future * future, axis=1, keepdims=True))
    cc = tf.matmul(history_2, future_2, transpose_b=True) # cc(bs:bs)

    scaled_cosine_sim = tf.math.truediv(sim, cc, name="cosine") / temp # scaled by temp
    loss = K.softmax(scaled_cosine_sim, 1)
    # sim = K.exp(sim)
    # div = K.sum(sim, axis=1)

    # top = tf.linalg.diag_part(sim)
    # loss = tf.math.truediv(top, div, name="cosine")
    # loss = -1*tf.math.log(loss)

    #loss = tf.linalg.diag_part(loss)
    ##div = K.sum(sim, axis=1)
    ##loss =  tf.divide(loss, div, name="infoNCE")

    return loss

def DotLayer(a, b):
    # similarity based on dot product
    dot = K.sum(K.exp(a * b),axis=1, keepdims=True)
    return dot
# Calculates the euclidean distance used in the output layer
def euclidean_distance(x,y):
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def infoNCE_loss(y_true, y_pred):
    # Normalized Temperature based Cross Entropy Loss

    # need to implement...
    return T

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    y_true = tf.cast(y_true, dtype=tf.float32)
    square_pred = K.square((1 - y_pred))
    margin_square = K.square(K.maximum(y_pred, 0))

    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def network_cpc(h_shape, f_shape, terms, predict_terms, code_size, learning_rate, sim = "cosine_bin", temperature = 1):
    # loss : dot, cosine, pearson
    ''' Define the CPC network combining encoder and autoregressive model '''
    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    if sim == 'cosine_bin':
        loss = contrastive_loss
        metric = tf.keras.metrics.BinaryAccuracy()
    elif sim == 'cosine_ce':
        loss = tf.keras.losses.categorical_crossentropy
        metric = tf.keras.metrics.CategoricalAccuracy()
        #loss = tf.keras.losses.binary_crossentropy
        #metric = tf.keras.metrics.BinaryAccuracy()

    K.set_learning_phase(1)
    # Define encoder model
    # History frame Input
    x_input = keras.layers.Input(h_shape, name = "Input_History")
    # Encode History frame to extract predictive representation
    history_encoded = mResTCN_encoder(x_input, code_size, nb_filters=64, kernel_size = 4, nb_stacks=2,
                                     dilations=[1, 2, 4, 16], padding='same', use_skip_connections=True)
    history_encoded = K.l2_normalize(history_encoded)
    #history_encoded = network_prediction(history_encoded, (terms, code_size), l_name="history_pred")

    # Future frame Input
    y_input = keras.layers.Input((f_shape[0], f_shape[1]), name="Input_Future")
    # Encode future frame to extract representation
    future_encoded = mResTCN_encoder(y_input, code_size, nb_filters=64, kernel_size = 4, nb_stacks=1,
                                     dilations=[1, 2, 4], padding='same', use_skip_connections=True)
    future_encoded = K.l2_normalize(future_encoded)
    #f_preds = network_prediction(future_encoded, f_shape, l_name="future_pred")


    # Loss
    if sim == "cosine_ce":
        sim_output = infoNCEsim(history_encoded, future_encoded)
    elif sim =="cosine_bin":
        sim_output = cosine_dist(history_encoded, future_encoded)
    elif sim == "euclidean":
        sim_output = euclidean_distance(history_encoded, future_encoded)
    elif sim == "dot": # loss : dotproduct sigmoid
        sim_output = DotLayer(history_encoded, future_encoded)


    # Model #keras.models.
    #cpc_model = Model(inputs=[x_input, y_input], outputs=[dot_product_probs,f_preds])
    cpc_model = Model(inputs=[x_input, y_input], outputs=[sim_output])

    # Compile model
    # define two dictionaries: one that specifies the loss method for
    # each output of the network along with a second dictionary that
    # specifies the weight per loss

    #losses = [contrastive_loss,"mse"]
    #lossWeights = [0.5,0.5]
    cpc_model.compile(
        optimizer = keras.optimizers.Adam(lr=learning_rate),
        loss = loss,
        metrics = [metric]
    )
    cpc_model.summary()

    return cpc_model
