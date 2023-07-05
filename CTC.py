import keras
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Input, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# def add_framewise_classification_layer(model):
#     time_distributed_layer = TimeDistributed(Dense(num_classes, activation='softmax'))(model.layers[-2].output)
#     return Model(inputs=model.input, outputs=[model.output, time_distributed_layer])

def add_framewise_classification_layer(model):
    blstm_output = model.get_layer('dense').output
    time_distributed_layer = TimeDistributed(Dense(num_classes, activation='softmax'))(blstm_output)
    return Model(inputs=model.input, outputs=[model.output, time_distributed_layer])
  
num_features = 26 
num_classes = 62  
hidden_units = 100  
input_data = Input(name='the_input', shape=(None, num_features), dtype='float32')  
forward_lstm = LSTM(hidden_units, return_sequences=True, name='forward_lstm')(input_data)
backward_lstm = LSTM(hidden_units, return_sequences=True, go_backwards=True, name='backward_lstm')(input_data)
blstm = keras.layers.concatenate([forward_lstm, backward_lstm], axis=-1)
blstm = Dense(num_classes, activation='softmax', name='dense')(blstm)

labels = Input(name='the_labels', shape=[None,], dtype='int32')
input_length = Input(name='input_length', shape=[1], dtype='int32')
label_length = Input(name='label_length', shape=[1], dtype='int32')

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([blstm, labels, input_length, label_length])
model = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

model = add_framewise_classification_layer(model)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'time_distributed': 'categorical_crossentropy'}, 
              optimizer=Adam(lr=1e-4), 
              metrics={'time_distributed': 'accuracy'})

print(model.summary())
