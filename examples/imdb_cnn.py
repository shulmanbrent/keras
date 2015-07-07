from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.datasets import imdb

'''
    Train a LSTM on the IMDB sentiment classification task.

    The dataset is actually too small for LSTM to be of any advantage 
    compared to simpler, much faster methods such as TF-IDF+LogReg.

    Notes: 

    - RNNs are tricky. Choice of batch size is important, 
    choice of loss and optimizer is critical, etc. 
    Some configurations won't converge.

    - LSTM loss decrease patterns during training can be quite different 
    from what you see with CNNs/MLPs/etc. 

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

max_features = 20000
maxlen = 100 # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

X_train = X_train.astype("int32")
X_test = X_test.astype("int32")

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)



print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Build model...')


nb_feature_maps = 32
embedding_size = 64

#ngram_filters = [3, 5, 7, 9]
ngram_filters = [2, 4, 6, 8]
conv_filters = []

for n_gram in ngram_filters:
    sequential = Sequential()
    conv_filters.append(sequential)

    sequential.add(Embedding(max_features, embedding_size))
    sequential.add(Reshape(1, maxlen, embedding_size))
    sequential.add(Convolution2D(nb_feature_maps, 1, n_gram, embedding_size))
    sequential.add(Activation("relu"))
    sequential.add(MaxPooling2D(poolsize=(maxlen - n_gram + 1, 1)))
    sequential.add(Flatten())

model = Sequential()
model.add(Merge(conv_filters, mode='concat'))
model.add(Dropout(0.5))
model.add(Dense(nb_feature_maps * len(conv_filters), 1))
model.add(Activation("sigmoid"))


# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)
