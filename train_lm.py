"""
Builds a longuage model to estimate the liklihood of any sentence

"""

import os
import argparse
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as k

import numpy as np
from numpy.random import choice

from sklearn.model_selection import train_test_split

from itertools import permutations

from nlp_tools import load_text, clean_text, create_vocabulary, strip_punctuations, tokenize_sentence

def prepare_data(word2id, token_sentences, max_sentence_words = 12 ):
    """ Prepares dataset for the model
    Args:
        word2id: dictionary to convert from words to id
        token_sentences: a python array of sentences
        max_sentence_words: maximum number of words in a senetnce
    Return:
        X: Python array of words sequnces
        y: Python array of next word in each sequnce
    """
    data = []
    for sentence in token_sentences:
        sentence = strip_punctuations(sentence)
        sentence = sentence.lower()
        sentence_token_words = sentence.split()
        sentence_token_words = ['<BGN>'] + sentence_token_words + ['<EOS>']
        sentence_size = min(len(sentence_token_words), max_sentence_words)
        for word_index in range(2, sentence_size+1):
            token_words = sentence_token_words[: word_index]
            num_pads = max_sentence_words - word_index
            token_words_padded =  ['<PAD>']*num_pads + token_words
            token_words_id_padded = [word2id[word] if word in word2id else word2id['<UNK>'] for word in token_words_padded]
            data.append(token_words_id_padded)

    data = np.array(data)
    X = data[:, :-1]
    y = data[:,-1]
    return X, y

def create_model(vocab_size, embedding_dim=40):
    """ Creates longuage model using keras
    Args:
        vocabulary vocab_size
        embedding dimmestion
    Returns:
        model
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        LSTM(70, dropout=0.00, return_sequences=False),
        Dense(vocab_size),
        Activation('softmax'),
    ])
    return model

def train_model(model, X_train, X_valid, y_train, y_valid, epochs=100):
    """ Trains the keras model
    Args:
        model: sequential model
        X: train dataset
        y: train labels
    Return:
        model: trained model
    """
    #callbacks = [EarlyStopping(monitor='val_acc', patience=5)]
    callbacks = [ModelCheckpoint('models/model.chkpt'), save_best_only=True, save_weights_only=False)]
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Nadam',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks, verbose=2, validation_data=(X_valid,y_valid))
    return model


def config_gpu():
    """ Configure tensorflow to run on GPU

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    k.tensorflow_backend.set_session(tf.Session(config=config))

def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name",   type=str, default="English",            help="specify the longuage model name")
    ap.add_argument("-p", "--path",   type=str, default="./data/English.txt", help="Specify the train data path")
    ap.add_argument("-c", "--count",  type=int, default=16000,                help="Specify the maximum number of senetnces to train model")
    ap.add_argument("-v", "--vsize",  type=int, default=40000,                help="Specify the vocabulary size")
    ap.add_argument("-l", "--length", type=int, default=15,                   help="Specify the maximum senetnce length (number of words)")
    ap.add_argument("-e", "--epochs", type=int, default=100,                  help="Specify the number of epoch to train the model")
    ap.add_argument("-g", "--gpu",                                            help="Specify to use GPU for training the model", action='store_true')
    args = vars(ap.parse_args())
    model_name         = args["name"]
    data_path          = args["path"]
    num_sentences      = args["count"]
    vocab_size         = args["vsize"]
    max_sentence_words = args["length"]
    num_epochs         = args["epochs"]
    use_gpu            = args["gpu"]

    if use_gpu:
        config_gpu()

    data = load_text(data_path)
    cleaned_data = clean_text(data)
    word2id, id2word = create_vocabulary(cleaned_data, vocab_size)
    token_senetnces = tokenize_sentence(cleaned_data)
    token_senetnces = token_senetnces[:num_sentences]
    print("Training longuage model %s using %d sentences" % (model_name, len(token_senetnces)))
    X, y = prepare_data(word2id, token_senetnces, max_sentence_words)

    model = create_model(vocab_size)
    model.summary()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, random_state=42)
    model = train_model(model, X_train, X_valid, y_train, y_valid, num_epochs)

    model_path = './models/' + model_name + '_model.h5'
    model.save(model_path)
    meta_data_path = './models/' + model_name + '_metadata.pickle'
    with open(meta_data_path,'wb') as f:
        pickle.dump([word2id, id2word], f)

if __name__ == '__main__':
    main()
