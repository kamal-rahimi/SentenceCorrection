"""
Word based longuage model
"""
import os
import argparse
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Activation
from keras import backend as k

import numpy as np
from numpy.random import choice

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
        LSTM(70, return_sequences=False),
        Dense(vocab_size),
        Activation('softmax'),
    ])
    return model

def train_model(model, X, y, epochs=100):
    """ Trains the keras model
    Args:
        model: sequential model
        X: train dataset
        y: train labels
    Return:
        model: trained model
    """
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Nadam',
                  metrics=['accuracy'])
    model.fit(X, y, epochs, verbose=2)
    return model


def finish_sentence(model, word2id, id2word, input_seq = [], max_sentence_words = 12, num_sentences = 1):
    """ genrates a senetnce based on input words sequnce and the longuage model
    Args:
        model: trained longuage model object
        word2id: dictionary to convert from word to id
        id2word: dictionary to convert from id to word
        input_seq (text): input sequnce of words to be completed as sentence
        max_sentence_words: maximum number of words in a senetnce
        num_sentences: number of sentnces to be generated

    Return:
        sentences: a paython list of completed senetnces based on input sequnce
    """
    input_seq = strip_punctuations(input_seq)
    input_seq = input_seq.lower()
    input_seq_token_words = input_seq.split()

    vocab_size = len(word2id)
    sentences = []
    for _ in range(num_sentences):
        seq = [word2id['<BGN>']] +  [word2id[word] if word in word2id else word2id['<UNK>'] for word in input_seq_token_words]
        input_seq_size = len(seq)
        for i in range(0, max_sentence_words - input_seq_size + 1):
            x = pad_sequences([seq], maxlen=max_sentence_words-1, truncating='pre')
            #y = model.predict_classes(x, verbose=0)
            y_prob = model.predict(x, verbose=0)
            p = y_prob.reshape(-1,1)[:,0]
            a = np.array(range(0, vocab_size)).reshape(-1,1)[:,0]
            p[word2id['<UNK>']] = 0
            p = p / np.sum(p)
            predict_word = choice(a, p=p)
            seq.append(predict_word)
            if predict_word == word2id['<EOS>']:
                break

        new_words = [id2word[id] for id in seq]
        senetnce = ' '.join(new_words)
        sentences.append(senetnce)

    return sentences

def analyze_sequence(model, words_id_order, max_sentence_words):
    """ Computes the liklihood of the input sequnce of words using the longuage model
        Args:
            model: trained longuage model object
            words_ids: inout sequnce of word ids
            max_sentence_words: maximum number of words in a senetnce
        Returns:
            p_sentence: the liklihood of inout sequnce in the longuage model
            p_words: a python array of the liklihood of each word given its predecessor words
    """
    p_words = [1]
    p_sentence = 1
    for word_index in range(1, len(words_id_order)-1):
        seq = words_id_order[:word_index]
        x = pad_sequences([seq], maxlen=max_sentence_words-1, truncating='pre')
        y = words_id_order[word_index]
        predict_prob = model.predict(x, verbose=0)

        predict_prob = np.array(predict_prob).reshape(-1,)
        prob = predict_prob[y]
        p_words.append(prob)
        p_sentence = p_sentence*prob

    return p_sentence, p_words

def process_sentence(model, word2id, id2word, window_size, input_sentence, max_sentence_words = 12):
    """ analyzes the inout sentnces and reorders the word to form a senetnces which
    has the highest liklihood in the longuage model.

    Args:
        model: trained longuage model object
        word2id: dictionary to convert from word to id
        id2word: dictionary to convert from id to word
        window_size:  word reordering search window size
        input_sentnce (text): input sentnce
        max_sentence_words: maximum number of words in a senetnce
    Returns:
        most_likely_sentence: the word reordred senetnce that has highes liklihood in the longuage model
        most_likely_word_order_prob: liklihood of the reordred sentence
    """
    input_sentence = strip_punctuations(input_sentence)
    input_sentence = input_sentence.lower()
    sentence_words = input_sentence.split()
    sentence_words_id = [word2id[word] if word in word2id else word2id['<UNK>'] for word in sentence_words]

    full_sentence_words_id = [word2id['<BGN>']] + sentence_words_id + [word2id['<EOS>']]
    inout_word_order_prob, _ = analyze_sequence(model, full_sentence_words_id, max_sentence_words)

    sentence_words_id_permutations = []
    num_iterations = max(1, len(sentence_words_id) - window_size + 1)
    for i in range(0, num_iterations):
        words_id_permutations = [ sentence_words_id[0 : i] + list(l) for l in permutations(sentence_words_id[i : window_size + i]) ]
        num_permutations = len(words_id_permutations)
        sentence_size = len(words_id_permutations[0])

        words_id_permutations_prob = []
        for words_id_order_index in range(0, num_permutations):
            words_id_order = list(words_id_permutations[words_id_order_index])
            words_id_order = [word2id['<BGN>']] + words_id_order
            if i == num_iterations-1:
                words_id_order = words_id_order + [word2id['<EOS>']]

            p_sentence, p_words = analyze_sequence(model, words_id_order, max_sentence_words)

            words_id_permutations_prob.append(p_sentence)

        most_likely_word_order_index = np.argmax(words_id_permutations_prob)
        most_likely_word_order_prob = words_id_permutations_prob[most_likely_word_order_index]
        most_likely_words_id_order = words_id_permutations[most_likely_word_order_index]

        sentence_words_id = most_likely_words_id_order + sentence_words_id[window_size + i : ]

    most_likely_words_order = [id2word[id] for id in sentence_words_id]
    most_likely_sentence = ' '.join(most_likely_words_order)

    return inout_word_order_prob, most_likely_sentence, most_likely_word_order_prob


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
    #ap.add_argument("-m", "--model", type=str, default="knn", help="type of python machine learning model to use")
    ap.add_argument("-p", "--path",  type=str, default="./data/English.txt", help="Specify the data file path")
    ap.add_argument("-v", "--vsize", type=int, default=40000,                help="Specify the vocabulary size")
    ap.add_argument("-s", "--ssize", type=int, default=12,                   help="Specify maximum senetnce size (number of words)")
    ap.add_argument("-g", "--gpu",                                           help="Specify to use GPU for training the model", action='store_true')
    args = vars(ap.parse_args())
    data_path = args["path"]
    vocab_size = args["vsize"]
    max_sentence_words = args["ssize"]
    use_gpu = args["gpu"]

    if use_gpu:
        config_gpu()

    if os.path.isfile("models/English/word_lm-40.h5") == False:

        data = load_text(data_path)
        cleaned_data = clean_text(data)
        word2id, id2word = create_vocabulary(cleaned_data, vocab_size)
        token_senetnces = tokenize_sentence(cleaned_data)
        print(len(token_senetnces))
        X, y = prepare_data(word2id, token_senetnces[:16000], max_sentence_words)

        model = create_model(vocab_size)
        model.summary()

        model = train_model(model, X, y)

        model.save('models/English/word_lm-new.h5')

        with open('meta_data.pickle','wb') as f:
            pickle.dump([word2id, id2word], f)
    else:
        model = load_model('models/English/word_lm-40.h5')
        with open('meta_data.pickle','rb') as f:
            word2id, id2word = pickle.load(f)

    #sentences = finish_sentence(model, word2id, id2word, 'to be or not to be  ' , 12, 10)
    #for sentence in sentences:
    #    print(sentence)

    # This is a test project
    # cat is there in a the room
    # it will rain tomorrow
    # the boy fall in love
    input_sentence = 'day is bright and night is dark'
    input_sentences_liklihood, corrected_sentence, corrected_sentence_liklihood = process_sentence(model, word2id, id2word, 5, input_sentence)
    print('\nInput: ')
    print(input_sentence)
    print('Liklihood:')
    print(input_sentences_liklihood)
    print('\nCorrected: ')
    print(corrected_sentence)
    print('Liklihood:')
    print(corrected_sentence_liklihood)
    print('\n')

if __name__ == '__main__':
    main()
