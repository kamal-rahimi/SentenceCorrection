"""
Estimates the likihood of an input senetnce and finds an order of words
that is most likely in the longuage model
"""

import os
import argparse
import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as k

import numpy as np

from itertools import permutations

from nlp_tools import strip_punctuations

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

def process_sentence(model_name, input_sentence, window_size, max_sentence_words = 12):
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
    model_path = './models/' + model_name + '_model.h5'
    meta_data_path = './models/' + model_name + '_metadata.pickle'
    if (os.path.isfile(model_path) == True) and (os.path.isfile(model_path) == True):
        model = load_model(model_path)
        with open(meta_data_path,'rb') as f:
            word2id, id2word = pickle.load(f)
    else:
        print('No model with name \"%s\" is trained yet' % model_name)
        return


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

    k.clear_session()

    most_likely_words_order = [id2word[id] for id in sentence_words_id]
    most_likely_sentence = ' '.join(most_likely_words_order)
    return inout_word_order_prob, most_likely_sentence, most_likely_word_order_prob


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name",     type=str, default="English", help="specify the longuage model name")
    ap.add_argument("-s", "--sentence", type=str, default="This is", help="specify the longuage model name")
    ap.add_argument("-w", "--window",   type=int, default=5,         help="specify the window size to reorder words")
    ap.add_argument("-g", "--gpu",                                   help="Specify to use GPU for training the model", action='store_true')

    args = vars(ap.parse_args())
    model_name     = args["name"]
    input_sentence = args['sentence']
    window_size    = args['window']
    use_gpu        = args["gpu"]

    if use_gpu:
        config_gpu()

    input_sentences_liklihood, corrected_sentence, corrected_sentence_liklihood = process_sentence(model_name, input_sentence, window_size)
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
