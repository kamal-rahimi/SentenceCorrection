""""
Completes an input sequnce of words  as a sentnces (finish sentence)

"""

import os
import argparse
import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as k

import numpy as np
from numpy.random import choice

from nlp_tools import strip_punctuations



def finish_sentence(model_name, input_seq = [], max_sentence_words = 12, num_sentences = 1):
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
    model_path = './models/' + model_name + '_model.h5'
    meta_data_path = './models/' + model_name + '_metadata.pickle'
    if (os.path.isfile(model_path) == True) and (os.path.isfile(model_path) == True):
        model = load_model(model_path)
        with open(meta_data_path,'rb') as f:
            word2id, id2word = pickle.load(f)
    else:
        print('No model with name \"%s\" is trained yet' % model_name)
        return

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

    k.clear_session()
    return sentences


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name",         type=str, default="English", help="specify the longuage model name")
    ap.add_argument("-s", "--sentence",     type=str, default="This is", help="specify the longuage model name")
    ap.add_argument("-l", "--length",       type=int, default=12,        help="Specify the maximum senetnce length (number of words)")
    ap.add_argument("-c", "--count",       type=int, default=5,        help="Specify the number of senetnces to be generated")
    ap.add_argument("-g", "--gpu",                                       help="Specify to use GPU for training the model", action='store_true')
    args = vars(ap.parse_args())
    model_name = args["name"]
    input_seq  = args['sentence']
    max_sentence_words = args["length"]
    num_sentences = args['count']
    use_gpu    = args["gpu"]

    if use_gpu:
        config_gpu()

    sentences = finish_sentence(model_name, input_seq, max_sentence_words, num_sentences)
    for sentence in sentences:
        print(sentence)


if __name__ == '__main__':
    main()
