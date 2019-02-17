"""
Word based longuage model
"""

import argparse
from nltk import tokenize
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Activation
from collections import Counter
import string
import numpy as np

from keras import backend as k

import os

from numpy.random import choice

from keras.preprocessing.sequence import pad_sequences

from itertools import permutations


def config_gpu():
    """Configure tensorflow to run on GPU

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    k.tensorflow_backend.set_session(tf.Session(config=config))


def load_text(file_path):
    """ Load text filename
    Args:
        data file path (text file)
    Returns:
        raw text (string)
    """
    with open(file_path, 'r') as f:
        text = f.read()

    return text

def clean_text(raw_text):
    """
    Clean text
    Args:
        raw texts
    Returns:
        Cleaned text
    """
    token_words = raw_text.split()
    cleaned_text = ' '.join(token_words)
    return cleaned_text

def create_vocabulary(text, max_vocab_size = 2000):
    """ Create Vocabulary dictionary
    Args:
        text(str)
        max_vocab_size: maximum number of words in vocabulary
    Returns:
        word2id(dict): word to id mapping
        id2word(dict): id to word mapping
    """
    words = text.split()
    freq = Counter(words)
    word2id = {'<PAD>' : 0, '<BGN>' : 1, '<UNK>' : 2, '<EOS>' : 3}
    id2word = {0 : '<PAD>', 1 : '<BGN>', 2 : '<UNK>', 3 : '<EOS>'}

    for word, _ in freq.most_common():
        id = len(word2id)
        word2id[word] = id
        id2word[id] = word
        if id == max_vocab_size - 1 :
            break

    return word2id, id2word


def tokenize_sentence(text):
    """ Tokenize senetences
    Args:
        text (str)
    Returns:
        tokenized sentences (python list of sentences)
    """
    token_sentences = tokenize.sent_tokenize(text)

    return token_sentences


def strip_punctuation(text):
    """ Remove punctuations from text
    Args:
        text(str)
    Returns
        clean_text(str)
    """
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator)
    return clean_text


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
        sentence = strip_punctuation(sentence)
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
    """ Create keras model
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


def generate_text(model, word2id, id2word, vocab_size, input_seq = [], max_sentence_words = 12, num_sentences = 1):
    """ generates new senetnces based on longuage models
    Args:
        model: trained model object
        word2id: dictionary to convert from word to id
        id2word: dictionary to convert from id to word
        vocab_size: vocabulary size
        input_seq: input sequnce of words to be completed as sentence
        max_sentence_words: maximum number of words in a senetnce
        num_sentences: number of sentnces to be generated

    Return:
        sentnces: a paython list of completed senetnces based on input sequnce
    """
    input_seq = strip_punctuation(input_seq)
    input_seq = input_seq.lower()
    input_seq_token_words = input_seq.split()

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

def analyze_senetnce(model, word2id, id2word, vocab_size, input_sentence, max_sentence_words = 12):
    input_sentence = strip_punctuation(input_sentence)
    input_sentence = input_sentence.lower()
    sentence_words = input_sentence.split()
    print(sentence_words)
    sentence_words_id = [word2id[word] if word in word2id else word2id['<UNK>'] for word in sentence_words]

    sentence_words_id_permutations = []
    window = 5
    num_iterations = max(1, len(sentence_words_id) - window + 1)
    for i in range(0, num_iterations):
        words_id_permutations = [ sentence_words_id[0:i] + list(l) for l in permutations(sentence_words_id[i:window+i]) ]
        num_permutations = len(words_id_permutations)
        sentence_size = len(words_id_permutations[0])
        words_id_permutations_prob = []
        for words_id_order_index in range(0, num_permutations):
            words_id_order = list(words_id_permutations[words_id_order_index])
            words_id_order = [word2id['<BGN>']] + words_id_order
            if words_id_order_index == num_permutations-1:
                words_id_order += [word2id['<EOS>']]
            P_Word = [1]
            p_sentence = 1
            for word_index in range(1, len(words_id_order)-1):
                seq = words_id_order[:word_index]
                x = pad_sequences([seq], maxlen=max_sentence_words-1, truncating='pre')
                y = words_id_order[word_index]
                predict_prob = model.predict(x, verbose=0)

                predict_prob = np.array(predict_prob).reshape(-1,)
                prob = predict_prob[y]
                P_Word.append(prob)
                p_sentence = p_sentence*prob

            words_id_permutations_prob.append(p_sentence)

        most_likely_word_order_index = np.argmax(words_id_permutations_prob)
        most_likely_word_order_prob = words_id_permutations_prob[most_likely_word_order_index]
        most_likely_words_id_order = words_id_permutations[most_likely_word_order_index]

        sentence_words_id = most_likely_words_id_order + sentence_words_id[window+i:]

    most_likely_words_order = [id2word[id] for id in sentence_words_id]
    most_likely_sentence = ' '.join(most_likely_words_order)
    print(most_likely_sentence)
    print(most_likely_word_order_prob)
    print('\n\n')


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    #ap.add_argument("-m", "--model", type=str, default="knn", help="type of python machine learning model to use")
    ap.add_argument("-p", "--path",  type=str, default="./data/English.txt", help="Specify the data path")
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

    data = load_text(data_path)
    cleaned_data = clean_text(data)
    word2id, id2word = create_vocabulary(cleaned_data, vocab_size)
    token_senetnces = tokenize_sentence(cleaned_data)
    print(len(token_senetnces))

    X, y = prepare_data(word2id, token_senetnces[:16000], max_sentence_words)

    if os.path.isfile("word_lm-40.h5") == False:
        model = create_model(vocab_size)
        model.summary()
        model = train_model(model, X, y)
        model.save('word_lm-new.h5')
        evaluate_model(model)
    else:
        model = load_model('word_lm-40.h5')

    sentences = generate_text(model, word2id, id2word, vocab_size, 'life is about ' , 12, 10)
    for sentence in sentences:
        print(sentence)
    # This is a test project
    # cat is there in a the room
    # school will we to go
    # it will rain tomorrow
    # solution proposed a they new
    # are many there people agree do who not
    #analyze_senetnce(model, word2id, id2word, vocab_size, input_sentence='are many there people agree do who not')



if __name__ == '__main__':
    main()
