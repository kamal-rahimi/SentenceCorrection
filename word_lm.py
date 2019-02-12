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

import os

from numpy.random import choice

from keras.preprocessing.sequence import pad_sequences

from itertools import permutations

def load_text(file_path):
    """
    Load text filename
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
        if id > max_vocab_size - 4:
            break
#    with open("temp.txt", 'w') as f:
#        for key, val in word2id.items():
#            f.write("%s: %s\n" % (key, val))

    return word2id, id2word

def tokenize_sentence(text):
    """
    Tokenize senetences
    Args:
        text
    Returns:
        tokenized senetences (python list of sentences)
    """
    token_sentences = tokenize.sent_tokenize(text)

#    with open ("./tokenize_sentence.tx","w") as t_s:
#        t_s.write('\n'.join(token_sentences) )

    return token_sentences

def strip_punctuation(text):
    """ Remove puctuations from text
    Args:
        text(str)
    Returns
        clean_text(str)
    """
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator)
    return clean_text

def prepare_data(word2id, token_sentences, max_sentence_words = 12 ):
    """Prepares dataset for a longuage model
    Args:
        word2id: dictionary to convert from words to id
        token_sentences: a python array of senetnces
        max_sentence_words: maximim senetnce length
    Return:
        X: sequences of words
        y: Next word in each sequnce
    """
    data = []
    for sentence in token_sentences:
        sentence = strip_punctuation(sentence)
        sentence = sentence.lower()
        sentence_token_words = sentence.split()
        sentence_token_words = ['<BGN>'] + sentence_token_words + ['<EOS>']
        senetnce_size = min(len(sentence_token_words), max_sentence_words)
    #    print(token_words)
        for word_size in range(2, senetnce_size+1):
            token_words = sentence_token_words[: word_size]
            num_pads = max_sentence_words - word_size
            token_words_padded =  ['<PAD>']*num_pads + token_words
            #print(token_words_padded)
            token_words_padded_id = [word2id[word] if word in word2id else word2id['<UNK>'] for word in token_words_padded]
            #senetence_words_id_padded = [word2id[word] for word in senetence_words_padded]
            data.append(token_words_padded_id)

    #print(data)
    data = np.array(data)
    X = data[:, :-1]
    y = data[:,-1]
    return X, y

def create_model(vocab_size, embedding_dim=40):
    """Create keras model
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

def train_model(model, X, y):
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
    model.fit(X, y, epochs=100, verbose=2)
    return model

def evaluate_model(model):
    """
    """
    pass

def generate_text(model, word2id, id2word, vocab_size, max_sentence_words = 12, num_sentences = 1):
    """ generates new senetnces based on longuage models
    Args:
        max_sentence_words: maximum number of words in a senetnce
        num_sentences: number of sentnces to be genarted

    Return:
        text: a paython list of genrated senetnces
    """
    text = []
    for _ in range(num_sentences):
        seq = [word2id['<BGN>']]
        for i in range(12):
            x = pad_sequences([seq], maxlen=max_sentence_words-1, truncating='pre')
            y = model.predict_classes(x, verbose=0)
            y_prob = model.predict(x, verbose=0)
            p = y_prob.reshape(-1,1)[:,0]
            a = np.array(range(0,vocab_size)).reshape(-1,1)[:,0]
            if i == 0:
                predict_word = choice(a, p=p)
            else:
                for k in y:
                    if k != word2id['<UNK>']:
                        predict_word = k

            seq.append(predict_word)
            if predict_word == word2id['<EOS>']:
                break
        new_words = [id2word[id] for id in seq]
        senetnce = ' '.join(new_words)
        text.append(senetnce)
        print(senetnce + '\n')
    return text

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
        window_permutation = [ sentence_words_id[0:i] + list(l) + sentence_words_id[i+window:] for l in permutations(sentence_words_id[i:window+i]) ]
        #print(window_permutation)
        #it_sentence_words_id_permutations = [sentence_words_id[0:i]] + window_permutation + [sentence_words_id[i+window:]]
        for per in window_permutation:
            sentence_words_id_permutations.append(per)

#    print(sentence_words_id_permutations)

#   sentence_words_id_permutations = list(permutations(sentence_words_id))

#    sentence_words_id_permutations = word2id['<BGN>'] + sentence_words_id_permutations + word2id['<EOS>']
    num_permutations = len(sentence_words_id_permutations)
    senrence_size = len(sentence_words_id_permutations[0])
    sentence_words_id_permutations_prob = []
    for sentence_order_index in range(0, num_permutations):
        sentence_order_words_id = list(sentence_words_id_permutations[sentence_order_index])
    #    print(sentence_order_words_id)
        sentence_order_words_id = [word2id['<BGN>']] + sentence_order_words_id + [word2id['<EOS>']]
        P_Word = [1]
        p_sentence = 1
        for word_index in range(1, senrence_size+2):
            seq = sentence_order_words_id[:word_index]
            x = pad_sequences([seq], maxlen=max_sentence_words-1, truncating='pre')
            y = sentence_order_words_id[word_index]
            predict_prob = model.predict(x, verbose=0)

            predict_prob = np.array(predict_prob).reshape(-1,)
            prob = predict_prob[y]
            P_Word.append(prob)
            p_sentence = p_sentence*prob

        sentence_words_id_permutations_prob.append(p_sentence)

    input_sentence_words_id = sentence_words_id_permutations[0]
    input_sentence_prob = sentence_words_id_permutations_prob[0]
    input_sentence_words = [id2word[id] for id in input_sentence_words_id]
    input_sentence = ' '.join(input_sentence_words)
    print(input_sentence)
    print(input_sentence_prob)
    print('\n')


    most_likely_word_order_index = np.argmax(sentence_words_id_permutations_prob)
    most_likely_word_order_prob = sentence_words_id_permutations_prob[most_likely_word_order_index]
    most_likely_words_id_order = sentence_words_id_permutations[most_likely_word_order_index]

    most_likely_words_order = [id2word[id] for id in most_likely_words_id_order]
    most_likely_sentence = ' '.join(most_likely_words_order)
    print(most_likely_sentence)
    print(most_likely_word_order_prob)
    print('\n\n')


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    #ap.add_argument("-m", "--model", type=str, default="knn", help="type of python machine learning model to use")
    ap.add_argument("-p", "--path", type=str, default="./data/English.txt", help="Specify the data path")
    args = vars(ap.parse_args())
    data_path = args["path"]

    data = load_text(data_path)

    cleaned_data = clean_text(data)
    vocab_size = 40000
    word2id, id2word = create_vocabulary(cleaned_data, vocab_size)
    token_senetnces = tokenize_sentence(cleaned_data)
    print(len(token_senetnces))
    max_sentence_words = 12
    X, y = prepare_data(word2id, token_senetnces[:16000], max_sentence_words) #     x_train, y_train, x_test, y_test

    if os.path.isfile("word_lm-40.h5") == False:
        model = create_model(vocab_size)
        model.summary()
        model = train_model(model, X, y)
        model.save('word_lm-40.h5')
        evaluate_model(model)
    else:
        model = load_model('word_lm-40.h5')

    #text = generate_text(model, word2id, id2word, vocab_size, max_sentence_words, 10)
    # This is a test project
    # cat is there in a the room
    # school will we to go
    analyze_senetnce(model, word2id, id2word, vocab_size, input_sentence='it will rain tomorrow')

if __name__ == '__main__':
    main()
