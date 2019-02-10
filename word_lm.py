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
    sentence_words = ['<BGN>'] + sentence_words + ['<EOS>']
    print(sentence_words)
    sentence_words_id = [word2id[word] if word in word2id else word2id['<UNK>'] for word in sentence_words]
    tested_words = ['<BGN>']+ ['<EOS>']
    for test_word_in in range(0, len(sentence_words)):
        sentence_words_ids = []
        sentence_words_ids_p = []
        for index in range(1, len(sentence_words)-2):
            P = [1]
            p = 1
            for i in range(1, len(sentence_words)-1):
                seq = sentence_words_id[:i]
                x = pad_sequences([seq], maxlen=max_sentence_words-1, truncating='pre')
                y = sentence_words_id[i]
                predict_prob = model.predict(x, verbose=0)
                predict_prob = np.array(predict_prob).reshape(-1,)
                prob = predict_prob[y]
#                print(prob)
                P.append(prob)
                p = p*prob

#            new_words = [id2word[id] for id in sentence_words_id]
#            senetnce = ' '.join(new_words)
#            print(senetnce)
#
#            print('Senetnce liklihood = ')
#            print(p)
#            print('\n')
            if index==1:
                selection_P = []
                for id_in in range(0, len(sentence_words)):
                    if id2word[sentence_words_id[id_in]] not in tested_words:
                        selection_P.append(P[id_in])
                    else:
                        selection_P.append(1)
                print(P)
                print('\n')
                print(selection_P)
                print('\n')
                least_prob = np.argmin(selection_P)
                sentence_words_id_old = sentence_words_id
                least_prob_word = sentence_words_id.pop(least_prob)
                tested_words.append(id2word[least_prob_word])
            else:
                least_prob_word = sentence_words_id.pop(index)

            sentence_words_id.insert(index+1, least_prob_word)
            sentence_words_ids.append(sentence_words_id_old)
            sentence_words_ids_p.append(p)

        most_likely_senetnce_index = np.argmax(sentence_words_ids_p)
        sentence_words_id = sentence_words_ids[most_likely_senetnce_index]
        new_words = [id2word[id] for id in sentence_words_id]
        senetnce = ' '.join(new_words)
        print(senetnce)
        print('liklihood:')
        print(sentence_words_ids_p[most_likely_senetnce_index])
        print('\n')

"""
    seq_input = sentence_words_id
    seq = []
    seq_input.remove(word2id['<BGN>'])
    #seq_input.remove(word2id['<EOS>'])
    seq.append(word2id['<BGN>'])
    for j in range(0,len(sentence_words)-1):
        x = pad_sequences([seq], maxlen=max_sentence_words-1, truncating='pre')
        predict_prob = model.predict(x, verbose=0)
        predict_prob = np.array(predict_prob).reshape(-1,)
        prob = [predict_prob[k] if k in seq_input else 0 for k in range(0, vocab_size)]
        predict_word = np.argmax(np.array(prob))
        seq.append(predict_word)
        seq_input.remove(predict_word)

    new_words = [id2word[id] for id in seq]
    senetnce = ' '.join(new_words)
    print(senetnce + '\n')

"""

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
    analyze_senetnce(model, word2id, id2word, vocab_size, input_sentence='We will a find solution.')

if __name__ == '__main__':
    main()
