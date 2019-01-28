"""
Word based longuage model
"""

import argparse
from nltk import tokenize
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation
from collections import Counter
import string

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

def create_vocabulary(text, max_vocab_size = 400):
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
    word2id = {'<pad>' : 0, '<UNK>' : 1}
    id2word = {0 : '<pad>', 1 : '<UNK>'}

    for word, _ in freq.most_common():
        id = len(word2id)
        word2id[word] = id
        id2word[id] = word
        if id > max_vocab_size:
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

def prepare_data(cleaned_data):
    """
    """
    pass

def create_model(vocab_size, embedding_dim=40):
    """
    Create modelself.
    Args:
        vocabulary vocab_size
        embedding dimmestion
    Returns:
        model
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        LSTM(70),
        Dense(vocab_size),
        Activation('softmax'),
    ])
    return model

def train_model():
    """
    """
    pass

def evaluate_model():
    """
    """
    pass


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    #ap.add_argument("-m", "--model", type=str, default="knn", help="type of python machine learning model to use")
    ap.add_argument("-p", "--path", type=str, default="./data/English.txt", help="Specify the data path")
    args = vars(ap.parse_args())
    data_path = args["path"]

    data = load_text(data_path)

    cleaned_data = clean_text(data)
    create_vocabulary(cleaned_data)
    tokenize_sentence(cleaned_data)
    prepare_data(cleaned_data) #     x_train, y_train, x_test, y_test
    vocab_size=2000
    create_model(vocab_size)
    train_model()
    evaluate_model()

if __name__ == '__main__':
    main()
