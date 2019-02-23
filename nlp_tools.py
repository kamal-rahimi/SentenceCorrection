
from collections import Counter
from nltk import tokenize
import string

def load_text(file_path):
    """ Load text file
    Args:
        data file path (text file)
    Returns:
        raw text (string)
    """
    with open(file_path, 'r') as f:
        text = f.read()

    return text

def clean_text(raw_text):
    """ Clean text by removing extra spaces between words
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
        text(str): inout text
        max_vocab_size: maximum number of words in the vocabulary
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
        text (str): input text
    Returns:
        tokenized sentences (python list of sentences)
    """
    token_sentences = tokenize.sent_tokenize(text)

    return token_sentences


def strip_punctuations(text):
    """ Remove punctuations from text
    Args:
        text(str)
    Returns
        clean_text(str)
    """
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator)
    return clean_text
