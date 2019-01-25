"""
Word based longuage model
"""

import argparse

def load_text(file_path):
    """
    Load text filename
    Args:
        data file path (text file)
    return:
        text (string)
    """
    with open(file_path) as f:
        text = f.read()

    return text

def clean_text(data):
    """
    """
    pass

def prepare_data(cleaned_data):
    """
    """
    pass

def create_model():
    """
    """
    pass

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
    prepare_data(cleaned_data) #     x_train, y_train, x_test, y_test
    create_model()
    train_model()
    evaluate_model()

if __name__ == '__main__':
    main()
