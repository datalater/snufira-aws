from string import punctuation
import os
from os import listdir
from collections import Counter
from nltk.corpus import stopwords


def load_doc(filename):
    """
    load the documents
    """
    file = open(filename, 'r', errors='replace')
    text = file.read()
    file.close()
    return text

def clean_doc(doc):
    """
    clean the documents
    (1) exclude punctuation (2) only include alphabet tokens
    (3) exclude stopwords (4) exclude tokens of which length is 1
    """
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def add_doc_to_vocab(filename, vocab):
    """
    update vocabulary using tokens after loading and cleaning the documents
    """
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

def process_docs(directory, vocab):
    """
    iterate all the data directories and update vocabulary
    """
    for file_name in listdir(directory):
        file_path = directory + '/' + file_name
        add_doc_to_vocab(file_path, vocab)

def save_list(lines, filename):
    """
    save a token list as a file one token per one line
    """
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
