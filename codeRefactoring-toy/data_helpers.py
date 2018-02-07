from string import punctuation
from os import listdir
from gensim.models import Word2Vec

def load_doc(filename):
    file = open(filename, 'r', errors='replace')
    text = file.read()
    file.close()
    return text

def doc_to_lines(doc):
    """
    convert a document into list one line per one element
    (1) only when the line is not null
    """
    total_lines = []
    lines = [i.lower() for i in doc.splitlines() if i]

    return lines

def process_directory(data_path):
    total_lines = []
    for filename in listdir(data_path):
        filepath = data_path + '/' + filename
        doc = load_doc(filepath)
        lines = doc_to_lines(doc)
        total_lines += lines
    return total_lines

def doc_to_clean_lines(filename):
    total_lines = load_doc(filename)
    clean_lines = [i.lower() for i in total_lines.splitlines() if len(i) > 5 if "." in i] # 5개 단어 이상으로 이루어지고 마침표가 있는 문장만 포함

    return clean_lines

def save_list(lines, filename):
    """
    save a token list as a file one token per one line
    """
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def doc_to_vocab_lines(filename, vocab):
    """
    convert clean lines into vocab lines
    (1) only when the tokens of a line is included in the vocabulary
    (2) only when the length of a line is more than 5 after process (1)
    """
    clean_lines = load_doc(filename)
    vocab_lines = []
    for i in clean_lines.splitlines():
        words = i.split()
        words = [word for word in words if word in vocab]
        words = [word for word in words if len(words) >= 5]
        vocab_line = " ".join(words)
        if len(vocab_line):
            vocab_line += "."
            vocab_line = [vocab_line]
            vocab_lines += vocab_line

    return vocab_lines

def load_word2vec(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    lines = file.readlines()[1:]
    for line in lines:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('# word2vec file [{}] is successfully loaded!'.format(filename))
    file.close()
    return vocab,embd
