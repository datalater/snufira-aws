(c) JMC 2018

**last updated:** 2018.02.05 (Mon)

---

# 0. Find the data path

```python
import os
len(os.listdir('../bookcorpus/'))
dir_list = [i for i in os.listdir('../bookcorpus/') if not ".tar" in i]
print(dir_list)
data_path = '../bookcorpus/'
# os.path.join(data_path, "hello")
```

+ `data_path` : top-level directory where data exists
+ `dir_list` : children of top-level directory (directories of 16 genres)

# I. Define vocabulary

```python
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords


def load_doc(filename):
    """
    load the documents
    """
    file = open(filename, 'r', errors='ignore')
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

def process_docs(dir_name, vocab):
    """
    iterate all the data directories and update vocabulary
    e.g. print(dir_path) : '../bookcorpus/Adventure'
    e.g. print(file_path) : '../bookcorpus/Adventure/459850.txt'
    """
    dir_path = os.path.join(data_path, dir_name)
    for file_name in listdir(dir_path):
        file_path = dir_path + '/' + file_name
        add_doc_to_vocab(file_path, vocab)

def save_list(lines, filename):
    """
    save a token list as a file one token per one line
    """
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

vocab = Counter()

###################################
#                                 #
# for문으로 모든 디렉터리 순회하는 코드  #
#                                 #
###################################
for i, dir_name in enumerate(dir_list):
    print("{}/{} {}".format(i, len(i), dir_name))
    process_docs(dir_name, vocab)

# print(len(vocab))
# print(vocab.most_common(50))

min_occurence = 1
tokens = [k for k,c in vocab.items() if c >= min_occurence]
print(len(tokens))

save_list(tokens, 'corpusToLines_vocab.txt')
print("\n# 단어 {}개의 [corpusToLines_vocab.txt]로 저장했습니다.".format(len(tokens)))

vocab_filename = 'corpusToLines_vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
print("# 단어 {}개의 [{}]을 [vocab]으로 불러왔습니다.".format(len(vocab), vocab_filename))
```

# II. CorpusToLines

```python
from string import punctuation
from os import listdir
from gensim.models import Word2Vec

def doc_to_lines(doc):
    """
    convert a document into list one line per one element
    (1) only when the line is not null
    """
    total_lines = []
    lines = [i.lower() for i in doc.splitlines() if i]

    return lines

def process_directory(dir_name):
    dir_path = os.path.join(data_path, dir_name)
    total_lines = []
    for file_name in listdir(dir_path):
        file_path = dir_path + '/' + file_name
        doc = load_doc(file_path)
        lines = doc_to_lines(doc)
        total_lines += lines
    return total_lines

###################################
#                                 #
# for문으로 모든 디렉터리 순회하는 코드  #
#                                 #
###################################
for i, dir_name in enumerate(dir_list):
    print("{}/{} {}".format(i, len(i), dir_name))
    sentences = process_directory(dir_name)
    sentences += sentences

save_list(sentences, 'total_lines.txt')
print("\n# 문장 {}개의 [total_lines.txt]로 저장했습니다.".format(len(sentences)))
filename = 'total_lines.txt'
total_lines = load_doc(filename)
total_lines = [i for i in total_lines.splitlines()]
total_vocab = set()
for i in total_lines:
    total_vocab.update(i)
print("# unique words in [total_lines.txt]: [{}]".format(len(total_vocab)))
```

```python
def doc_to_clean_lines(filename):
    """
    clean the total_lines
    (1) only when the length of line is more than 5
    (2) only when the line has a full stop
    """
    total_lines = load_doc(filename)
    clean_lines = [i.lower() for i in total_lines.splitlines() if len(i) > 5 if "." in i]

    return clean_lines

filename = "total_lines.txt"
clean_lines = doc_to_clean_lines(filename)
save_list(clean_lines, 'clean_lines.txt')
print("# 문장 {}개가 [clean_lines.txt]로 저장되었습니다.".format(len(clean_lines)))
filename = 'clean_lines.txt'
clean_lines = load_doc(filename)
clean_lines = [i for i in clean_lines.splitlines()]
clean_vocab = set()
for i in clean_lines:
    clean_vocab.update(i)
print("# unique words in [clean_lines.txt]: [{}]".format(len(clean_vocab)))


def doc_to_vocab_lines(filename):
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

filename = "clean_lines.txt"
vocab_lines = doc_to_vocab_lines(filename)
save_list(vocab_lines, 'vocab_lines.txt')
print("# 문장 {}개가 [vocab_lines.txt]로 저장되었습니다.".format(len(vocab_lines)))
filename = 'vocab_lines.txt'
vocab_lines = load_doc(filename)
vocab_lines = [i for i in vocab_lines.splitlines()]
vocab_vocab = set()
for i in vocab_lines:
    vocab_vocab.update(i)
print("# unique words in [vocab_lines.txt]: [{}]".format(len(vocab_vocab)))
```

# III. word2vec

```python
filename = "vocab_lines.txt"

file = open(filename, 'r', errors='replace')
text = file.read()
file.close()

vocab_lines = [i for i in text.splitlines()]

list_lines = []
for i in vocab_lines:
    i = i.split()
    list_lines.append(i)

print(list_lines[0])
```

```python
sentences = list_lines
print("Total training sentences:{}".format(len(sentences)))

wv_sz = 100
# word2vec 모델을 훈련시킵니다.
model = Word2Vec(sentences, size=wv_sz, window=5, workers=8, min_count=1)
# 모델의 vocabulary size를 요약합니다.
words = list(model.wv.vocab)
print("Vocabulary size: %d" % len(words))
print("Wordvector size: %d" % (wv_sz))
print("Embedding size: {}x{}".format(len(words), wv_sz))

# 모델을 ASCII 포맷으로 저장합니다.
filename = 'fantasy_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)
print("\n# word2vec 파일 [{}]이 저장되었습니다.".format(filename))
```

# IV. Use pre-trained word vector

**encoded_lines to x_data**

```python
import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn

max_length = max([len(s.split()) for s in vocab_lines])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
encoded_lines = np.array(list(vocab_processor.fit_transform(vocab_lines)))
print("-"*80,"# [vocab_lines]가 [encoded_lines]로 인코딩 및 패딩 되었습니다. (max_length:{})".format(max_length), "-"*80, sep='\n')
print("BEFORE: \n{}".format(vocab_lines[0]))
print("\nAFTER: \n{}".format(encoded_lines[0]))

x_data = np.array(list(encoded_lines))
print("\n", "-"*80,"# 최종 [x_data] (max_length: {})".format(max_length), "-"*80, sep='\n')
print("EXAMPLE: \n{}".format(x_data[0]))

vocab_dict = vocab_processor.vocabulary_._mapping
vocab_size = len(vocab_dict.keys())
print("\n", "-"*80,"# 최종 [vocab_dict] (vocab_size: {})".format(vocab_size), "-"*80, sep='\n')
print("EXAMPLE: \n[{}] is mapped to [{}].".format(vocab_lines[0].split()[0], vocab_dict[vocab_lines[0].split()[0]]))
```

**load embedding**

```python
def load_word2vec(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    lines = file.readlines()[1:]
    for line in lines:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded {}!'.format(filename))
    file.close()
    return vocab,embd

filename = 'fantasy_embedding_word2vec.txt'
vocab,embd = load_word2vec(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
```

## V. Build a model using TensorFlow

**hyperparameters**

```python
sequence_length = max_length

input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
```


---
